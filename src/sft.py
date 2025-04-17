import logging
import random
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from datasets import concatenate_datasets, load_dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from sklearn.manifold import TSNE
from umap import UMAP
from tqdm.notebook import tqdm
import os
import torch
import json
from typing import List, Optional
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
#import wandb
from dotenv import dotenv_values
from huggingface_hub import login
from fire import Fire


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# config
RANDOM_SEED = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def train(
    # model/data params
    base_model: str = "",
    new_model:str = "", 
    train_data_path: str = "",
    valid_data_path: str = "",
    load_in_8bit: bool = True,
    output_dir: str = "./logs",
    continuous_correction: bool = False,
    saved_full_model_path: Optional[str] = None, # load the full saved peft model
    ################################################################################
    # QLoRA parameters
    ################################################################################
    lora_r: int = 32, # LoRA attention dimension
    lora_alpha: int = 16, # Alpha parameter for LoRA scaling
    lora_dropout: float = 0.1, # Dropout probability for LoRA layers
    ################################################################################
    # bitsandbytes parameters
    ################################################################################
    use_4bit: bool = True,# Activate 4-bit precision base model loading
    bnb_4bit_compute_dtype: str = "float16", # Compute dtype for 4-bit base models
    bnb_4bit_quant_type: str = "nf4",  # Quantization type (fp4 or nf4)
    use_nested_quant :bool = False, # Activate nested quantization for 4-bit base models (double quantization)
    ################################################################################
    # TrainingArguments parameters
    ################################################################################
    num_train_epochs: int = 15, # Number of training epochs tuning 4.6=4651/1004
    fp16: bool = False,
    bf16: bool = False, # set bf16 to True with an A100
    per_device_train_batch_size: int = 2, # Batch size per GPU for training
    per_device_eval_batch_size: int = 2, # Batch size per GPU for evaluation
    gradient_accumulation_steps: int = 2, # Number of update steps to accumulate the gradients for
    # per_device_train_batch_size * device_num * gradient_accumulation_steps = batch_sizeのはず
    gradient_checkpointing: bool = True, # Enable gradient checkpointing
    max_grad_norm: int = 0.3, # Maximum gradient normal (gradient clipping)
    learning_rate: float = 5e-4, # Initial learning rate (AdamW optimizer)
    weight_decay: float = 0.001, # Weight decay to apply to all layers except bias/LayerNorm weights
    optim: str = "paged_adamw_32bit", # Optimizer to use
    lr_scheduler_type: str = "linear", # "cosine" # Learning rate schedule
    max_steps: int = -1, # Number of training steps (overrides num_train_epochs)
    warmup_ratio: float = 0.03, # Ratio of steps for a linear warmup (from 0 to learning rate)
    group_by_length: bool = True, # Group sequences into batches with same length, Saves memory and speeds up training considerably
    save_strategy: str = "steps",
    evaluation_strategy: str = "steps",
    save_steps: int = 200, # Save checkpoint every X updates steps
    eval_steps: int = 100, # When load_best_model_at_end set to True, the parameters save_strategy needs to be the same as evaluation_strategy, and in the case it is “steps”, save_steps must be a round multiple of eval_step
    save_total_limit: int = 3,
    load_best_model_at_end: bool =False, # store best model on evaluation score 
    logging_steps = 25, # Log every X updates steps
    ################################################################################
    # SFT parameters
    ################################################################################
    max_seq_length = None, # Maximum sequence length to use
    packing:bool = False, # Pack multiple short examples in the same input sequence to increase efficiency
    device_map: str = "auto", # Load gpu setting on ABCI
    ################################################################################
    # huggingface parameters
    ################################################################################
    upload_to_huggingface: bool = True,
    ################################################################################
    # wandb parameters
    ################################################################################
    use_wandb: bool = False,
    wandb_project: str = "metano2",
    wandb_run_name: str = "default_run",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    ):
    '''
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
        )
    '''

    # Load data
    train_data = train_data_path
    val_data = valid_data_path

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype) 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )



    # Load base model
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        #load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        report_to="wandb" if use_wandb else "tensorboard",
        run_name=wandb_run_name if use_wandb else None,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        #tokenizer=tokenizer, deprecated
        dataset_text_field="meta_sentence",
        #formatting_func=formatting_prompts_func_for_solver if 'solver' in prompt_template else formatting_prompts_func_for_L2T,    
        packing=packing, #formatting_func使うならfalseにする
        #data_collator=collator, # 学習対象を回答部分に限定する v2
        args=training_arguments,
        #compute_metrics=compute_metrics
    )
    # Train model
    trainer.train()
    # Save trained model
    trainer.model.save_pretrained(new_model)

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Reload model in FP16 and merge it with LoRA weights
    base_model = LlamaForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()


    if upload_to_huggingface:
        config = dotenv_values(".env")
        login(token=config['HUGGINGFACE_TOKEN_W'], write_permission=True)

        model.push_to_hub(new_model, use_temp_dir=True)
        tokenizer.push_to_hub(new_model, use_temp_dir=True)

def main(dataset_type, position=None, base_model_name=None, meta_rep_known=None, meta_rep_unknown=None, meta_rep_others=None):
    if dataset_type == "known_unknown":
        #meta_rep_known, meta_rep_unknown
        dd = load_dataset(f"kenken6696/ALCUNA_meta_affirmative_{meta_rep_known}_{meta_rep_unknown}")
        dd_fix_head = dd['meta_position_head'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_middle = dd['meta_position_middle'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_tail = dd['meta_position_tail'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_head.push_to_hub(f"ALCUNA_meta_affirmative_{meta_rep_known}_{meta_rep_unknown}_for_fix_head_train")
        dd_fix_middle.push_to_hub(f"ALCUNA_meta_affirmative_{meta_rep_known}_{meta_rep_unknown}_for_fix_middle_train")
        dd_fix_tail.push_to_hub(f"ALCUNA_meta_affirmative_{meta_rep_known}_{meta_rep_unknown}_for_fix_tail_train")
        dd_dict = {'head':dd_fix_head, 'middle':dd_fix_middle, 'tail':dd_fix_tail}

        if base_model_name == "NousResearch/Llama-3.2-1B":
            for position in ['head', 'middle', 'tail']:
                train(
                    base_model = base_model_name,
                    new_model = f"Llama-3.2-1B_{meta_rep_known}_{meta_rep_unknown}_fix_{position}", 
                    train_data_path = dd_dict[position]['train'],
                    valid_data_path= dd_dict[position]['test'],
                )
        elif base_model_name == "meta-llama/Llama-3.2-3B":
            if position not in ['head', 'middle', 'tail']:
                raise Exception(f'position{position} need to set in [head, middle, tail]')
            train(
                base_model = base_model_name,
                new_model = f"Llama-3.2-3B_{meta_rep_known}_{meta_rep_unknown}_fix_{position}", 
                train_data_path = dd_dict[position]['train'],
                valid_data_path= dd_dict[position]['test'],
            )
    elif dataset_type == "known_unknown_others":
        # meta_rep_known = 'known', meta_rep_unknown = 'unknown', meta_rep_others = 'boring'
        dd = load_dataset(f"kenken6696/ALCUNA_meta_affirmative_{meta_rep_known}_{meta_rep_unknown}_{meta_rep_others}")
        dd_fix_head = dd['meta_position_head'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_middle = dd['meta_position_middle'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_tail = dd['meta_position_tail'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_head.push_to_hub(f"ALCUNA_meta_affirmative_{meta_rep_known}_{meta_rep_unknown}_{meta_rep_others}_for_fix_head_train")
        dd_fix_middle.push_to_hub(f"ALCUNA_meta_affirmative_{meta_rep_known}_{meta_rep_unknown}_{meta_rep_others}_for_fix_middle_train")
        dd_fix_tail.push_to_hub(f"ALCUNA_meta_affirmative_{meta_rep_known}_{meta_rep_unknown}_{meta_rep_others}_for_fix_tail_train")
        dd_dict = {'head':dd_fix_head, 'middle':dd_fix_middle, 'tail':dd_fix_tail}

        if base_model_name == "NousResearch/Llama-3.2-1B":
            for position in ['head', 'middle', 'tail']:
                train(
                    base_model = base_model_name,
                    new_model = f"Llama-3.2-1B_{meta_rep_known}_{meta_rep_unknown}_{meta_rep_others}_fix_{position}", 
                    train_data_path = dd_dict[position]['train'],
                    valid_data_path= dd_dict[position]['test'],
                )
        elif base_model_name == "meta-llama/Llama-3.2-3B":
            if position not in ['head', 'middle', 'tail']:
                raise Exception(f'position{position} need to set in [head, middle, tail]')
            train(
                base_model = base_model_name,
                new_model = f"Llama-3.2-3B_{meta_rep_known}_{meta_rep_unknown}_{meta_rep_others}_fix_{position}", 
                train_data_path = dd_dict[position]['train'],
                valid_data_path= dd_dict[position]['test'],
            )
    elif dataset_type == "4x3":
        dd = load_dataset(f"kenken6696/ALCUNA_meta_affirmative_4x3")
        dd_fix_head = dd['meta_position_head'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_middle = dd['meta_position_middle'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_tail = dd['meta_position_tail'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_head.push_to_hub(f"ALCUNA_meta_affirmative_4x3_for_fix_head_train")
        dd_fix_middle.push_to_hub(f"ALCUNA_meta_affirmative_4x3_for_fix_middle_train")
        dd_fix_tail.push_to_hub(f"ALCUNA_meta_affirmative_4x3_for_fix_tail_train")
        dd_dict = {'head':dd_fix_head, 'middle':dd_fix_middle, 'tail':dd_fix_tail}

        if base_model_name == "NousResearch/Llama-3.2-1B":
            for position in ['head', 'middle', 'tail']:
                train(
                    base_model = base_model_name,
                    new_model = f"Llama-3.2-1B_4x3_fix_{position}", 
                    train_data_path = dd_dict[position]['train'],
                    valid_data_path= dd_dict[position]['test'],
                )
        elif base_model_name == "meta-llama/Llama-3.2-3B":
            if position not in ['head', 'middle', 'tail']:
                raise Exception(f'position{position} need to set in [head, middle, tail]')
            train(
                base_model = base_model_name,
                new_model = f"Llama-3.2-3B_4x3_fix_{position}", 
                train_data_path = dd_dict[position]['train'],
                valid_data_path= dd_dict[position]['test'],
            )

    elif dataset_type == "4_mix":
        dd = load_dataset(f"kenken6696/ALCUNA_meta_affirmative_4_mix_position")
        dd_fix_mix = dd['meta_position_mix'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_mix.push_to_hub(f"ALCUNA_meta_affirmative_4_mix_position_train")

        train(
            # model/data params
            base_model = "NousResearch/Llama-3.2-1B",
            new_model = f"Llama-3.2-1B_4_mix_positon", 
            train_data_path = dd_fix_mix['train'],
            valid_data_path= dd_fix_mix['test'],
            )

        train(
            # model/data params
            base_model = "meta-llama/Llama-3.2-3B",
            new_model = f"Llama-3.2-3B_4_mix_positon", 
            train_data_path = dd_fix_mix['train'],
            valid_data_path= dd_fix_mix['test'],
            )
    elif dataset_type == "4x3_mix":
        dd = load_dataset(f"kenken6696/ALCUNA_meta_affirmative_4x3_mix_position")
        dd_fix_mix = dd['meta_position_mix'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        dd_fix_mix.push_to_hub(f"ALCUNA_meta_affirmative_4x3_mix_position_train")
        train(
            # model/data params
            base_model = "NousResearch/Llama-3.2-1B",
            new_model = f"Llama-3.2-1B_4x3_mix_positon", 
            train_data_path = dd_fix_mix['train'],
            valid_data_path= dd_fix_mix['test'],
            )

        train(
            # model/data params
            base_model = "meta-llama/Llama-3.2-3B",
            new_model = f"Llama-3.2-3B_4x3_mix_positon", 
            train_data_path = dd_fix_mix['train'],
            valid_data_path= dd_fix_mix['test'],
            )
    else:
        raise Exception('dataset_type is excehibited')
    
if __name__ == "__main__":
    Fire(main)