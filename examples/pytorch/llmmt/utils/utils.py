#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import copy
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import numpy as np

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    LlamaTokenizer,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from peft import PeftModel, PeftConfig
from collections import defaultdict
from transformers.trainer_callback import TrainerCallback
from datasets import concatenate_datasets


# class SavePeftModelCallback(TrainerCallback):
#     def on_save(
#         self,
#         args,
#         state,
#         control,
#         **kwargs,
#     ):
#         checkpoint_folder = os.path.join(
#             args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
#         )       

#         peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
#         kwargs["model"].save_pretrained(peft_model_path)

#         pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
#         try:
#             if os.path.exists(pytorch_model_path):
#                 os.remove(pytorch_model_path)
#         except:
#             pass
#         return control

LANG_TABLE = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "is": "Icelandic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ha": "Hausa",
    "ro": "Romanian",
}

INSTRUCT_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def load_mmt_dataset(pairs, data_args, model_args, training_args, logger):
    seen_files =set([])
    train_raw_data, valid_raw_data, test_raw_data = {}, {}, {}
    for pair in pairs:
        src_lang = pair.split("-")[0]
        tgt_lang = pair.split("-")[1]

        # The directory is always "xxen", e.g., deen
        first_lang = src_lang if src_lang != "en" else tgt_lang
        second_lang = "en"
        pair_dir = first_lang + second_lang

        if first_lang in ["de", "cs", "ru"] or second_lang in ["de", "cs", "ru"]:
            h_suffix = f"-{10000}" if data_args.suffix else ""
        else:
            h_suffix = f"-{data_args.suffix}" if data_args.suffix else ""
        h_suffix = f"-{data_args.suffix}" if data_args.suffix else ""
        train_file = os.path.join(data_args.mmt_data_path, pair_dir, f"train.{first_lang}-{second_lang}{h_suffix}.json")
        valid_file = os.path.join(data_args.mmt_data_path, pair_dir, f"valid.{first_lang}-{second_lang}.json")
        test_file = os.path.join(data_args.mmt_data_path, pair_dir, f"test.{src_lang}-{tgt_lang}.json")
        
        if not os.path.isfile(train_file):
            logger.info(f"Warning: training file {train_file} does not exist!")
        elif train_file not in seen_files and training_args.do_train:
            train_raw_data[f"{first_lang}-{second_lang}"] = load_dataset(
                "json",
                data_files={"train": train_file},
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                )
        if not os.path.isfile(valid_file):
            logger.info(f"Warning: validation file {valid_file} does not exist!")
        elif valid_file not in seen_files and training_args.do_eval:
            valid_raw_data[f"{first_lang}-{second_lang}"] = load_dataset(
                "json",
                data_files={"validation": valid_file},
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                )
        if not os.path.isfile(test_file):
            logger.info(f"Warning: test file {test_file} does not exist!")
        elif test_file not in seen_files and training_args.do_predict:
            test_raw_data[f"{src_lang}-{tgt_lang}"] = load_dataset(
                "json",
                data_files={"test": test_file},
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                )
            test_raw_data[f"{src_lang}-{tgt_lang}"] = test_raw_data[f"{src_lang}-{tgt_lang}"].rename_column("translation", f"{src_lang}-{tgt_lang}")

        seen_files.add(train_file)
        seen_files.add(valid_file)
        seen_files.add(test_file)

    return train_raw_data, valid_raw_data, test_raw_data


def get_first_non_pad_index(input_tensor):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 1
    first_non_pad_index = (input_tensor != -100).nonzero(as_tuple=True)[0][0]
    return first_non_pad_index.item()

def get_first_special_index(input_tensor, special):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 1
    first_pad_index = (input_tensor == special).nonzero(as_tuple=True)[0]
    if len(first_pad_index) > 0:
        return first_pad_index[0].item()
    else:
        return -1

def get_first_non_specical_index(input_tensor, special):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 1
    first_non_pad_index = (input_tensor != special).nonzero(as_tuple=True)[0][0]
    return first_non_pad_index.item()

def get_prompt(source_lang, target_lang, ex):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: "
    suffix = f"\n{tgt_fullname}:"
    prompt = prefix + ex[source_lang] + suffix
    return prompt

def get_prompt_mt_instruct(source_lang, target_lang, ex):
    prompt_input = INSTRUCT_PROMPT_DICT["prompt_input"]
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    instruction = f"Translate this input from {src_fullname} to {tgt_fullname}."
    input = ex[source_lang]
    prompt = prompt_input.format_map({"instruction": instruction, "input": input})
    return prompt


def check_add_eos(tokenized_inputs, tokenizer):
    if tokenized_inputs.input_ids[0][-1] != tokenizer.eos_token_id:
        for idx in range(len(tokenized_inputs.input_ids)):
            tokenized_inputs.input_ids[idx].append(tokenizer.eos_token_id)
            tokenized_inputs.attention_mask[idx].append(1)

def check_add_eos_right_pad(tokenized_inputs, tokenizer):
    for idx in range(len(tokenized_inputs.input_ids)):
        first_non_pad_idx = get_first_special_index(tokenized_inputs.input_ids[idx], tokenizer.pad_token_id)
        tokenized_inputs.input_ids[idx][first_non_pad_idx] = tokenizer.eos_token_id
        tokenized_inputs.attention_mask[idx][first_non_pad_idx] = 1
    

# def check_remove_eos(tokenized_inputs):
#     if tokenized_inputs.input_ids[0][-1] == tokenizer.eos_token_id:
#         for idx in range(len(tokenized_inputs.input_ids)):
#             tokenized_inputs.input_ids[idx].pop(-1)
#             tokenized_inputs.attention_mask[idx].pop(-1)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def clean_outputstring(output, key_word, logger):
    try:
        out = output.split(key_word)[1].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif out[1].strip() != "":
            ## If there is an EOL directly after the suffix, ignore it
            logger.info(f"Detect empty output, we ignore it and move to next EOL: {out[1].strip()}")
            return out[1].strip()
        else:
            logger.info(f"Detect empty output AGAIN, we ignore it and move to next EOL: {out[2].strip()}")
            return out[2].strip()
    except:
        logger.info(f"Can not recover the translation by moving to the next EOL.. Trying move to the next suffix")
        
    try:
        return output.split(key_word)[2].split("\n")[0].strip()
    except:
        logger.info(f"Can not solve the edge case, recover the translation to empty string! The output is {output}")
        return ""

def load_model(data_args, model_args, training_args, tokenizer, logger):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and (training_args.do_train or training_args.do_predict ) and not training_args.overwrite_output_dir:
        last_checkpoint = training_args.output_dir
        # last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # Set seed before initializing model.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
        "max_length": data_args.max_source_length + data_args.max_new_tokens
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    ## Model Loading
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path if last_checkpoint is None or model_args.use_peft else last_checkpoint,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            trust_remote_code=True,
            # load_in_8bit=model_args.load_in_8bit,
        )
        model.generation_config.max_length = data_args.max_source_length + data_args.max_new_tokens
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    if "llama" in model_args.model_name_or_path:
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
    elif "mpt" in model_args.model_name_or_path:
        model.config.pad_token_id = 1
        model.config.bos_token_id = 0
        model.config.eos_token_id = 0
        model.config.attn_config["attn_impl"] = "flash"
        model.generation_config.pad_token_id = 1
        model.generation_config.bos_token_id = 0
        model.generation_config.eos_token_id = 0
        
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    ## PEFT: LORA:
    if model_args.use_peft:
        if last_checkpoint:
            config = PeftConfig.from_pretrained(last_checkpoint)
            model = PeftModel.from_pretrained(model, last_checkpoint)
        else:
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
            
        print_trainable_parameters(model)
    return model
    
def load_tokenizer(data_args, model_args, training_args, logger):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
        
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        if "llama" in model_args.model_name_or_path:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_args.model_name_or_path, 
                **tokenizer_kwargs, 
                padding_side='left' if not data_args.right_pad else "right", 
                add_eos_token=False
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                **tokenizer_kwargs,
                padding_side='left' if not data_args.right_pad else "right", 
                add_eos_token=False
            )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )


    # if tokenizer.pad_token == None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama" in model_args.model_name_or_path:
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.eos_token = "</s>"
        tokenizer.bos_token = "<s>"
    elif "mpt" in model_args.model_name_or_path:
        tokenizer.pad_token_id = 1
        tokenizer.bos_token_id = 0
        tokenizer.eos_token_id = 0
        tokenizer.pad_token = "<|padding|>"

    return tokenizer