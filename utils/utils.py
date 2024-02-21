#!/usr/bin/env python
# coding=utf-8

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
from datasets import load_dataset, Dataset, DatasetDict

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
from tqdm import tqdm
from trl import AutoModelForCausalLMWithValueHead

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

        if os.path.isfile(pytorch_model_path) and torch.distributed.get_rank() == 0:
            os.remove(pytorch_model_path)
            # create an empty toy file to avoid error in deleting old checkpoints
            open(pytorch_model_path, 'w').close()
        return control

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
    "gu": "Gujarati",
}

## Prefix and suffix for prompt in target language (only from English to target language if the target is non-English)
## Note that prefix and suffix for other languages are only used for zero-shot evaluation of other models.
## ALMA should only use English Prompt.
PREFIX = {
    "de": "Übersetzen Sie dies vom Englischen ins Deutsche:\nEnglisch: ",
    "fr": "Traduisez ceci de l'anglais vers le français :\nAnglais: ",
    "cs": "Přeložte toto z angličtiny do češtiny:\nanglicky: ",
    "is": "Þýddu þetta úr ensku yfir á íslensku:\nEnska: ",
    "zh": "将其从英文翻译成中文：\n英语：",
    "ja": "これを英語から日本語に翻訳してください:\n英語：",
    "ru": "Переведите это с английского на русский:\nАнглийский: ",
    "uk": "Перекладіть це з англійської на українську:\nАнглійська: ",
    "ha": "Fassara wannan daga Turanci zuwa Hausa:\nTuranci: ",
}

SUFFIX = {
    "en": "\nEnglish:",
    "de": "\nDeutsch:",
    "fr": "\nFrançais :",
    "cs": "\nčesky:",
    "is": "\nÍslenska:",
    "zh": "\n中文：",
    "ja": "\n日本語：",
    "ru": "\nРусский:",
    "uk": "\nУкраїнська:",
    "ha": "\nHausa:",
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
                streaming=data_args.streaming,
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
            if data_args.override_test_data_path:
                test_raw_data[f"{src_lang}-{tgt_lang}"] = load_dataset(
                    data_args.override_test_data_path,
                    f"{src_lang}-{tgt_lang}",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    )
            else:
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

def load_a_single_text_file(pairs, data_args, model_args):
    assert len(pairs) == 1, "Specific translation text source file only needs one translation direction!"
    src_lang, tgt_lang = list(pairs)[0].split("-")
    test_raw_data = {}
    pair = f"{src_lang}-{tgt_lang}"
    test_raw_data[pair] = load_dataset(
        'text',
        data_files={"test": data_args.text_test_file},
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        )
    def format_features(example):
        return {pair: {src_lang: example["text"], tgt_lang: ""}}

    test_raw_data[pair] = test_raw_data[pair].map(format_features, remove_columns=["text"])

    return test_raw_data

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

def get_first_special_index_batch(input_tensor, special):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 2
    matches = input_tensor.eq(special).long()
    indices = matches.argmax(dim=1)
    indices[matches.sum(dim=1) == 0] = -1
    return indices 

def get_first_non_specical_index(input_tensor, special):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 1
    first_non_pad_index = (input_tensor != special).nonzero(as_tuple=True)[0][0]
    return first_non_pad_index.item()

# Suffix for splitting and getting the generated sentences
def get_key_suffix(tgt_lang, data_args):
    if data_args.use_target_lang_prompt_eval:
        return SUFFIX[tgt_lang]
    else:
        return f"\n{LANG_TABLE[tgt_lang]}:"

def get_prompt_few_shot(source_lang, target_lang, ex, shots_eval_dict):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    shots = shots_eval_dict[f"{source_lang}-{target_lang}"]
    prefix = f"Translate this from {src_fullname} to {tgt_fullname}:"
    shot_prompt = ""
    for shot in shots:
        shot_src = shot['source']
        shot_tgt = shot['target']
        shot_prompt += f"\n{src_fullname}: " + shot_src + f"\n{tgt_fullname}: " + shot_tgt
    suffix = f"\n{tgt_fullname}:"
    prompt = prefix + shot_prompt + f"\n{src_fullname}: " + ex[source_lang] + suffix
    return prompt

def get_prompt(source_lang, target_lang, ex, shots_eval_dict={}, use_target_lang_prompt_eval=False):
    if len(shots_eval_dict) != 0:
        return get_prompt_few_shot(source_lang, target_lang, ex, shots_eval_dict)
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    if use_target_lang_prompt_eval and target_lang != "en":
        prefix = PREFIX[target_lang]
        suffix = SUFFIX[target_lang]
    else:
        prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: "
        suffix = f"\n{tgt_fullname}:"
    prompt = prefix + ex[source_lang] + suffix
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


def clean_outputstring(output, key_word, logger, split_idx):
    try:
        out = output.split(key_word)[split_idx].split("\n")
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

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
        "max_length": data_args.max_source_length + data_args.max_new_tokens,
        # "norm_type": "low_precision_rmsnorm",
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
    if "mpt" in model_args.model_name_or_path:
        config.attn_config["prefix_lm"] = data_args.use_prefix_lm

    ## Model Loading
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        if model_args.multi_gpu_one_model and not training_args.do_train:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path if last_checkpoint is None else last_checkpoint,
                device_map="auto",
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path if last_checkpoint is None else last_checkpoint,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                trust_remote_code=True,
            )
        model.generation_config.max_length = data_args.max_source_length + data_args.max_new_tokens
        model.generation_config.use_cache = True
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.use_peft:
        if model_args.peft_model_id:
            model = PeftModel.from_pretrained(model, model_args.peft_model_id)
            ## If still need to fine-tune
            for name, param in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad = True
        else:
            config = LoraConfig(
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_rank * 2,
                target_modules=["down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
        print_trainable_parameters(model)

    if "llama" in model_args.model_name_or_path:
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
    elif "BigTranslate" in model_args.model_name_or_path:
        model.config.pad_token_id = 2
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.generation_config.pad_token_id = 2
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2 
    elif "mpt" in model_args.model_name_or_path:
        model.config.pad_token_id = 1
        model.config.bos_token_id = 0
        model.config.eos_token_id = 0
        model.generation_config.pad_token_id = 1
        model.generation_config.bos_token_id = 0
        model.generation_config.eos_token_id = 0
        for name, param in model.named_parameters():
            # To be compatible with AMD cards
            if "norm" in name:
                param.requires_grad = False
        
    return model

def load_tokenizer(data_args, model_args, training_args, logger):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "padding_side": 'left' if not data_args.right_pad else "right",
        "add_eos_token": False,
    }
        
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        if "llama" in model_args.model_name_or_path or "BigTranslate" in model_args.model_name_or_path or "ALMA" in model_args.model_name_or_path:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_args.model_name_or_path, 
                **tokenizer_kwargs, 
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                **tokenizer_kwargs,
            )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if "llama" in model_args.model_name_or_path:
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.eos_token = "</s>"
        tokenizer.bos_token = "<s>"
    elif "Mistral" in model_args.model_name_or_path:
        tokenizer.pad_token_id = 0
    elif "mpt" in model_args.model_name_or_path:
        tokenizer.pad_token_id = 1
        tokenizer.bos_token_id = 0
        tokenizer.eos_token_id = 0
        tokenizer.pad_token = "<|padding|>"

    return tokenizer

def get_preprocessed_data(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args):
    def tokenize_function_train_eval_left_pad(examples):
        inputs = []
        prompts = []
        for ex in examples["translation"]:
            source_lang, target_lang = list(ex.keys())
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt(source_lang, target_lang, ex)
                prompts.append(prompt)
                inputs.append(prompt + ex[target_lang])
            if f"{target_lang}-{source_lang}" in pairs:
                prompt = get_prompt(target_lang, source_lang, ex)
                prompts.append(prompt)
                inputs.append(prompt + ex[source_lang])
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length + data_args.max_new_tokens - 1, padding=padding, truncation=True, add_special_tokens=True)
        check_add_eos(model_inputs, tokenizer)
        labels = copy.deepcopy(model_inputs)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            if data_args.ignore_prompt_token_for_loss:
                for idx, prompt in enumerate(prompts):
                    prompt = tokenizer(prompt, max_length=data_args.max_source_length, add_special_tokens=False).input_ids
                    first_non_pad_idx = get_first_non_pad_index(labels["input_ids"][idx])
                    labels["input_ids"][idx][first_non_pad_idx: first_non_pad_idx + len(prompt)] = [-100] * len(prompt) 
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_function_train_eval_right_pad(examples):
        inputs = []
        prompts = []
        for ex in examples["translation"]:
            source_lang, target_lang = list(ex.keys())
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt(source_lang, target_lang, ex)
                prompts.append(prompt)
                inputs.append(prompt + ex[target_lang])
            if f"{target_lang}-{source_lang}" in pairs:
                prompt = get_prompt(target_lang, source_lang, ex)
                prompts.append(prompt)
                inputs.append(prompt + ex[source_lang])
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length + data_args.max_new_tokens, padding=padding, truncation=True, add_special_tokens=True)
        check_add_eos_right_pad(model_inputs, tokenizer)
        labels = copy.deepcopy(model_inputs)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if data_args.use_prefix_lm:
            assert data_args.ignore_prompt_token_for_loss
            model_inputs["prefix_mask"] = []

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            if data_args.ignore_prompt_token_for_loss:
                for idx, prompt in enumerate(prompts):
                    prompt = tokenizer(prompt, max_length=data_args.max_source_length, add_special_tokens=False).input_ids
                    labels["input_ids"][idx][: len(prompt)] = [-100] * len(prompt) 
                    if data_args.use_prefix_lm:
                        prefix_mask = [0] * len(model_inputs["attention_mask"][idx])
                        prefix_mask[: len(prompt)] = [1] * len(prompt)
                        model_inputs["prefix_mask"].append(prefix_mask)
                    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_function_train_mono(examples):
        if data_args.use_prefix_lm:
            inputs = {"input_ids": [], "attention_mask": [], "prefix_mask": []}      
        else:
            inputs = {"input_ids": [], "attention_mask": []}
        block_size = data_args.max_source_length + data_args.max_new_tokens
        for ex in examples["translation"]:
            lang1, lang2 = list(ex.keys())
            lang = lang1 if lang1 != "en" else lang2

            for lang in [lang1, lang2]:
                if ex[lang] == "":
                    continue
                _input = tokenizer(ex[lang], max_length=4096, add_special_tokens=True)
                _input['input_ids'].append(tokenizer.eos_token_id)
                _input['attention_mask'].append(1)
                if data_args.use_prefix_lm:
                    _input['prefix_mask'] = [0] * len(_input['attention_mask'])
                    inputs["prefix_mask"].append(_input['prefix_mask'])
                inputs["input_ids"].append(_input['input_ids'])
                inputs['attention_mask'].append(_input['attention_mask'])
            
        
        concatenated_inputs = {k: list(chain(*inputs[k])) for k in inputs.keys()}
        total_length = len(concatenated_inputs[list(inputs.keys())[0]])
        total_length = (total_length // block_size) * block_size

        model_inputs = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_inputs.items()
        }

        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
        return model_inputs

    def tokenize_function_train_oscar_mono(examples):
        if data_args.use_prefix_lm:
            inputs = {"input_ids": [], "attention_mask": [], "prefix_mask": []}      
        else:
            inputs = {"input_ids": [], "attention_mask": []}
        block_size = data_args.max_source_length + data_args.max_new_tokens
        for ex in examples["text"]:
            _input = tokenizer(ex, max_length=4096, add_special_tokens=True)
            _input['input_ids'].append(tokenizer.eos_token_id)
            _input['attention_mask'].append(1)
            if data_args.use_prefix_lm:
                _input['prefix_mask'] = [0] * len(_input['attention_mask'])
                inputs["prefix_mask"].append(_input['prefix_mask'])
            inputs["input_ids"].append(_input['input_ids'])
            inputs['attention_mask'].append(_input['attention_mask'])
            
        
        concatenated_inputs = {k: list(chain(*inputs[k])) for k in inputs.keys()}
        total_length = len(concatenated_inputs[list(inputs.keys())[0]])
        total_length = (total_length // block_size) * block_size

        model_inputs = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_inputs.items()
        }

        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
        return model_inputs

    def tokenize_function_test(examples):
        prompts = []
        targets = []
        feature_name = list(examples.keys())[0]
        source_lang, target_lang = feature_name.split("-")
        for ex in examples[feature_name]:
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt(source_lang, target_lang, ex, shots_eval_dict, data_args.use_target_lang_prompt_eval)
                prompts.append(prompt)
                targets.append(prompt + ex[target_lang])
        original_padding_side = tokenizer.padding_side
        if original_padding_side != "left":
            tokenizer.padding_side = "left"
        model_inputs = tokenizer(prompts, max_length=data_args.max_source_length, padding=padding, truncation=True, add_special_tokens=True)
        tokenizer.padding_side = original_padding_side
        if data_args.use_prefix_lm:
            model_inputs["prefix_mask"] = []
            for idx, prompt in enumerate(prompts):
                prefix_mask = model_inputs["attention_mask"][idx]
                model_inputs["prefix_mask"].append(prefix_mask)
        return model_inputs

    
    # Preprocessing the datasets.
    if data_args.mmt_data_path or data_args.mono_data_path:
        column_names_mmt = ["translation"]
    if data_args.oscar_data_path:
        column_name_oscar = ["id", "meta", "text"]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    padding = "max_length"

    train_datasets, eval_datasets, test_datasets = None, None, None
    mmt_train_eval_tok_func = tokenize_function_train_eval_right_pad if data_args.right_pad else tokenize_function_train_eval_left_pad
    
    if training_args.do_train:
        processed_datasets = []
        if data_args.mmt_data_path:
            for lg_pair, sub_raw_data in train_raw_data.items():
                train_dataset = sub_raw_data["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    if not data_args.streaming:
                        train_dataset = train_dataset.map(
                            mmt_train_eval_tok_func,
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=column_names_mmt,
                            cache_file_name=f"{os.environ['HF_DATASETS_CACHE']}/{model_args.model_name_or_path.split('/')[-1]}-train-mmt-{lg_pair}-{data_args.language_pairs}-{data_args.suffix}",
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc="Running tokenizer on MMT train dataset",
                        )
                    else:
                        train_dataset = train_dataset.map(
                            mmt_train_eval_tok_func,
                            batched=True,
                            remove_columns=column_names_mmt,
                        )    
                processed_datasets.append(train_dataset)

        if data_args.mono_data_path:
            train_dataset = train_raw_data['train']
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                if not data_args.streaming:
                    train_dataset = train_dataset.map(
                        tokenize_function_train_mono,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names_mmt,
                        cache_file_name=f"{os.environ['HF_DATASETS_CACHE']}/{model_args.model_name_or_path.split('/')[-1]}-{data_args.mono_data_path.split('/')[-1]}",
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on monolingual train dataset",
                    )
                else:
                    train_dataset = train_dataset.map(
                        tokenize_function_train_mono,
                        batched=True,
                        remove_columns=column_names_mmt,
                    )
            processed_datasets.append(train_dataset)
        if data_args.oscar_data_path:
            train_dataset = train_raw_data
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    tokenize_function_train_oscar_mono,
                    batched=True,
                    remove_columns=column_name_oscar,
                )
            processed_datasets.append(train_dataset)
        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)
        
    if training_args.do_eval:
        processed_datasets = []
        for lg_pair, sub_raw_data in valid_raw_data.items():
            eval_dataset = sub_raw_data["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    mmt_train_eval_tok_func,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names_mmt,
                    cache_file_name=f"{os.environ['HF_DATASETS_CACHE']}/{model_args.model_name_or_path.split('/')[-1]}-valid-mmt-{lg_pair}-{data_args.language_pairs}-{data_args.suffix}",
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer valid dataset",
                )
            processed_datasets.append(eval_dataset)
        eval_datasets = concatenate_datasets(processed_datasets)
        eval_datasets = eval_datasets.shuffle(seed=training_args.seed)

    if training_args.do_predict:
        test_datasets = {}
        for lg_pair, sub_raw_data in test_raw_data.items():
            test_dataset = sub_raw_data["test"]
            if data_args.max_test_samples is not None:
                max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                test_dataset = test_dataset.select(range(max_test_samples))
            with training_args.main_process_first(desc="test dataset map pre-processing"):
                test_dataset = test_dataset.map(
                    tokenize_function_test,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=[lg_pair],
                    cache_file_name=f"{os.environ['HF_DATASETS_CACHE']}/{model_args.model_name_or_path.split('/')[-1]}-test-mmt-{lg_pair}-{data_args.language_pairs}-{data_args.suffix}",
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer test dataset",
                )
            test_datasets[lg_pair] = test_dataset
    return train_datasets, eval_datasets, test_datasets

def preprocess_cpo_data(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args):
    def get_chosen_reject(example, target_lang):
        sys1_score_key = f"gpt4_{target_lang}_{data_args.cpo_scorer}"
        sys2_score_key = f"alma_{target_lang}_{data_args.cpo_scorer}"
        ref_score_key = f"ref_{target_lang}_{data_args.cpo_scorer}"

        sys1_output_key = f"gpt4_{target_lang}"
        sys2_output_key = f"alma_{target_lang}"
        ref_output_key = target_lang

        # sys_key = "sys_" + target_lang
        # gold_key = target_lang

        # Human eval
        if "Delta" in example and example["Delta"] != 0:
            if example["Delta"] > 0:
                return example[sys1_output_key], example[sys2_output_key]
            else:
                return example[sys2_output_key], example[sys1_output_key]

        # Defining the sentences and their scores
        sentences = [example[ref_output_key], example[sys1_output_key], example[sys2_output_key]]
        scores = [example[ref_score_key], example[sys1_score_key], example[sys2_score_key]]

        # Finding the indexes for the highest and lowest scores
        highest_score_index = scores.index(max(scores))
        lowest_score_index = scores.index(min(scores))

        # Assigning the corresponding sentences
        highest_score_sentence = sentences[highest_score_index]
        lowest_score_sentence = sentences[lowest_score_index]
        return highest_score_sentence, lowest_score_sentence
            
    def meet_requirements(prompt_tok, example, target_lang):
        # if prompt is too long
        if len(prompt_tok) > data_args.max_source_length:
            return False

        # if the order is fixed, e.g., it has to be en->de
        if "required_directions" in example and example["required_directions"] != "":
            tgt = example["required_directions"].split("-")[1]
            if tgt != target_lang:
                return False
        return True 

    def cpo_prompt_function(examples):
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
        for ex in examples["translation"]:
            source_lang, target_lang = ex["language_pair"].split("-")
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt(source_lang, target_lang, ex)
                prompt_tok = tokenizer(prompt, max_length=data_args.max_source_length, padding=True, truncation=True, add_special_tokens=True).input_ids
                if meet_requirements(prompt_tok, ex, target_lang):
                    new_examples["prompt"].append(prompt)
                    chosen, rejected = get_chosen_reject(ex, target_lang)
                    new_examples["chosen"].append(chosen)
                    new_examples["rejected"].append(rejected)
            if f"{target_lang}-{source_lang}" in pairs:
                prompt = get_prompt(target_lang, source_lang, ex)
                prompt_tok = tokenizer(prompt, max_length=data_args.max_source_length, padding=True, truncation=True, add_special_tokens=True).input_ids
                if meet_requirements(prompt_tok, ex, source_lang):
                    new_examples["prompt"].append(prompt)
                    chosen, rejected = get_chosen_reject(ex, source_lang)
                    new_examples["chosen"].append(chosen)
                    new_examples["rejected"].append(rejected)
        return new_examples
    
    # Preprocessing the datasets.
    train_datasets, eval_datasets, test_datasets = None, None, None
    if training_args.do_train:
        processed_datasets = []
        for lg_pair, sub_raw_data in train_raw_data.items():
            train_dataset = sub_raw_data["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            with training_args.main_process_first(desc="CPO train dataset map pre-processing"):
                if not data_args.streaming:
                    train_dataset = train_dataset.map(
                        cpo_prompt_function,
                        batched=True,
                        batch_size=1,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=["translation"],
                        cache_file_name=f"{os.environ['HF_DATASETS_CACHE']}/{model_args.model_name_or_path.split('/')[-1]}-train-mmt-{lg_pair}-{data_args.language_pairs}-{data_args.suffix}-CPO",
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running CPO preprocessing",
                    )
                else:
                    train_dataset = train_dataset.map(
                        cpo_prompt_function,
                        batched=True,
                        batch_size=1,
                        remove_columns=["translation"],
                    )    
            processed_datasets.append(train_dataset)
        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)        

    return train_datasets, eval_datasets, test_datasets
