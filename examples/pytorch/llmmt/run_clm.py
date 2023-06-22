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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

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

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.30.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

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
TEST_SRC_LANG = "en"
TEST_TGT_LANG = "is"

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

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
    
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether load model with int8"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether use PEFT (parameter efficient fine-tuning)"
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    language_pairs: str = field(default="", metadata={"help": "training language pairs"})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_path: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes, truncate the number of test examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    ignore_prompt_token_for_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore the prompt tokens in the loss computation or not."
        },
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum new tokens to generate except the prompt."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Beam size for generation"
            )
        }
    )
    # predict_source_lang: str = field(default="", metadata={"help": "The source language for testing"})
    # predict_target_lang: str = field(default="en", metadata={"help": "The target language for testing"})

    suffix: Optional[str] = field(default="", metadata={"help": "The suffix of the training file."})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

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

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and (training_args.do_train or training_args.do_predict ) and not training_args.overwrite_output_dir:
        last_checkpoint = training_args.output_dir
        # last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # Hash language pairs
    pairs = set(data_args.language_pairs.split(","))
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
    else:
        seen_files =set([])
        train_raw_data, valid_raw_data, test_raw_data = {}, {}, {}
        for pair in pairs:
            src_lang = pair.split("-")[0]
            tgt_lang = pair.split("-")[1]

            # The directory is always "xxen", e.g., deen
            first_lang = src_lang if src_lang != "en" else tgt_lang
            second_lang = "en"
            pair_dir = first_lang + second_lang

            train_file = os.path.join(data_args.data_path, pair_dir, f"train.{first_lang}-{second_lang}-{data_args.suffix}.json")
            valid_file = os.path.join(data_args.data_path, pair_dir, f"valid.{first_lang}-{second_lang}.json")
            test_file = os.path.join(data_args.data_path, pair_dir, f"test.{src_lang}-{tgt_lang}.json")
            
            if not os.path.isfile(train_file):
                logger.info(f"Warning: training file {train_file} does not exist!")
            elif train_file not in seen_files:
                train_raw_data[f"{first_lang}-{second_lang}"] = load_dataset(
                    "json",
                    data_files={"train": train_file},
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    )
            if not os.path.isfile(valid_file):
                logger.info(f"Warning: validation file {valid_file} does not exist!")
            elif valid_file not in seen_files:
                valid_raw_data[f"{first_lang}-{second_lang}"] = load_dataset(
                    "json",
                    data_files={"validation": valid_file},
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    )
            if not os.path.isfile(test_file):
                logger.info(f"Warning: test file {test_file} does not exist!")
            elif test_file not in seen_files:
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

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

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
                padding_side='left', 
                add_eos_token=False
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                **tokenizer_kwargs,
                padding_side='left', 
                add_eos_token=False
            )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )


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


    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama" in model_args.model_name_or_path:
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
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

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = ["translation"]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    padding = "max_length"

    def get_first_non_pad_index(input_tensor):
        input_tensor = torch.tensor(input_tensor)
        assert input_tensor.ndim == 1
        first_non_pad_index = (input_tensor != -100).nonzero(as_tuple=True)[0][0]
        return first_non_pad_index.item()

    def get_prompt(source_lang, target_lang, ex):
        src_fullname = LANG_TABLE[source_lang]
        tgt_fullname = LANG_TABLE[target_lang]
        prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: "
        suffix = f"\n{tgt_fullname}:"
        prompt = prefix + ex[source_lang] + suffix
        return prompt
    
    def check_add_eos(tokenized_inputs):
        if tokenized_inputs.input_ids[0][-1] != tokenizer.eos_token_id:
            for idx in range(len(tokenized_inputs.input_ids)):
                tokenized_inputs.input_ids[idx].append(tokenizer.eos_token_id)
                tokenized_inputs.attention_mask[idx].append(1)
    # def check_remove_eos(tokenized_inputs):
    #     if tokenized_inputs.input_ids[0][-1] == tokenizer.eos_token_id:
    #         for idx in range(len(tokenized_inputs.input_ids)):
    #             tokenized_inputs.input_ids[idx].pop(-1)
    #             tokenized_inputs.attention_mask[idx].pop(-1)

    def tokenize_function_train_eval(examples):
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
        model_inputs = tokenizer(inputs, max_length=model.config.max_length - 1, padding=padding, truncation=True, add_special_tokens=True)
        check_add_eos(model_inputs)
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

    def tokenize_function_test(examples):
        prompts = []
        targets = []
        feature_name = list(examples.keys())[0]
        source_lang, target_lang = feature_name.split("-")
        for ex in examples[feature_name]:
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt(source_lang, target_lang, ex)
                prompts.append(prompt)
                targets.append(prompt + ex[target_lang])
        model_inputs = tokenizer(prompts, max_length=data_args.max_source_length, padding=padding, truncation=True, add_special_tokens=True)
        # check_remove_eos(model_inputs)
        labels = tokenizer(targets, max_length=model.config.max_length - 1, padding=padding, truncation=True, add_special_tokens=True)
        check_add_eos(labels)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            if data_args.ignore_prompt_token_for_loss:
                for idx, prompt in enumerate(prompts):
                    prompt = tokenizer(prompt, max_length=data_args.max_source_length, add_special_tokens=False).input_ids # remove the bos
                    first_non_pad_idx = get_first_non_pad_index(labels["input_ids"][idx])
                    labels["input_ids"][idx][first_non_pad_idx: first_non_pad_idx + len(prompt)] = [-100] * len(prompt) 
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        processed_datasets = []
        for lg_pair, sub_raw_data in train_raw_data.items():
            train_dataset = sub_raw_data["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    tokenize_function_train_eval,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
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
                    tokenize_function_train_eval,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
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
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer test dataset",
                )
            test_datasets[lg_pair] = test_dataset

    metric = evaluate.load("sacrebleu")

    ## MTSEQ2SEQ
    def clean_outputstring(output, key_word):
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
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        global TEST_SRC_LANG, TEST_TGT_LANG
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        for idx in range(10):
            print("------------------------")
            print(decoded_preds[idx])
            print(decoded_labels[idx])
        with open(os.path.join(training_args.output_dir, f"test-{TEST_SRC_LANG}-{TEST_TGT_LANG}"), "w", encoding="utf-8") as f:
            suffix = f"\n{LANG_TABLE[TEST_TGT_LANG]}:"
            for pred in decoded_preds:
                pred = clean_outputstring(pred, suffix)
                f.writelines([pred, "\n"])
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if (training_args.do_eval or training_args.do_predict) and not is_torch_tpu_available() else None,
        # callbacks=[SavePeftModelCallback],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None and not model_args.use_peft:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if model_args.use_peft:
            model.save_pretrained(training_args.output_dir) 
    # Prediction
    if training_args.do_predict:
        trainer.args.prediction_loss_only = False
        for lg_pair, test_dataset in test_datasets.items():
            logger.info(f"*** Prediction for {lg_pair}***")
            TEST_SRC_LANG, TEST_TGT_LANG = lg_pair.split("-")
            metrics = trainer.predict(
                test_dataset=test_dataset, 
                max_new_tokens=data_args.max_new_tokens, 
                num_beams=data_args.num_beams, 
                metric_key_prefix="test"
            )

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

