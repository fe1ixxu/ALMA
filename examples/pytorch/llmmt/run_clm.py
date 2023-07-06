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
from utils.utils import LANG_TABLE, INSTRUCT_PROMPT_DICT
from utils.utils import load_mmt_dataset, get_prompt_mt_instruct, check_add_eos_right_pad, get_first_non_pad_index, clean_outputstring, get_prompt, check_add_eos, load_tokenizer, load_model
from utils.arguments import ModelArguments, DataTrainingArguments
from utils.ul2collator import DataCollatorForUL2
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.30.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    
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
        if data_args.mmt_data_path:
            train_raw_data, valid_raw_data, test_raw_data = load_mmt_dataset(pairs, data_args, model_args, training_args, logger)
        if data_args.instruct_data_path:
            train_instruct_raw_data = load_dataset(
                "json",
                data_files=data_args.instruct_data_path,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    ## load tokenizer
    set_seed(training_args.seed)
    tokenizer = load_tokenizer(data_args, model_args, training_args, logger)
    # Preprocessing the datasets.
    if data_args.instruct_data_path:
        column_names_instruct = ["instruction", "output", "input"]
    if data_args.mmt_data_path:
        column_names_mmt = ["translation"]
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    padding = "max_length"

    #### Decide which prompt used for prompting:
    
    get_prompt_fun = get_prompt_mt_instruct if data_args.instruct_data_path else get_prompt
    def instruct_tokenize_function_left_pad(examples):
        inputs = []
        prompts = []
        prompt_input, prompt_no_input = INSTRUCT_PROMPT_DICT["prompt_input"], INSTRUCT_PROMPT_DICT["prompt_no_input"]
        for instruction, output, input in zip(examples['instruction'], examples['output'], examples['input']):
            ex = {
                "instruction": instruction,
                "input": input,
            }
            prompt = prompt_input.format_map(ex) if input != "" else prompt_no_input.format_map(ex)
            prompts.append(prompt)
            inputs.append(prompt + output)
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

    def tokenize_function_train_eval_left_pad(examples):
        inputs = []
        prompts = []
        for ex in examples["translation"]:
            source_lang, target_lang = list(ex.keys())
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt_fun(source_lang, target_lang, ex)
                prompts.append(prompt)
                inputs.append(prompt + ex[target_lang])
            if f"{target_lang}-{source_lang}" in pairs:
                prompt = get_prompt_fun(target_lang, source_lang, ex)
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
                prompt = get_prompt_fun(source_lang, target_lang, ex)
                prompts.append(prompt)
                inputs.append(prompt + ex[target_lang])
            if f"{target_lang}-{source_lang}" in pairs:
                prompt = get_prompt_fun(target_lang, source_lang, ex)
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

    def tokenize_function_eval_right_pad(examples):
        inputs = []
        prompts = []
        for ex in examples["translation"]:
            source_lang, target_lang = list(ex.keys())
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt_fun(source_lang, target_lang, ex)
                prompts.append(prompt)
                inputs.append(prompt + ex[target_lang])
            if f"{target_lang}-{source_lang}" in pairs:
                prompt = get_prompt_fun(target_lang, source_lang, ex)
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


    def instruct_tokenize_function_right_pad(examples):
        inputs = []
        prompts = []
        prompt_input, prompt_no_input = INSTRUCT_PROMPT_DICT["prompt_input"], INSTRUCT_PROMPT_DICT["prompt_no_input"]
        for instruction, output, input in zip(examples['instruction'], examples['output'], examples['input']):
            ex = {
                "instruction": instruction,
                "input": input,
            }
            prompt = prompt_input.format_map(ex) if input != "" else prompt_no_input.format_map(ex)
            prompts.append(prompt)
            inputs.append(prompt + output)
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

    def tokenize_function_test(examples):
        prompts = []
        targets = []
        feature_name = list(examples.keys())[0]
        source_lang, target_lang = feature_name.split("-")
        for ex in examples[feature_name]:
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt_fun(source_lang, target_lang, ex)
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

    if data_args.right_pad:
        mmt_train_eval_tok_func = tokenize_function_train_eval_right_pad
        instruct_tok_func = instruct_tokenize_function_right_pad
        mmt_eval_tok_func = tokenize_function_eval_right_pad
    else:
        mmt_train_eval_tok_func = tokenize_function_train_eval_left_pad
        instruct_tok_func = instruct_tokenize_function_left_pad
        mmt_eval_tok_func = tokenize_function_train_eval_right_pad
    if training_args.do_train:
        processed_datasets = []
        if data_args.mmt_data_path:
            for lg_pair, sub_raw_data in train_raw_data.items():
                train_dataset = sub_raw_data["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        mmt_train_eval_tok_func,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names_mmt,
                        cache_file_name=f"{os.environ['HF_DATASETS_CACHE']}/{model_args.model_name_or_path}-train-mmt-{lg_pair}-{data_args.suffix}",
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on MMT train dataset",
                    )
                processed_datasets.append(train_dataset)
        if data_args.instruct_data_path:
            train_dataset = train_instruct_raw_data["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    instruct_tok_func,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names_instruct,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on instruct train dataset",
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
                    mmt_eval_tok_func,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names_mmt,
                    cache_file_name=f"{os.environ['HF_DATASETS_CACHE']}/{model_args.model_name_or_path}-valid-mmt-{lg_pair}-{data_args.suffix}",
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
                    cache_file_name=f"{os.environ['HF_DATASETS_CACHE']}/{model_args.model_name_or_path}-test-mmt-{lg_pair}-{data_args.suffix}",
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer test dataset",
                )
            test_datasets[lg_pair] = test_dataset

    metric = evaluate.load("sacrebleu")

    ## Load model
    model = load_model(data_args, model_args, training_args, tokenizer, logger)
    
    if data_args.use_ul2:
        assert data_args.use_prefix_lm, "Must enable use prefix language model"
    collate_fn = DataCollatorForUL2(model, tokenizer) if data_args.use_ul2 else default_data_collator
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=collate_fn 
        # compute_metrics=compute_metrics if (training_args.do_eval or training_args.do_predict) and not is_torch_tpu_available() else None,
        # callbacks=[SavePeftModelCallback],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

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
        else:
            trainer.save_model()  # Saves the tokenizer too for easy upload
    # Prediction
    if training_args.do_predict:
        trainer.args.prediction_loss_only = False
        lg_pairs = sorted(test_datasets.keys()) # make sure each device print in the same order
        for lg_pair in lg_pairs:
            test_dataset = test_datasets[lg_pair]
            src_lang, tgt_lang = lg_pair.split("-")
            logger.info(f"*** Prediction for {lg_pair}***")
            preds, _, _ = trainer.predict(
                test_dataset=test_dataset, 
                max_new_tokens=data_args.max_new_tokens, 
                num_beams=data_args.num_beams, 
                metric_key_prefix="test",
                use_cache=True,
            )

            # Replace -100s used for padding as we can't decode them
            if int(torch.cuda.current_device()) == 0:
                preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

                # Some simple post-processing
                decoded_preds = [pred.strip() for pred in decoded_preds]

                # for idx in range(10):
                #     print("------------------------")
                #     print(decoded_preds[idx])

                with open(os.path.join(training_args.output_dir, f"test-{src_lang}-{tgt_lang}"), "w", encoding="utf-8") as f:
                    suffix = f"\n{LANG_TABLE[tgt_lang]}:" if not data_args.instruct_data_path else "### Response:"
                    for pred in decoded_preds:
                        pred = clean_outputstring(pred, suffix, logger)
                        f.writelines([pred, "\n"])

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

