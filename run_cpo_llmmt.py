#!/usr/bin/env python
# coding=utf-8

import logging
import copy
import math
import os
import sys
import json
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
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.utils import send_example_telemetry
from collections import defaultdict
from datasets import  interleave_datasets
from utils.trainer_llmmt import LlmmtTrainer
from utils.utils import LANG_TABLE, load_mmt_dataset, preprocess_cpo_data, clean_outputstring, load_tokenizer, load_model, SavePeftModelCallback, get_key_suffix
from utils.arguments import ModelArguments, DataTrainingArguments
from trl import CPOTrainer, CPOConfig

logger = logging.getLogger(__name__)

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CPOConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    # Get the datasets
    pairs = set(data_args.language_pairs.split(","))
    train_raw_data, valid_raw_data, test_raw_data = None, None, None
    seen = set()
    ## load cpo dataset
    train_raw_data = {}
    for pair in pairs:
        src_lang, tgt_lang = pair.split("-")
        first_lang = src_lang if src_lang != "en" else tgt_lang
        second_lang = "en"
        if (first_lang, second_lang) not in seen and training_args.do_train:
            train_raw_data[f"{first_lang}-{second_lang}"] = load_dataset(
                data_args.cpo_data_path,
                f"{first_lang}-{second_lang}",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
                )
        seen.add((first_lang, second_lang))
    
    # load tokenizer
    set_seed(training_args.seed)
    tokenizer = load_tokenizer(data_args, model_args, training_args, logger)

    shots_eval_dict = {}
    if data_args.few_shot_eval_path:
        for lg_pair in test_raw_data.keys():
            pair_shot_path = os.path.join(data_args.few_shot_eval_path, f"shots.{lg_pair}.json")
            if not os.path.isfile(pair_shot_path):
                ValueError(f"Make sure the language pair {lg_pair} is in the few shot eval folder!")
            with open(pair_shot_path) as f:
                shots_eval_dict[lg_pair] = json.load(f)

    # Preprocess data
    train_datasets, eval_datasets, test_datasets = preprocess_cpo_data(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args)

    # Load model
    model = load_model(data_args, model_args, training_args, tokenizer, logger) 

    # Initialize our Trainer
    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        tokenizer=tokenizer,
        callbacks=[SavePeftModelCallback] if model_args.use_peft else None,
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_state()
        if model_args.use_peft:
            if torch.distributed.get_rank() == 0:
                model.save_pretrained(training_args.output_dir) 
        else:
            trainer.save_model()  # Saves the tokenizer too for easy upload

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()