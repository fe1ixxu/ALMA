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
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
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
from datasets import concatenate_datasets, interleave_datasets
from utils.trainer_llmmt import LlmmtTrainer
from utils.utils import LANG_TABLE, load_mmt_dataset, get_preprocessed_data, clean_outputstring, load_tokenizer, load_model, SavePeftModelCallback, get_key_suffix
from utils.arguments import ModelArguments, DataTrainingArguments
from utils.ul2collator import DataCollatorForUL2

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
    send_example_telemetry("run_llmmt", model_args, data_args)

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
    if data_args.mmt_data_path:
        train_raw_data, valid_raw_data, test_raw_data = load_mmt_dataset(pairs, data_args, model_args, training_args, logger)
    if data_args.mono_data_path:
        train_raw_data = load_dataset(
            "json",
            data_files=data_args.mono_data_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
    if data_args.oscar_data_path:
        oscar_langs = data_args.oscar_data_lang.split(",")
        if data_args.interleave_probs:
            interleave_probs = [float(p) for p in data_args.interleave_probs.split(",")]
        else:
            interleave_probs = [1/len(oscar_langs)] * len(oscar_langs)
        oscar_langs = [x for x, _ in sorted(zip(oscar_langs, interleave_probs), key=lambda zippair: zippair[1])]
        interleave_probs = sorted(interleave_probs)
        train_raw_data = []
        for lg in oscar_langs:
            train_raw_data.append(
                load_dataset(
                    data_args.oscar_data_path,
                    lg,
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=data_args.streaming,
                )['train']
            )
        train_raw_data = interleave_datasets(train_raw_data, probabilities=interleave_probs, seed=training_args.seed, stopping_strategy="all_exhausted")
    
    # load tokenizer
    set_seed(training_args.seed)
    tokenizer = load_tokenizer(data_args, model_args, training_args, logger)
    if data_args.use_ul2:
        assert data_args.use_prefix_lm, "Must enable use prefix language model"

    shots_eval_dict = {}
    if data_args.few_shot_eval_path:
        for lg_pair in test_raw_data.keys():
            pair_shot_path = os.path.join(data_args.few_shot_eval_path, f"shots.{lg_pair}.json")
            if not os.path.isfile(pair_shot_path):
                ValueError(f"Make sure the language pair {lg_pair} is in the few shot eval folder!")
            with open(pair_shot_path) as f:
                shots_eval_dict[lg_pair] = json.load(f)

    train_datasets, eval_datasets, test_datasets = get_preprocessed_data(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args)
    metric = evaluate.load("sacrebleu")

    # Load model
    model = load_model(data_args, model_args, training_args, tokenizer, logger)
    collate_fn = DataCollatorForUL2(model, tokenizer) if data_args.use_ul2 else default_data_collator
    
    # Initialize our Trainer
    trainer = LlmmtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        callbacks=[SavePeftModelCallback] if model_args.use_peft else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

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

                for idx in range(data_args.display_num_translations):
                    print("------------------------")
                    print(decoded_preds[idx])

                with open(os.path.join(training_args.output_dir, f"test-{src_lang}-{tgt_lang}{data_args.suffix_eval_file}"), "w", encoding="utf-8") as f:
                    suffix = get_key_suffix(tgt_lang, data_args)
                    if len(shots_eval_dict) != 0:
                        split_idx = len(shots_eval_dict[lg_pair]) + 1
                    else:
                        split_idx = 1
                    for pred in decoded_preds:
                        pred = clean_outputstring(pred, suffix, logger, split_idx)
                        f.writelines([pred, "\n"])

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

