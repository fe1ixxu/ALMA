#!/bin/bash
# source ~/.bashrc
# conda activate llmmt

# model: 
# opt-6.7b: facebook/opt-6.7b; covering languages: UNKOWN
#           overlapping: UNKOWN
# llama-7b: decapoda-research/llama-7b-hf #covering language:bg, ca, cs, da, de, en, es, fr, hr, hu, it, nl, pl, pt, ro, ru, sl, sr, sv, uk.
#           overlapping: cs, de, fr, en, ru, uk, 
# falcon-7b: tiiuae/falcon-7b
# BLOOM-7b: bigscience/bloom-7b1
# mpt-7b: mosaicml/mpt-7b
# falcon-7b-instruct: tiiuae/falcon-7b-instruct
# mpt-7b-instruct: mosaicml/mpt-7b-instruct
########################################
########################################
export HF_DATASETS_CACHE="/home/aiscuser/huggingface_cache/datasets"

suffix=${1:-10000}
export WANDB_PROJECT=LLMMT-pre
export WANDB_NAME=our_alpaca-cosine-bf16
# OUTPUT_DIR=./tmp/full-ft-mmt
# accelerate launch --config_file deepspeed_train_config.yaml \
#      run_clm.py \
#     --model_name_or_path facebook/opt-125m \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --language_pairs en-de,de-en \
#     --load_best_model_at_end \
#     --fp16 \
#     --suffix ${suffix} \
#     --mmt_data_path /home/aiscuser/filtered_wmt22/ \
#     --learning_rate 0.0001 \
#     --weight_decay 0.1 \
#     --gradient_accumulation_steps 1 \
#     --lr_scheduler_type inverse_sqrt \
#     --warmup_ratio 0.3 \
#     --ignore_pad_token_for_loss \
#     --ignore_prompt_token_for_loss \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --evaluation_strategy steps \
#     --eval_steps 0.1 \
#     --save_strategy steps \
#     --save_steps 0.2 \
#     --save_total_limit 2 \
#     --logging_strategy steps \
#     --logging_steps 0.05 \
#     --output_dir ${OUTPUT_DIR} \
#     --num_train_epochs 3 \
#     --predict_with_generate \
#     --prediction_loss_only \
#     --max_new_tokens 256 \
#     --max_source_length 256 \
#     --seed 42 \
#     --overwrite_output_dir \
#     --report_to wandb



### Instruct Tuning:
accelerate launch --config_file deepspeed_train_config.yaml \
     run_clm.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --do_train \
    --bf16 \
    --instruct_data_path /home/aiscuser/stanford_alpaca/alpaca_data.json \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --output_dir ./tmp/test \
    --num_train_epochs 3 \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --overwrite_output_dir \
    --report_to wandb

        # --overwrite_cache \
        # facebook/opt-125m
# /home/aiscuser/.cache/huggingface/datasets/json/default-fc0d51b772e14738/0.0.0/e347ab1c932092252e717ff3f949105a4dd28b27e842dd53157d2f72e276c2e4/cache-ee15f27a57925f61.arrow
    # SACREBLEU_FORMAT=text sacrebleu -tok 13a -w 2 ${OUTPUT_DIR}/de-en-test </home/aiscuser/filtered_wmt22/devset/test.de-en.en
    # --save_steps 0.1 \
    # --save_total_limit 2 \     --ignore_prompt_token_for_loss \
    # --fp16_full_eval \
    # --fp16_backend auto \
    # --torch_dtype float16 \
    #   --use_peft \
    # --lr_scheduler_type constant inverse_sqrt
    # --ignore_prompt_token_for_loss \
    # --max_eval_samples 100 \
    # --max_test_samples 100 \
    # --save_steps 0.1 \
    # --save_total_limit 2 \

# absolute_lr = base_lr * total_batch_size / 256",
#  9e-3 * 4*16 /256 0.0023




