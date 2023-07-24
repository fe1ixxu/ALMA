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
exp_name=${1:-""}
export HF_DATASETS_CACHE="/home/aiscuser/huggingface_cache/datasets"
export WANDB_PROJECT=LLMMT-pre
export WANDB_NAME=${exp_name}
OUTPUT_DIR=/mnt/sdrgmainz01wus2/t-haoranxu/checkpoints/llmmt-pre/${exp_name}
DATASET=/home/aiscuser/filtered_wmt22/
# DATASET=/home/aiscuser/flores200/

accelerate launch --config_file deepspeed_train_config.yaml \
     run_clm.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --oscar_data_path oscar-corpus/OSCAR-2301 \
    --oscar_data_lang ru,en \
    --interleave_probs "0.5,0.5" \
    --streaming \
    --max_steps 600000 \
    --do_train \
    --low_cpu_mem_usage \
    --fp16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 1 \
    --output_dir ${OUTPUT_DIR} \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --seed 42 \
    --overwrite_output_dir \
    --report_to wandb



