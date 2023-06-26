### Instruct Tuning:

export HF_DATASETS_CACHE="/home/aiscuser/huggingface_cache/datasets"
export WANDB_PROJECT=LLMMT-pre
export WANDB_NAME=our_alpaca-cosine

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

