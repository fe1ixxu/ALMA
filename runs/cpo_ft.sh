OUTPUT_DIR=${1:-"./alma-7b-dpo-ft"}
pairs=${2:-"de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"
# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

accelerate launch --main_process_port ${port} --config_file configs/deepspeed_train_config_bf16.yaml \
     run_cpo_llmmt.py \
    --model_name_or_path haoranxu/ALMA-13B-Pretrain \
    --tokenizer_name haoranxu/ALMA-13B-Pretrain \
    --peft_model_id  haoranxu/ALMA-13B-Pretrain-LoRA \
    --cpo_scorer kiwi_xcomet \
    --beta 0.1 \
    --use_peft \
    --use_fast_tokenizer False \
    --cpo_data_path  haoranxu/ALMA-R-Preference \
    --do_train \
    --language_pairs ${pairs} \
    --low_cpu_mem_usage \
    --bf16 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --prediction_loss_only \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --max_prompt_length 256 \
    --max_length 512 \
    --seed 42 \
    --overwrite_output_dir \
    --report_to none \
    --overwrite_cache 