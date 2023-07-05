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
suffix=${2:-"1000"}
pairs=${3:-"en-de,en-cs,en-is,en-zh,en-ja,en-ru,en-uk,en-ha,de-en,cs-en,is-en,zh-en,ja-en,ru-en,uk-en,ha-en"}
export HF_DATASETS_CACHE="/home/aiscuser/huggingface_cache/datasets"
export WANDB_PROJECT=LLMMT-pre
export WANDB_NAME=${exp_name}
OUTPUT_DIR=/home/aiscuser/checkpoints/llmmt-pre/${exp_name}
DATASET=/home/aiscuser/filtered_wmt22/
# DATASET=/home/aiscuser/flores200/

accelerate launch --config_file deepspeed_train_config.yaml \
     run_clm.py \
    --model_name_or_path mosaicml/mpt-7b \
    --mmt_data_path  ${DATASET} \
    --suffix ${suffix} \
    --right_pad \
    --do_train \
    --do_eval \
    --do_predict \
    --language_pairs ${pairs} \
    --load_best_model_at_end \
    --low_cpu_mem_usage \
    --fp16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.05 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy steps \
    --eval_steps 0.05 \
    --save_strategy steps \
    --save_steps 0.05 \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --predict_with_generate \
    --prediction_loss_only \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --seed 42 \
    --overwrite_output_dir \
    --num_beams 5 \
    --report_to none

source /home/aiscuser/anaconda3/bin/activate comet
for pair in ${pairs//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    TOK="13a"
    if [ ${tgt} == "zh" ]; then
        TOK="zh"
    elif [ ${tgt} == "ja" ]; then
        TOK="ja-mecab"
    fi
    echo "--------------------Results for ${pair}-------------------------------------"
    src_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${src}
    tgt_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${tgt}
    output_path=${OUTPUT_DIR}/test-${src}-${tgt}
    if [ ${src} == "uk" ]; then
        expand -t 4 ${tgt_path} > ${output_path}-tmp-uk-en.gold
        SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${output_path} < ${output_path}-tmp-uk-en.gold > ${output_path}.bleu
        rm ${output_path}-tmp-uk-en.gold
    else
        SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${output_path} < ${tgt_path} > ${output_path}.bleu
    fi 
    cat ${output_path}.bleu
    comet-score -s ${src_path} -t ${output_path} -r ${tgt_path} --batch_size 256 > ${output_path}.comet 
    tail -n 1 ${output_path}.comet
done

for pair in ${pairs//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    echo "---------------------------${src}-${tgt}-------------------------------"
    output_path=${OUTPUT_DIR}/test-${src}-${tgt}
    cat ${output_path}.bleu
    tail -n 1 ${output_path}.comet
done



# --use_ul2 \
#### Possible fields may be needed

    # --overwrite_cache \
    # facebook/opt-125m
    # SACREBLEU_FORMAT=text sacrebleu -tok 13a -w 2 /home/aiscuser/flores200/llama-13b/test-en-de </home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/ende/test.en-de.de
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




