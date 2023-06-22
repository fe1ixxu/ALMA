#!/bin/bash
# source ~/.bashrc
# conda activate llmmt2
suffix=${1:-10000}
OUTPUT_DIR=./tmp/full-ft-mmt
# accelerate launch --config_file deepspeed_train_config.yaml \
#      run_clm.py \
#     --model_name_or_path decapoda-research/llama-7b-hf \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --language_pairs en-de,de-en,en-ja,ja-en,en-zh,zh-en,en-is,is-en \
#     --load_best_model_at_end \
#     --fp16 \
#     --suffix ${suffix} \
#     --data_path /home/aiscuser/filtered_wmt22/ \
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
#     --report_to wandb \
#     --run_name ${OUTPUT_DIR} \
#     --num_train_epochs 3 \
#     --predict_with_generate \
#     --prediction_loss_only \
#     --max_new_tokens 256 \
#     --max_source_length 256 \
#     --seed 42 \
#     --overwrite_output_dir

        # --overwrite_cache \
# /home/aiscuser/.cache/huggingface/datasets/json/default-fc0d51b772e14738/0.0.0/e347ab1c932092252e717ff3f949105a4dd28b27e842dd53157d2f72e276c2e4/cache-ee15f27a57925f61.arrow
    SACREBLEU_FORMAT=text sacrebleu -tok 13a -w 2 ${OUTPUT_DIR}/de-en-test </home/aiscuser/filtered_wmt22/devset/test.de-en.en
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
# exit
    #     --fp16 \
    # --fp16_full_eval \
    # --fp16_backend auto \
    # --torch_dtype float16 \
# absolute_lr = base_lr * total_batch_size / 256",
#  9e-3 * 4*16 /256 0.0023
# decapoda-research/llama-7b-hf
# torchrun --nproc_per_node 8 
# exit
accelerate launch --config_file deepspeed_eval_config.yaml \
    run_clm.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --do_predict \
    --language_pairs en-is \
    --data_path /home/aiscuser/filtered_wmt22/ \
    --ignore_prompt_token_for_loss \
    --per_device_eval_batch_size 4 \
    --output_dir ${OUTPUT_DIR}/checkpoint-2250/ \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --fp16 \
    --seed 42 \
    --num_beams 5

exit
src=de
tgt=en
src_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${src}
tgt_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${tgt}
output_path=./tmp/test-llama/de-en.txt #./tmp/test-clm/de-en.txt
TOK="13a"
if [ ${tgt} == "zh" ]; then
    TOK="zh"
elif [ ${tgt} == "ja" ]; then
    TOK="ja-mecab"
fi
SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${output_path} < ${tgt_path} > ${output_path}.bleu
cat ${output_path}.bleu
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


