source activate llmmt2
# python run_clm.py \
#     --model_name_or_path facebook/opt-125m \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --language_pairs de-en \
#     --suffix 100000 \
#     --data_path /home/aiscuser/filtered_wmt22/ \
#     --ignore_prompt_token_for_loss \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 8 \
#     --evaluation_strategy steps \
#     --eval_steps 0.05 \
#     --save_strategy steps \
#     --save_steps 0.05 \
#     --save_total_limit 2 \
#     --logging_strategy steps \
#     --logging_steps 0.05 \
#     --output_dir ./tmp/test-clm \
#     --num_train_epochs 10 \
#     --max_eval_samples 100 \
#     --max_test_samples 100 \
#     --predict_with_generate \
#     --prediction_loss_only \
#     --max_new_tokens 64 \
#     --max_source_length 128 \
#     # --overwrite_cache \
#     # --overwrite_output_dir \
    


    #     --fp16 \
    # --fp16_full_eval \
    # --fp16_backend auto \
    # --torch_dtype float16 \

# decapoda-research/llama-7b-hf
python run_clm.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --do_predict \
    --language_pairs de-en \
    --data_path /home/aiscuser/filtered_wmt22/ \
    --ignore_prompt_token_for_loss \
    --per_device_eval_batch_size 1 \
    --output_dir ./tmp/llama \
    --predict_with_generate \
    --max_new_tokens 128 \
    --max_source_length 128 \

src=de
tgt=en
src_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${src}
tgt_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${tgt}
output_path=./tmp/llama/de-en.txt #./tmp/test-clm/de-en.txt
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


