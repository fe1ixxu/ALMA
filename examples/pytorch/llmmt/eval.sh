TEST_PAIRS="en-de,en-cs,en-is,en-zh,en-ja,en-ru,en-uk,en-ha,de-en,cs-en,is-en,zh-en,ja-en,ru-en,uk-en,ha-en"
OUTPUT_DIR=/home/aiscuser/checkpoints/llmmt-pre/lla-7b-mmt-instruct-10k
accelerate launch --config_file deepspeed_eval_config.yaml \
    run_clm.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --do_predict \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path /home/aiscuser/filtered_wmt22/ \
    --per_device_eval_batch_size 4 \
    --output_dir ${OUTPUT_DIR} \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --fp16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_output_dir 

for pair in ${TEST_PAIRS//,/ }; do
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
        SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${output_path} < ${output_path}-tmp-uk-en.gold
        rm ${output_path}-tmp-uk-en.gold
    else
        SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${output_path} < ${tgt_path}
    fi
    
done