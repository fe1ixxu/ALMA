TEST_PAIRS=${1:-"en-de,en-cs,en-is,en-zh,en-ja,en-ru,en-uk,en-ha,de-en,cs-en,is-en,zh-en,ja-en,ru-en,uk-en,ha-en"}
OUTPUT_DIR=${2}
export HF_DATASETS_CACHE="/home/aiscuser/huggingface_cache/datasets"
accelerate launch --config_file deepspeed_eval_config.yaml \
    run_clm.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path /home/aiscuser/filtered_wmt22/ \
    --per_device_eval_batch_size 4 \
    --output_dir ${OUTPUT_DIR} \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --fp16 \
    --seed 42 \
    --num_beams 5 

if [[ ${TEST_PAIRS} == *zh-en* ]]; then
    accelerate launch --config_file deepspeed_eval_config.yaml \
        run_clm.py \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --do_predict \
        --low_cpu_mem_usage \
        --language_pairs zh-en \
        --mmt_data_path /home/aiscuser/filtered_wmt22/ \
        --per_device_eval_batch_size 4 \
        --output_dir ${OUTPUT_DIR} \
        --predict_with_generate \
        --max_new_tokens 256 \
        --max_source_length 512 \
        --fp16 \
        --seed 42 \
        --num_beams 5 
fi

source /home/aiscuser/anaconda3/bin/activate comet
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
        SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${output_path} < ${output_path}-tmp-uk-en.gold > ${output_path}.bleu
        rm ${output_path}-tmp-uk-en.gold
    else
        SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${output_path} < ${tgt_path} > ${output_path}.bleu
    fi 
    cat ${output_path}.bleu
    comet-score -s ${src_path} -t ${output_path} -r ${tgt_path} --batch_size 256 > ${output_path}.comet 
    comet-score -s ${src_path} -t ${output_path} -r ${tgt_path} --batch_size 256 --model /mnt/sdrgmainz01wus2/t-haoranxu/comet_models/wmt22-cometkiwi-da/checkpoints/model.ckpt > ${output_path}.comet-kiwi 
    tail -n 1 ${output_path}.comet
    tail -n 1 ${output_path}.comet-kiwi
done

for pair in ${TEST_PAIRS//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    echo "---------------------------${src}-${tgt}-------------------------------"
    output_path=${OUTPUT_DIR}/test-${src}-${tgt}
    cat ${output_path}.bleu
    tail -n 1 ${output_path}.comet
    tail -n 1 ${output_path}.comet-kiwi
done
