TEST_PAIRS=ru-en
accelerate launch --config_file deepspeed_eval_config.yaml \
    run_clm.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --do_predict \
    --language_pairs ${TEST_PAIRS} \
    --data_path /home/aiscuser/filtered_wmt22/ \
    --per_device_eval_batch_size 4 \
    --output_dir ./tmp/full-ft-mmt \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --fp16 \
    --seed 42 \
    --num_beams 1 \
    --max_test_samples 100 \
    --overwrite_output_dir

exit
for pair in ${TEST_PAIRS//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    echo "--------------------Results for ${pair}-------------------------------------"
    src_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${src}
    tgt_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${tgt}
    output_path=./tmp/full-ft-mmt/test.${src}-${tgt}.txt 
    SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${output_path} < ${tgt_path}
done