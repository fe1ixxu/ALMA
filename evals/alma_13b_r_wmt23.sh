OUTPUT_DIR=${1:-"./outputs-alma-13b-r-wmt23/"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"
# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

python \
    run_llmmt.py \
    --model_name_or_path haoranxu/ALMA-13B-R \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs en-ru,en-zh \
    --mmt_data_path ./human_written_data/ \
    --override_test_data_path haoranxu/WMT23-Test \
    --per_device_eval_batch_size 1 \
    --output_dir ${OUTPUT_DIR} \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --multi_gpu_one_model

python
    run_llmmt.py \
    --model_name_or_path haoranxu/ALMA-13B-R \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs ru-en,de-en,en-de,zh-en \
    --mmt_data_path ./human_written_data/ \
    --override_test_data_path haoranxu/WMT23-Test \
    --per_device_eval_batch_size 1 \
    --output_dir ${OUTPUT_DIR} \
    --predict_with_generate \
    --max_new_tokens 512 \
    --max_source_length 1024 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --multi_gpu_one_model

bash ./evals/eval_generation_wmt23.sh ${OUTPUT_DIR} de-en,zh-en,ru-en,en-de,en-ru,en-zh
