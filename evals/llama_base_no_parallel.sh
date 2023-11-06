OUTPUT_DIR=${1:-"./outputs-llama-base/"}
TEST_PAIRS=${2:-"yo-en,en-yo"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"
# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

## Generation
python \
    run_llmmt.py \
    --model_name_or_path "/mnt/disk/llama-lang-adapt/data/models/llama-2-7b-hf" \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path ./african_data/ \
    --per_device_eval_batch_size 2 \
    --output_dir ${OUTPUT_DIR} \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --fp16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --multi_gpu_one_model 

## Evaluation (CHRF++)
bash ./evals/eval_generation_chrf.sh ${OUTPUT_DIR} ${TEST_PAIRS}