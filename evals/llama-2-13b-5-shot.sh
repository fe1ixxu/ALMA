OUTPUT_DIR=${1:-"./outputs-llama-2-13b-5-shot/"}
TEST_PAIRS=${2:-"de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru"}

export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"
# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

accelerate launch --main_process_port ${port} --config_file configs/deepspeed_eval_config.yaml \
    run_llmmt.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --model_revision 9474c6d222f45e7eb328c0f6b55501e7da67c9c3 \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path ./human_written_data/ \
    --per_device_eval_batch_size 2 \
    --output_dir ${OUTPUT_DIR} \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 768 \
    --fp16 \
    --seed 42 \
    --num_beams 1 \
    --few_shot_eval_path ./human_written_data/HR-5-shot/ \
    --overwrite_cache \
    --overwrite_output_dir

## Evaluation (BLEU, COMET)
bash ./evals/eval_generation.sh ${OUTPUT_DIR} ${TEST_PAIRS}