MODEL_NAME=${1}
TEST_PAIRS=${2:-"de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru"}
MODEL="${MODEL_NAME//\//-}"
OUTPUT_DIR=outputs-${MODEL}

export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"
# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

if [[ ${MODEL_NAME} == "meta-llama/Llama-2-7b-hf" ]]; then
    REVISION="--model_revision 637a748546bb9abca62b0684183cc362bc1ece6d"
elif [[ ${MODEL_NAME} == "meta-llama/Llama-2-13b-hf" ]]; then
    REVISION="--model_revision --model_revision 9474c6d222f45e7eb328c0f6b55501e7da67c9c3"
fi

## Generation
accelerate launch --main_process_port ${port} --config_file configs/deepspeed_eval_config.yaml \
    run_llmmt.py \
    --model_name_or_path ${MODEL} \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path ./human_written_data/ \
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
    ${REVISION}

## Evaluation (BLEU, COMET)
bash ./evals/eval_generation.sh ${OUTPUT_DIR} ${TEST_PAIRS}