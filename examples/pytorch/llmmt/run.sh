python run_clm.py \
    --model_name_or_path facebook/opt-125m \
    --language_pairs de-en,en-de \
    --suffix 10000 \
    --data_path /home/aiscuser/filtered_wmt22/ \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --logging_steps 20 \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./tmp/test-clm \
    --ignore_prompt_token_for_loss \
    --num_train_epochs 1 \
    --max_eval_samples 10 \
    --predict_with_generate \
    --max_new_tokens 64 \
    --max_source_length 128


    #     --fp16 \
    # --fp16_full_eval \
    # --fp16_backend auto \
    # --torch_dtype float16 \

# decapoda-research/llama-7b-hf
# python ./run_translation.py \
#     --model_name_or_path Helsinki-NLP/opus-mt-en-ro \
#     --do_train \
#     --do_predict \
#     --source_lang en \
#     --target_lang ro \
#     --dataset_name wmt16 \
#     --dataset_config_name ro-en \
#     --half_precision_backend auto \
#     --output_dir ./tmp/tst-translation \
#     --per_device_train_batch_size=4 \
#     --per_device_eval_batch_size=4 \
#     --overwrite_output_dir \
#     --predict_with_generate

# python run_translation_no_trainer.py \
#     --model_name_or_path Helsinki-NLP/opus-mt-en-ro \
#     --source_lang en \
#     --target_lang ro \
#     --dataset_name wmt16 \
#     --dataset_config_name ro-en \
#     --output_dir ~/tmp/tst-translation