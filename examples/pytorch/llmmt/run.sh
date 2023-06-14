accelerate launch run_clm.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --language_pairs de-en,en-de \
    --suffix 10000 \
    --data_path /home/aiscuser/filtered_wmt22/ \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --overwrite_cache \
    --output_dir ./tmp/test-clm2 \
    --ignore_prompt_token_for_loss \
    --fp16 \
    --fp16_full_eval \
    --fp16_backend auto \
    --torch_dtype float16 \
    --num_train_epochs 10 \

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