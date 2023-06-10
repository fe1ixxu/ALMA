python run_clm.py \
    --model_name_or_path facebook/opt-6.7b \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --overwrite_cache \
    --output_dir ./tmp/test-clm

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