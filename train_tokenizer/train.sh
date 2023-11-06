# spm_train: train spm on target language corpus, prune all llama overlapping tokens and save spm
# add_tokens: extend hf tokenizer with new spm tokens
# spm from spm_train is used in wechsel 

python spm_train.py /
    --input_file $INPUT_FILE /
    --model_prefix $MODEL_PREFIX /
    --vocab_size $VOCAB_SIZE /
    --llama_dir $LLAMA_DIR 

python add_tokens.py /
    --llama_tokenizer_dir $LLAMA_DIR /
    --new_sp_model_file $MODEL_PREFIX.model /
    --output_hf_dir $OUTPUT_HF_DIR 
