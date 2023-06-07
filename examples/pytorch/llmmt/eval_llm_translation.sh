model=${1:-"facebook/opt-6.7b"}
src=${2:-de}
tgt=${3:-en}
bz=${4:-16}
beam=${5:-1}

eval_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${src}
gold_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${tgt}
output_path=/mnt/sdrgmainz01wus2/t-haoranxu/results/zero-shot-llmmt-beam${beam}/${model}/${src}${tgt}/
mkdir -p ${output_path}
output_path=${output_path}test.${src}-${tgt}.${tgt}

python eval_llm_translation.py \
--model_name ${model} \
--eval_path ${eval_path} \
--output_path ${output_path} \
--src_lang ${src} \
--tgt_lang ${tgt} \
--beam_size ${beam} \
--batch_size ${bz} \
--seed 42 \
--max_token_in_seq 512 \


SACREBLEU_FORMAT=text sacrebleu -tok flores200 -w 2 ${output_path} < ${gold_path} > ${output_path}.bleu

cat ${output_path}.bleu