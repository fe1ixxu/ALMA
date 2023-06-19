model=${1:-"facebook/opt-6.7b"}
src=${2:-de}
tgt=${3:-en}
bz=${4:-4}
beam=${5:-5}
gpu=${6:-0}
maxtoken=${7:-512}

# model: 
# opt-6.7b: facebook/opt-6.7b; covering languages: UNKOWN
#           overlapping: UNKOWN
# llama-7b: decapoda-research/llama-7b-hf #covering language:bg, ca, cs, da, de, en, es, fr, hr, hu, it, nl, pl, pt, ro, ru, sl, sr, sv, uk.
#           overlapping: cs, de, fr, en, ru, uk, 
# falcon-7b: tiiuae/falcon-7b
# BLOOM-7b: bigscience/bloom-7b1
# mpt-7b: mosaicml/mpt-7b
# falcon-7b-instruct: tiiuae/falcon-7b-instruct
# mpt-7b-instruct: mosaicml/mpt-7b-instruct

source activate llmmt2

echo "model ${model} src ${src} tgt ${tgt} beam ${beam} bz ${bz} gpu ${gpu}"
eval_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${src}
gold_path=/home/aiscuser/gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${tgt}
output_path=/mnt/sdrgmainz01wus2/t-haoranxu/results/zero-shot-llmmt-beam${beam}/${model}/${src}${tgt}/
mkdir -p ${output_path}
output_path=${output_path}test.${src}-${tgt}.${tgt}

export CUDA_VISIBLE_DEVICES=${gpu}
python eval_llm_translation.py \
--model_name ${model} \
--eval_path ${eval_path} \
--output_path ${output_path} \
--src_lang ${src} \
--tgt_lang ${tgt} \
--beam_size ${beam} \
--batch_size ${bz} \
--seed 42 \
--max_token_in_seq ${maxtoken} 

TOK="13a"
if [ ${tgt} == "zh" ]; then
    TOK="zh"
elif [ ${tgt} == "ja" ]; then
    TOK="ja-mecab"
fi

# if [ ${src} == "uk" ]; then
#     gold_path="test.uk-en.en"
# fi
mv ${output_path}.bleu ${output_path}.bleu-flores200
SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${output_path} < ${gold_path} > ${output_path}.bleu
comet-score -s ${eval_path} -t ${output_path} -r ${gold_path} > ${output_path}.comet

cat ${output_path}.bleu
tail -n 1 ${output_path}.comet

# python eval_llm_translation.py \
# --model_name decapoda-research/llama-7b-hf \
# --eval_path ./tmp/test-llama2/gold.de-en.de \
# --output_path ./tmp/test-llama2/eval-de-en.en \
# --src_lang de \
# --tgt_lang en \
# --beam_size 5 \
# --batch_size 1 \
# --seed 42 \
# --max_token_in_seq 64 
# src=de
# tgt=en
# comet-score -s ./gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${src}  -t ./gpt-MT/evaluation/system-outputs/text-davinci-003/zeroshot/${src}${tgt}/test.${src}-${tgt}.${tgt} -r ./gpt-MT/evaluation/testset/wmt-testset/${src}${tgt}/test.${src}-${tgt}.${tgt} 