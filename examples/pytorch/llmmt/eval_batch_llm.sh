############## en->xx #####################
src=en
tgt=${1}
gpu=${2}
beam=5
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
# tr '\t' ' ' < g.txt > g2.txt


for model in "facebook/opt-6.7b" "decapoda-research/llama-7b-hf" "tiiuae/falcon-7b" "bigscience/bloom-7b1" "mosaicml/mpt-7b" "tiiuae/falcon-7b-instruct" "mosaicml/mpt-7b-instruct"; do
    bash ./eval_llm_translation.sh ${model} ${src} ${tgt} 4 ${beam} ${gpu} 512
done

for model in "facebook/opt-6.7b" "decapoda-research/llama-7b-hf" "tiiuae/falcon-7b" "bigscience/bloom-7b1" "mosaicml/mpt-7b" "tiiuae/falcon-7b-instruct" "mosaicml/mpt-7b-instruct"; do
    output_path=/mnt/sdrgmainz01wus2/t-haoranxu/results/zero-shot-llmmt-beam${beam}/${model}/${src}${tgt}/test.${src}-${tgt}.${tgt}
    echo "---------------------------${model}-${src}-${tgt}-------------------------------"
    cat ${output_path}.bleu
    # tail -n 1 ${output_path}.comet
done

############## xx->en #####################
src=${tgt}
tgt=en
for model in "facebook/opt-6.7b" "decapoda-research/llama-7b-hf" "tiiuae/falcon-7b" "bigscience/bloom-7b1" "mosaicml/mpt-7b" "tiiuae/falcon-7b-instruct" "mosaicml/mpt-7b-instruct"; do
    bash ./eval_llm_translation.sh ${model} ${src} ${tgt} 4 ${beam} ${gpu} 512
done

for model in "facebook/opt-6.7b" "decapoda-research/llama-7b-hf" "tiiuae/falcon-7b" "bigscience/bloom-7b1" "mosaicml/mpt-7b" "tiiuae/falcon-7b-instruct" "mosaicml/mpt-7b-instruct"; do
    output_path=/mnt/sdrgmainz01wus2/t-haoranxu/results/zero-shot-llmmt-beam${beam}/${model}/${src}${tgt}/test.${src}-${tgt}.${tgt}
    echo "---------------------------${model}-${src}-${tgt}-------------------------------"
    cat ${output_path}.bleu
    # tail -n 1 ${output_path}.comet
done