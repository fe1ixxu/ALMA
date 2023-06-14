tgt=en
for ratio in 1000 10000 100000 1; do
    for src in  ha; do
        python transfer_txt2json.py \
        --path /home/aiscuser/filtered_wmt22/${src}${tgt}/top_1M/ \
        --src ${src} --tgt ${tgt} \
        --sample_ratio ${ratio}
    done
done