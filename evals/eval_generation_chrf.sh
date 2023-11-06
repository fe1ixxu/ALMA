OUTPUT_DIR=${1}
TEST_PAIRS=${2}

## Evaluation
source ~/.bashrc
conda activate alma-eval
for pair in ${TEST_PAIRS//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    echo "--------------------Results for ${pair}-------------------------------------"
    # Data path is the path to the test set.
    data_path=./data/${src}${tgt}/test.${src}-${tgt}.${tgt}
    output_path=${OUTPUT_DIR}/test-${src}-${tgt}
    SACREBLEU_FORMAT=text sacrebleu -m chrf --chrf-word-order 2 ${output_path} < ${data_path} > ${output_path}.chrf
    cat ${output_path}.chrf
done

for pair in ${TEST_PAIRS//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    echo "---------------------------${src}-${tgt}-------------------------------"
    output_path=${OUTPUT_DIR}/test-${src}-${tgt}
    cat ${output_path}.chrf
done
