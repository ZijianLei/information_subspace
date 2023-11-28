dict=(rte mrpc stsb cola)
for DATASET_NAME in rte mrpc stsb cola sst2 qnli qqp 
do
    if [[ "${dict[@]}" =~ "${DATASET_NAME}"   ]] ;
    then
        echo 'small data'
        echo $dict
    else
        echo 'large data'
    fi
done