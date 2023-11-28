export TASK_NAME=glue
export DIR=/home/datasets/zjlei
# export epoch=10
export AdapterType=sequencial_adapter
# for DATASET_NAME in cola  mrpc  qnli  qqp  rte  sst2  stsb
# for DATASET_NAME in rte mrpc stsb sst2 qnli  qqp   
# export lr=1e-5
export bs=8
export rf=8
dict=(rte mrpc stsb cola)
# export DATASET_NAME=cola2222 3333 4444 5555 rte mrpc stsb cola  
# export CUDA_VISIBLE_DEVICES=0mrpc stsb cola 
# export CUDA_VISIBLE_DEVICES=0,1,2,3rte mrpc stsb cola  sst2
for DATASET_NAME in   $1
do
    for train_samples in  100 200
    do
        if [[ "${dict[@]}" =~ "${DATASET_NAME}"  ]] ;
        then
            # epoch=10
            epoch=25
            epoch=20
            bs=8
        else
            # epoch=3
            epoch=20
            bs=8
        fi
        for lr in 1e-4 5e-4 5e-5 1e-5 #5e-4 1e-4 5e-5 #
        do
            for seed in 1111 2222 3333
            do
            python -m torch.distributed.launch --nproc_per_node 4 run_fewshot.py \
                --model_name_or_path $DIR/roberta-base \
                --task_name $DATASET_NAME \
                --dataset_name $DATASET_NAME \
                --do_train \
                --do_eval \
                --do_predict \
                --cache_dir /home/datasets/zjlei \
                --max_train_samples $train_samples \
                --max_seq_length 128 \
                --per_device_train_batch_size $bs \
                --learning_rate $lr \
                --num_train_epochs $epoch \
                --output_dir $DIR/fewshot-$TASK_NAME-result/$AdapterType/$DATASET_NAME/$DATASET_NAME-$bs-$epoch-$lr-$rf-$seed-train_sample-$train_samples/ \
                --overwrite_output_dir True\
                --hidden_dropout_prob 0.1 \
                --save_strategy epoch \
                --evaluation_strategy epoch \
                --seed $seed \
                --metric_for_best_model accuracy \
                --load_best_model_at_end \
                --train_adapter \
                --sequencial_adapter \
                --save_total_limit 1 \
                --adapter_reduction_factor $rf
                if [ $? -ne 0 ]; then
                    # 如果上一个命令返回非零退出状态码，则终止循环
                    echo "end because of error"
                    exit N
                fi
            done
        done
    done
done