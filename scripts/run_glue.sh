export TASK_NAME=glue
# export DATASET_NAME=rte
export DIR=/home/datasets/zjlei
export AdapterType=finetune
dict=(rte mrpc stsb cola)
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0qqp rte mrpc stsb cola sst2 qnli
for DATASET_NAME in  $1 #mrpc
do
    if [[ "${dict[@]}" =~ "${DATASET_NAME}"  ]] ;
    then
        epoch=1
        # epoch=2
        bs=8
    else
        epoch=3
        bs=32
    fi
    for lr in  5e-5 #5e-5 1e-4 5e-4 --warmup_ratio 0.06 \
    do
      for seed in 1111 #2222 3333 #2222 3333
      do
      python -m torch.distributed.launch --nproc_per_node 4 run.py \
        --model_name_or_path  $DIR/roberta-base \
        --task_name $DATASET_NAME \
        --dataset_name  $DATASET_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --cache_dir /home/datasets/zjlei \
        --max_seq_length 128 \
        --per_device_train_batch_size $bs \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --output_dir /$DIR/$TASK_NAME-result/$AdapterType/$DATASET_NAME-$bs-$lr-$seed-$epoch/ \
        --overwrite_output_dir True \
        --hidden_dropout_prob 0.1 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --seed $seed \
        --train_adapter False \
        --warmup_ratio 0.06 \
        --load_best_model_at_end \
        --metric_for_best_model accuracy \
        --save_total_limit 1
        done
    done
done