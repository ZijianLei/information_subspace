export TASK_NAME=glue
export DIR=/home/datasets/zjlei
export epoch=10 # 50 epoch for rte mrpc cola stsb 25 epoch for qqp and qnli
export AdapterType=ib_adapter
dict=(rte mrpc stsb cola)
# export DATASET_NAME=qnli
# for DATASET_NAME in cola  mrpc  qnli  qqp  rte  sst2  stsb
# for DATASET_NAME in rte mrpc stsb sst2 qnli  qqp   2222 3333 4444 5555 rte cola
# export lr=1e-3
export bs=8
export beta=0.1
export NPROC_PER_NODE=4
# echo $NPROC_PER_NODE
# echo $SLURM_JOB_NUM_NODES
# echo $SLURM_PROCID
                       
# --nnodes=$SLURM_JOB_NUM_NODES  \
                        # --node_rank=$SLURM_PROCID \
                        # --master_addr=$MASTER_ADDR \
                        # --master_port=$MASTER_PORT \
# export id=50000 #32768
# export CUDA_VISIBLE_DEVICES=0,2,3,4
# env=$CONDA_DEFAULT_ENV
# if env=='a100';
# then
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# fi -m torch.distributed.launch --nproc_per_node 4 
# for DATASET_NAME in sst2 16384 bert-base-uncased deberta-v3-base qnli qqp 
# if the direction selection rate == 0, means using the first K direction
for DATASET_NAME in qnli cola #rte mrpc stsb cola # #rte mrpc stsb #qnli qqp # rte mrpc stsb cola  # sst2 mnli  #cola stsb  sst2  # # cola mrpc stsb cola 
do
    if [[ "${dict[@]}" =~ "${DATASET_NAME}"  ]] ;
    then
        epoch=10
        # epoch=25
        bs=8
        # bs=32
    else
        epoch=3
        # epoch=5
        bs=32
    fi
    # id = 100002 means it considering the moving average
    for id in  100002 #200001 #  16384 65536 32768 131072 32768
    do
        for lr in 1e-5 5e-5 #1e-5 # 1e-3 1e-4 #5e-4 #1e-4 1e-5 #  5e-4 1e-4 #1e-2 1e-3 5e-3  1e-4 1e-5 5e-5  #5e-5 1e-4   #5e-5 1e-41e-5
        do  # the direction selection_rate is the dropout rate
            for selection_step in 100 #40 60 80 #40 60 80 #40 60 80 #random #ipt_projection #ipt  
            do #this new rate is the number of paramter is used for general information and and rest of them is used for block-wise variation
                for direction_selection_rate in 1 #0.9 #0.9 1 #0.2 0.5 #0.5 0.75 #2 controling the len of mask
                do
                    for seed in 1111 2222 3333
                    do
                    dropout_method=random #ipt
                    python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
                        run.py \
                        --model_name_or_path $DIR/roberta-base \
                        --task_name $DATASET_NAME \
                        --dataset_name $DATASET_NAME \
                        --do_train \
                        --do_eval \
                        --do_predict \
                        --cache_dir /home/datasets/zjlei \
                        --max_seq_length 128 \
                        --per_device_train_batch_size $bs \
                        --learning_rate $lr \
                        --num_train_epochs $epoch \
                        --output_dir $DIR/$TASK_NAME-result/$AdapterType/$DATASET_NAME/$DATASET_NAME-$bs-$epoch-$lr-$id-$selection_step-$direction_selection_rate-$dropout_method-$seed-dataib/ \
                        --overwrite_output_dir True\
                        --hidden_dropout_prob 0.1 \
                        --save_strategy epoch \
                        --evaluation_strategy epoch \
                        --train_adapter False \
                        --zj_adapter False  \
                        --seed $seed \
                        --direction_selection_rate $direction_selection_rate \
                        --intrinsic_dim $id \
                        --warmup_ratio 0.06 \
                        --load_best_model_at_end \
                        --metric_for_best_model accuracy \
                        --save_total_limit 1 \
                        --dropout_method $dropout_method \
                        --selection_step $selection_step \
                        --ib \
                        --ib_dim 384 \
                        # --ddp_find_unused_parameters False 
                    done 
                done
            done
        done
    done
done