#!/bin/bash

# 服务器列表
# servers=("hkbugpusrv02" "hkbugpusrv03" "hkbugpusrv04" "hkbugpusrv05" "hkbugpusrv06" )
servers=("hkbugpusrv02" "hkbugpusrv03" "hkbugpusrv05" "hkbugpusrv04")
# 任务列表 
tasks=("rte" "mrpc" "stsb" "sst2" "qnli" )
# tasks=("rte" "qnli")

# tasks=("sst2" "qnli" "qqp")
# 初始化一个字典来存储每个服务器的任务
declare -A server_tasks
# 遍历任务列表
for i in "${!tasks[@]}"
do
  
  # 使用模运算符来轮流分配任务给服务器
  server=${servers[$i%${#servers[@]}]}
  echo $server_tasks[$server]
  if [ -z "${server_tasks[$server]}" ]; 
  then
    # server_tasks[$server]="bash scripts/run_glue_dataib_fewshot.sh ${tasks[$i]} ;" # data ib few-shot
    # server_tasks[$server]="bash scripts/run_glue_modelib.sh ${tasks[$i]} ;"  # model ib
    server_tasks[$server]="${server_tasks[$server]} bash scripts/run_glue_modelib_fewshot.sh ${tasks[$i]} ;" # model ib few-shot
    # server_tasks[$server]="${server_tasks[$server]} bash scripts/glue_lora_fewshot.sh ${tasks[$i]} ;" # model lora few-shot
    # server_tasks[$server]="${server_tasks[$server]} bash scripts/glue_adapter_fewshot.sh ${tasks[$i]} ;" # model adapter few-shot
    # server_tasks[$server]="${server_tasks[$server]} bash scripts/glue_mam_fewshot.sh ${tasks[$i]} ;" # model mam few-shot
  else
    # server_tasks[$server]="${server_tasks[$server]} bash scripts/run_glue_dataib_fewshot.sh ${tasks[$i]} ;"
    # server_tasks[$server]="${server_tasks[$server]} bash scripts/run_glue_modelib.sh ${tasks[$i]} ;"
    server_tasks[$server]="${server_tasks[$server]} bash scripts/run_glue_modelib_fewshot.sh ${tasks[$i]} ;"
    # server_tasks[$server]="${server_tasks[$server]} bash scripts/glue_lora_fewshot.sh ${tasks[$i]} ;"
    # server_tasks[$server]="${server_tasks[$server]} bash scripts/glue_adapter_fewshot.sh ${tasks[$i]} ;"
    # server_tasks[$server]="${server_tasks[$server]} bash scripts/glue_mam_fewshot.sh ${tasks[$i]} ;" # model mam few-shot
  fi

  # 添加任务到服务器的任务列表

#   server_tasks[$server]+="bash scripts/run_glue_dataib_fewshot.sh ${tasks[$i]} ;"
#   echo "bash scripts/run_glue_dataib_fewshot.sh ${tasks[$i]}"
done 
# echo ${server_tasks[@]}

for server in "${servers[@]}"
do
  # 使用 SSH 在每个服务器上运行分配给它的任务
#   echo $server
#   echo "${server_tasks[$server]}" 
#   ssh $server "cd transformer/variational_subspace && echo "${server_tasks[$server]}" "
  ssh $server "cd transformer/variational_subspace && ${server_tasks[$server]}" &
done