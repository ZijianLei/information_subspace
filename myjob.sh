#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --mem-per-gpu=20G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --exclude=hkbugpudgx01,hkbugpusrv[07,08]

#SBATCH --output=ib.out

export NPROC_PER_NODE=4
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4) + $SLURM_ARRAY_TASK_ID)
export MASTER_PORT=1234
echo "MASTER_PORT="$MASTER_PORT
srun bash scripts/run_glue_ib.sh

