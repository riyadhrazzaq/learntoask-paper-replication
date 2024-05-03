#!/bin/bash
#SBATCH --job-name araieval-train-new
#SBATCH --partition RTXA6000
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --time=03-00:00:00


lr=0.0000935  # from optuna
weight_decay=0.006504135871871216  # from optuna
warmup_steps=413  # from optuna

echo $lr

srun --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
    --task-prolog="`pwd`/install.sh" \
    -u \
    python task1/src/train.py \
    task1/data/task1_train.jsonl \
    task1/data/task1_dev.jsonl \
    optuna-reproduce-new \
    --max-epoch 200 \
    --lr $lr \
    --weight-decay $weight_decay \
    --warmup-steps $warmup_steps
