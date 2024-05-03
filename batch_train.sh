#!/bin/bash
#SBATCH --job-name araieval-train-longer
#SBATCH --partition RTXA6000
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --time=03-00:00:00

lr=0.00006  # from optuna
weight_decay=0.0012404502272307953  # from optuna
warmup_steps=120  # from optuna


srun --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
    --task-prolog="`pwd`/install.sh" \
    -u \
    python task1/src/train.py \
    task1/data/task1_train.jsonl \
    task1/data/task1_dev.jsonl \
    optuna-reproduce-final-longer \
    --max-epoch 400 \
    --lr $lr \
    --weight-decay $weight_decay \
    --warmup-steps $warmup_steps

