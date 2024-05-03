#!/bin/bash
#SBATCH --job-name araieval-tune
#SBATCH --partition RTXA6000
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --time=03-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=md.riyadh.23@um.edu.mt


srun --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
    --task-prolog="`pwd`/install.sh" \
    python task1/src/tune.py \
    task1/data/task1_train.jsonl \
    task1/data/task1_dev.jsonl \
    tune-adamw-clswt-seed-again \
    --num-trials 50

