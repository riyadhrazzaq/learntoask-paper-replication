#!/bin/bash

srun --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
    --partition=RTX3090 \
    --mem 30GB \
    --gpus=1 \
    --task-prolog="`pwd`/install.sh" \
    python src/generate.py data/test.src data/test.tgt $1 --search $2 --p $3 --outfile $1/test.tgt.$2

