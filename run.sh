#!/bin/bash

srun --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
    --partition=RTX3090 \
    --gpus=1 \
    --task-prolog="`pwd`/install.sh" \
    --pty /bin/bash

