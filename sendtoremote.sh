#!/bin/bash

REMOTE_DIR=/fscratch/mriyadh/learntoask-paper-replication

rsync -aP \
    --exclude-from .rsync-exclude \
    ./* \
    mriyadh@login3.pegasus.kl.dfki.de:$REMOTE_DIR
