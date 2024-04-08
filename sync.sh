#!/bin/bash
rsync -aP \
    --exclude "venv/" \
    --exclude "checkpoint/" \
    --exclude "data/" \
    * \
    mriyadh@login2.pegasus.kl.dfki.de:~/learn2ask
