#!/bin/bash
rsync -aP \
    --exclude-from .rsync-exclude \
    * \
    mriyadh@login3.pegasus.kl.dfki.de:/netscratch/mriyadh/learningtoask
