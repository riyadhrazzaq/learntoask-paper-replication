#!/bin/bash

set -e

WORK_DIR="checkpoints/optuna-reproduce-second"
INPUT_FILE="task1/data/task1_dev.jsonl"
CHECKPOINT="$WORK_DIR/model_best"
OUTPUT_FILE="$WORK_DIR/task1_dev_mela.jsonl"

# generate
python task1/src/generate.py \
$INPUT_FILE \
$OUTPUT_FILE \
$CHECKPOINT

# score
python -m task1.scorer.task1 \
-g $INPUT_FILE \
-p $OUTPUT_FILE