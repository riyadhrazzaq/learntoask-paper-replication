#!/bin/bash
# setup
pip install -r requirements.txt
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyODkyMGQ2OS0xZTJhLTQwMDAtOWM1My1jZDNlOGU3YzdkZDQifQ=="

# train
python train.py data/processed/src-train.txt data/processed/tgt-train.txt \
    data/processed/src-dev.txt data/processed/tgt-dev.txt  \
    --glove_embedding_dir data/ \
    --src_max_seq 40 --tgt_max_seq 15 \
    --hidden_dim 300 --num_layers 2 --dropout 0.3 \
    --lr 1.0 --clip_norm 5 --lr_decay 0.9 --lr_decay_from 17 \
    --max_epoch 40 --batch_size 64 \
    --enable_neptune --experiment_name without-lr-decay
