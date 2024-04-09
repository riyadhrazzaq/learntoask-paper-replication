#!/bin/bash
data_dir="data/squad"
train="train-v2.0.json"
dev="dev-v2.0.json"

if [ -f $data_dir/$train ]; then
  echo "SQuAD exists. Skipping download."
else
  echo "Downloading data"
  mkdir -p data/squad
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
  mv train-v2.0.json dev-v2.0.json $data_dir
  # parses these files into train.src, train.tgt, dev.src, dev.tgt
  python jsontotxt.py
fi


# for tokenization
if [ -d mosesdecoder ]; then
    echo "mosesdecoder exists. Skipping download."
else
    git clone https://github.com/moses-smt/mosesdecoder.git
fi


mkdir -p $data_dir/processed tmp

for split in train dev; do
  for type in src tgt; do
    mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $data_dir/"$split"."$type" > tmp/"$split"."$type"
    mosesdecoder/scripts/tokenizer/lowercase.perl -l en < tmp/"$split"."$type" > $data_dir/processed/"$split"."$type"
  done
done
#rm -rf tmp