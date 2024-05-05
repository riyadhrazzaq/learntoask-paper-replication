#!/bin/bash
base_url="https://raw.githubusercontent.com/xinyadu/nqg/master/data/processed"
for split in "train" "test" "dev"
do
    for type in "src" "tgt"
    do
        url=$base_url/$type-$split.txt
        wget $url -O $split.$type
    done
done
