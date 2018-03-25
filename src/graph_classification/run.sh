#!/bin/bash

DATA=0-1_10000

gm=loopy_bp

LV=3
CONV_SIZE=64
FP_LEN=0
n_hidden=128
bsize=100
num_epochs=200
learning_rate=0.001
fold=1

python2 main.py \
    -seed 1 \
    -mode gpu \
    -data $DATA \
    -learning_rate $learning_rate \
    -num_epochs $num_epochs \
    -hidden $n_hidden \
    -max_lv $LV \
    -latent_dim $CONV_SIZE \
    -out_dim $FP_LEN \
    -batch_size $bsize \
    $@
