#!/bin/bash

DATA=0-05_10000_0-25

gm=loopy_bp

LV=3
CONV_SIZE=64
FP_LEN=0
n_hidden=128
bsize=128
num_epochs=2000
learning_rate=0.1
optim=Adagrad
momentum=0
lr_decay=0
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
    -optim $optim \
    -lr_decay $lr_decay \
    -momentum $momentum \
    $@
