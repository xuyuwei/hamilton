#!/bin/bash

DATA=ACTUAL_DATA

gm=loopy_bp

LV=3
CONV_SIZE=64
FP_LEN=64
n_hidden=128
bsize=20
num_epochs=10
learning_rate=0.1
optim=Adagrad
momentum=0
lr_decay=0
fold=2
save_dir=best-model/

python2 main.py \
    -seed 1 \
    -mode cpu \
    -gm $gm \
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
    -save_dir $save_dir \
    $@
