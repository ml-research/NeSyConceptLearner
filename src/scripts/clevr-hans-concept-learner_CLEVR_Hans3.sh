#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="concept-learner-$NUM"
DATASET=clevr-hans-state
OUTPATH="out/clevr-state/$MODEL-$ITER"

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

# For gpu
CUDA_VISIBLE_DEVICES=$DEVICE python train_nesy_concept_learner_clevr_hans.py --data-dir $DATA --dataset $DATASET \
--epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
--mode train

# For cpu
#CUDA_VISIBLE_DEVICES=$DEVICE python train_nesy_concept_learner_clevr_hans.py --data-dir $DATA --dataset $DATASET \
#--epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
#--mode train --no-cuda
