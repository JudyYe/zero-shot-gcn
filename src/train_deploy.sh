#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=$1 \
python gcn/train_gcn.py \
	--dataset ../data/glove_res50/ \
	--save_path ../output/