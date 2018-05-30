#!/usr/bin/env bash

cd tools

python prepare_list.py
python obtain_word_embedding.py --wv glove
python convert_to_gcn_data.py --wv glove --fc res50

cd ../..