import argparse
import json
import numpy as np
import os

import pickle as pkl
import scipy.io as sio

def test(image_file, fc, feat_dir):
    index_list, label_list = [], []
    with open(image_file) as fp:
        for line in fp:
            index, l = line.split()
            index_list.append(index.split('.')[0])
            label_list.append(int(l))

    top_retrv = [1, 5]
    hit_count = np.zeros((len(top_retrv)))
    cnt_valid = len(index_list)
    for i, index in enumerate(index_list):
        feat_path = os.path.join(feat_dir, index + '.npz')
        feat = np.load(feat_path)['feat']

        scores = np.matmul(feat, fc)
        # print(scores.shape)
        # print(feat)
        if i % 10000 == 0:
            print(i, len(index_list))

        ids = np.argsort(-scores)
        # print(label_list[i], ids[0: 3], scores[ids[0: 3]])

        for k2 in range(len(top_retrv)):
            current_len = top_retrv[k2]
            for sort_id in range(current_len):
                lbl = ids[sort_id]
                if lbl == label_list[i]:
                    hit_count[k2] = hit_count[k2] + 1
                    break
    hit_count = hit_count / cnt_valid
    outstr = ''
    for k in range(len(top_retrv)):
        outstr = outstr + ' ' + str(hit_count[k])
    print(outstr)
    print('total: %d', cnt_valid)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fc', type=str, default='../../pretrain_weights/res50_fc7.npz',
                        help='word embedding type: [inception, res50]')
    parser.add_argument('--image_file', type=str, default='/scratch/yufeiy2/nell_data_tmp/val.txt',
                        help='list of image file')
    parser.add_argument('--feat_dir', type=str, default='../../feats/new/',
                        help='directory to save features')

    args = parser.parse_args()

    fc = np.load(args.fc)['classifier']
    print(fc)
    # exit()
    test(args.image_file, fc, args.feat_dir)