from __future__ import print_function
import argparse
import json
import numpy as np
import os

import pickle as pkl
import scipy.io as sio
import time


def test_imagenet_zero(fc_file_pred, has_train=1):
    with open(classids_file_retrain) as fp:
        classids = json.load(fp)

    with open(word2vec_file, 'rb') as fp:
        word2vec_feat = pkl.load(fp)

    testlist = []
    testlabels = []
    with open(vallist_folder) as fp:
        for line in fp:
            fname, lbl = line.split()
            assert int(lbl) >= 1000
            # feat_name = os.path.join(feat_folder, fname.replace('.JPEG', '.mat'))
            feat_name = os.path.join(feat_folder, fname.replace('.JPEG', '.npz'))
            if not os.path.exists(feat_name):
                print('not feature', feat_name)
                continue
            testlist.append(feat_name)
            testlabels.append(int(lbl))

    with open(fc_file_pred, 'rb') as fp:
        fc_layers_pred = pkl.load(fp)
    fc_layers_pred = np.array(fc_layers_pred)
    print('fc output', fc_layers_pred.shape)

    # remove invalid classes(wv = 0)
    valid_clss = np.zeros(22000)
    cnt_zero_wv = 0
    for j in range(len(classids)):
        if classids[j][1] == 1:
            twv = word2vec_feat[j]
            twv = twv / (np.linalg.norm(twv) + 1e-6)

            if np.linalg.norm(twv) == 0:
                cnt_zero_wv = cnt_zero_wv + 1
                continue
            valid_clss[classids[j][0]] = 1

    # process 'train' classes. they are possible candidates during inference
    cnt_zero_wv = 0
    labels_train, word2vec_train = [], []
    fc_now = []
    for j in range(len(classids)):
        tfc = fc_layers_pred[j]
        if has_train:
            if classids[j][0] < 0:
                continue
        else:
            if classids[j][1] == 0:
                continue

        if classids[j][0] >= 0:
            twv = word2vec_feat[j]
            if np.linalg.norm(twv) == 0:
                cnt_zero_wv = cnt_zero_wv + 1
                continue

            labels_train.append(classids[j][0])
            word2vec_train.append(twv)

            feat_len = len(tfc)
            tfc = tfc[feat_len - fc_dim: feat_len]
            fc_now.append(tfc)
    fc_now = np.array(fc_now)
    print('skip candidate class due to no word embedding: %d / %d:' % (cnt_zero_wv, len(labels_train) + cnt_zero_wv))
    print('candidate class shape: ', fc_now.shape)

    fc_now = fc_now.T
    labels_train = np.array(labels_train)
    print('train + test class: ', len(labels_train))

    topKs = [1]
    top_retrv = [1, 2, 5, 10, 20]
    hit_count = np.zeros((len(topKs), len(top_retrv)))

    cnt_valid = 0
    t = time.time()
    for j in range(len(testlist)):
        featname = testlist[j]

        if valid_clss[testlabels[j]] == 0:
            continue

        cnt_valid = cnt_valid + 1

        # matfeat = sio.loadmat(featname)
        matfeat = np.load(featname)
        matfeat = matfeat['feat']

        scores = np.dot(matfeat, fc_now).squeeze()

        scores = scores - scores.max()
        scores = np.exp(scores)
        scores = scores / scores.sum()

        ids = np.argsort(-scores)

        for k in range(len(topKs)):
            for k2 in range(len(top_retrv)):
                current_len = top_retrv[k2]
                for sort_id in range(current_len):
                    lbl = labels_train[ids[sort_id]]
                    if lbl == testlabels[j]:
                        hit_count[k][k2] = hit_count[k][k2] + 1
                        break
        if j % 10000 == 0:
            inter = time.time() - t
            print('processing %d / %d ' % (j, len(testlist)), ', Estimated time: ', inter / (j-1) * (len(testlist) - j))

    hit_count = hit_count / cnt_valid

    fout = open(fc_file_pred + '_result_pred_zero.txt', 'w')
    for j in range(len(topKs)):
        outstr = ''
        for k in range(len(top_retrv)):
            outstr = outstr + ' ' + str(hit_count[j][k])
        print(outstr)
        print('total: %d', cnt_valid)
        fout.write(outstr + '\n')
    fout.close()

    return hit_count


# global var
data_dir = '../data/list/'
classids_file_retrain = ""
word2vec_file = ""
vallist_folder = ""
fc_dim = 0
wv_dim = 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, default='model/wordnet_google_glove_feat_2048_1024_512_300',
                        help='path of model to test')
    parser.add_argument('--feat', type=str, default='../feats/res50/',
                        help='path of fc feature')
    parser.add_argument('--hop', type=str, default='2',
                        help='choice of unseen set: 2,3,all, separate with comma')
    parser.add_argument('--wv', type=str, default='glove',
                        help='word embedding type: [glove, google, fasttext]')
    parser.add_argument('--train', type=str, default='0,1',
                        help='contain train class or not')
    parser.add_argument('--fc', type=str, default='res50',
                        help='choice: [inception,res50]')

    args = parser.parse_args()

    print('-----------info-----------')
    print(args)
    print('--------------------------')

    if not os.path.exists(args.model):
        print('model does not exist: %s' % args.model)
        raise NotImplementedError

    if args.feat == None:
        print('please specify feature folder as --feat $path')
    feat_folder = args.feat

    if args.fc == 'inception':
        fc_dim = 1024
    elif args.fc == 'res50':
        fc_dim = 2048
    else:
        print('args.fc supports google (for Inception-v1)/ res50 (for Resnet-50)')
        raise ValueError

    if args.wv == 'glove':
        word2vec_file = '../data/word_embedding_model/glove_word2vec_wordnet.pkl'
    elif args.wv == 'google':
        word2vec_file = '../data/word_embedding_model/google_word2vec_wordnet.pkl'
    elif args.wv == 'fasttext':
        word2vec_file = '../data/word_embedding_model/fasttext_word2vec_wordnet.pkl'
    else:
        print('args.wv supports glove (for GloVe) / fast (for FastText)')
        raise ValueError

    hop_set = args.hop.split(',')
    train_set = args.train.split(',')
    results = []
    result_pool = []

    for hop in hop_set:
        for has_train in train_set:
            args.hop = hop
            args.train = int(has_train)

            param = 'Test Set: '
            if args.hop == '2':
                vallist_folder = os.path.join(data_dir, 'img-2-hops.txt')
                classids_file_retrain = os.path.join(data_dir, 'corresp-2-hops.json')
                param += '2-hops'
            elif args.hop == '3':
                vallist_folder = os.path.join(data_dir, 'img-3-hops.txt')
                classids_file_retrain = os.path.join(data_dir, 'corresp-3-hops.json')
                param += '3-hops'
            elif args.hop == 'all':
                vallist_folder = os.path.join(data_dir, 'img-all.txt')
                classids_file_retrain = os.path.join(data_dir, 'corresp-all.json')
                param += 'All'

            if int(has_train) == 1:
                param += ' (+ 1K)'
            param += ' ,with word embedding %s' % args.wv
            print('\nEvaluating %s ...\nPlease be patient for it takes a few minutes...' % param)
            res = test_imagenet_zero(fc_file_pred=args.model, has_train=args.train)
            output = ['{:.1f}'.format(i * 100) for i in res[0]]
            result_pool.append(output)

            results.append((args.model, output, param))
            print('----------------------')
            print('model : ', args.model)
            print('param : ', param)
            print('result: ', output)
            print('----------------------')

    print('\n======== summary: ========')
    for i in range(len(results)):
        print('%s: ' % str(results[i][2]))
        print('%s' % str(results[i][1]))
    print('for model %s' % args.model)
