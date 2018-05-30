import argparse
import json
import numpy as np
import pickle as pkl
from scipy import sparse
import os

from tensorflow.python import pywrap_tensorflow
from prepare_list import prepare_graph

# https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt
data_dir = '../../data/'


def convert_to_gcn_data(model_path, layer_name, offset, wv_file):
    save_dir = os.path.join(data_dir, '%s_%s' % (args.wv, args.fc))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Converting input')
    convert_input(wv_file, save_dir)
    print('Converting graph')
    convert_graph(save_dir)
    print('Converting label')
    convert_label(model_path, layer_name, save_dir, offset)
    print('Prepared data to %s' % save_dir)


def convert_input(wv_file, save_dir):
    with open(wv_file) as fp:
        feats = pkl.load(fp)
    feats = feats.tolist()
    sparse_feats = sparse.csr_matrix(feats)
    dense_feats = np.array(feats)

    sparse_file = os.path.join(save_dir, 'ind.NELL.allx')
    dense_file = os.path.join(save_dir, 'ind.NELL.allx_dense')
    with open(sparse_file, 'wb') as fp:
        pkl.dump(sparse_feats, fp)
    with open(dense_file, 'wb') as fp:
        pkl.dump(dense_feats, fp)

    print('Save feat in shape to', sparse_file, dense_file, 'with shape', dense_feats.shape)
    return


def convert_label(model_path, layer_name, save_dir, offset):  # get output's label and mask
    '''save visual classifier'''
    corresp_file = os.path.join(data_dir, 'list/corresp-all.json')  # 2-hops, 3-hops are also okay.
    with open(corresp_file) as fp:
        corresp_list = json.load(fp)

    def get_variables_in_checkpoint_file(file_name):
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map, reader

    var_keep_dic, reader = get_variables_in_checkpoint_file(model_path)
    for name in var_keep_dic:
        print(name, len(var_keep_dic[name]), var_keep_dic[name])
        if name == layer_name:
            print(name)
            print(reader.get_tensor(name).shape)
            fc = reader.get_tensor(name).squeeze()
            fc_dim = fc.shape[0]
            break

    fc_labels = np.zeros((len(corresp_list), fc_dim))
    print('fc dim', fc_labels.shape)
    for i in range(len(corresp_list)):
        class_id = corresp_list[i][0]
        if class_id == -1 or corresp_list[i][1] == 1:
            continue
        assert class_id < 1000
        fc_labels[i, :] = np.copy(fc[:, class_id + offset])

    test_index = []
    for i in range(len(corresp_list)):
        if corresp_list[i][0] == -1:
            test_index.append(-1)
        else:
            test_index.append(corresp_list[i][1])

    label_file = os.path.join(save_dir, 'ind.NELL.ally_multi')
    test_file = os.path.join(save_dir, 'ind.NELL.index')
    with open(label_file, 'wb') as fp:
        pkl.dump(fc_labels, fp)
    with open(test_file, 'wb') as fp:
        pkl.dump(test_index, fp)
    return


def convert_graph(save_dir):
    graph_file = os.path.join(data_dir, 'imagenet_graph.pkl')
    if not os.path.exists(graph_file):
        prepare_graph()
    save_file = os.path.join(save_dir, 'ind.NELL.graph')
    if os.path.exists(save_file):
        cmd = 'rm  %s' % save_file
        os.system(cmd)
    cmd = 'ln -s %s %s' % (graph_file, save_file)
    os.system(cmd)
    return


def parse_arg():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hop', type=str, default='2',
                        help='choice of unseen set: 2,3,all')
    parser.add_argument('--fc', type=str, default='res50',
                        help='choice: [inception,res50]')
    parser.add_argument('--wv', type=str, default='glove',
                        help='word embedding type: [glove, google, fasttext]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    if args.fc == 'inception':
        model_path = '../../pretrain_weights/inception_v1.ckpt'
        save_path = '../../pretrain_weights/inception_fc7.npz'
        layer_name = 'InceptionV1/Logits/Conv2d_0c_1x1/weights'
        offset = 1
    elif args.fc == 'res50':
        model_path = '../../pretrain_weights/resnet_v1_50.ckpt'
        save_path = '../../pretrain_weights/res50_fc7.npz'
        layer_name = 'resnet_v1_50/logits/weights'
        offset = 0
    else:
        raise NotImplementedError

    if args.wv == 'glove':
        wv_file = os.path.join(data_dir, 'word_embedding_model', 'glove_word2vec_wordnet.pkl')
    elif args.wv == 'google':
        wv_file = os.path.join(data_dir, 'word_embedding_model', 'google_word2vec_wordnet.pkl')
    elif args.wv == 'fasttext':
        wv_file = os.path.join(data_dir, 'word_embedding_model', 'fasttext_word2vec_wordnet.pkl')
    else:
        raise NotImplementedError

    convert_to_gcn_data(model_path, layer_name, offset, wv_file)
