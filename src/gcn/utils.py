import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sample_mask_sigmoid(idx, h, w):
    """Create mask."""
    mask = np.zeros((h, w))
    matrix_one = np.ones((h, w))
    mask[idx, :] = matrix_one[idx, :]
    return np.array(mask, dtype=np.bool)


def load_data_vis_multi(dataset_str, use_trainval, feat_suffix, label_suffix='ally_multi'):
    """Load data."""
    names = [feat_suffix, label_suffix, 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.NELL.{}".format(dataset_str, names[i]), 'rb') as f:
            print("{}/ind.NELL.{}".format(dataset_str, names[i]))
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    allx, ally, graph = tuple(objects)
    train_test_mask = []
    with open("{}/ind.NELL.index".format(dataset_str), 'rb') as f:
        train_test_mask = pkl.load(f)

    features = allx  # .tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.array(ally)

    idx_test = []
    idx_train = []
    idx_trainval = []

    if use_trainval == True:
        for i in range(len(train_test_mask)):

            if train_test_mask[i] == 0:
                idx_train.append(i)
            if train_test_mask[i] == 1:
                idx_test.append(i)

            if train_test_mask[i] >= 0:
                idx_trainval.append(i)
    else:
        for i in range(len(train_test_mask)):

            if train_test_mask[i] >= 0:
                idx_train.append(i)
            if train_test_mask[i] == 1:
                idx_test.append(i)

            if train_test_mask[i] >= 0:
                idx_trainval.append(i)

    idx_val = idx_test

    train_mask = sample_mask_sigmoid(idx_train, labels.shape[0], labels.shape[1])
    val_mask = sample_mask_sigmoid(idx_val, labels.shape[0], labels.shape[1])
    trainval_mask = sample_mask_sigmoid(idx_trainval, labels.shape[0], labels.shape[1])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_trainval = np.zeros(labels.shape)

    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_trainval[trainval_mask] = labels[trainval_mask]

    return adj, features, y_train, y_val, y_trainval, train_mask, val_mask, trainval_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def preprocess_features_dense(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features_dense2(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    div_mat = sp.diags(rowsum)

    return features, div_mat


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def create_config_proto():
    """Reset tf default config proto"""
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 0
    config.gpu_options.force_gpu_compatible = True
    # config.operation_timeout_in_ms=8000
    config.log_device_placement = False
    return config
