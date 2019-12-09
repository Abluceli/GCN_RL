import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import scipy.io as sio
import random
from sklearn import preprocessing


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def load_data(dataset):
    data = sio.loadmat("data/{}.mat".format(dataset))
    features = data["Attributes"]
    adj = data["Network"]
    labels = [label[0] for label in data['Label']]

    adj_norm = sparse_to_tuple(normalize_adj(adj + sp.eye(adj.shape[0])))
    features = sparse_to_tuple(features)

    return adj_norm, features, labels


def load_data_gcn(dataset, selected_id_list):
    data = sio.loadmat("../data/{}.mat".format(dataset))
    features = data["Attributes"]
    adj = data["Network"]
    labels = data['Label']

    node_num = len(labels)
    id_list = [i for i in range(0, node_num)]

    remain_list = list(set(id_list).difference(set(selected_id_list)))
    features_new = features[remain_list]
    labels_new = labels[remain_list]
    adj_new = adj[remain_list, :][:, remain_list]

    node_perm = np.random.permutation(labels.shape[0])
    num_train = int(0.1 * adj.shape[0])
    num_val = int(0.2 * adj.shape[0])
    idx_train = node_perm[:num_train]
    idx_val = node_perm[num_train:num_train + num_val]
    idx_test = node_perm[num_train + num_val:]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_rgcn(dataset):
    data = sio.loadmat("data/{}.mat".format(dataset))
    features = data["Attributes"]
    adj = data["Network"]
    labels = [label[0] for label in data['Label']]

    adj_1 = adj + sp.eye(adj.shape[0])
    adj_2 = sp.eye(adj.shape[0])
    adj_norm_1 = sparse_to_tuple(normalize_adj(adj_1))
    adj_norm_2 = sparse_to_tuple(normalize_adj(adj_2))

    adj_1 = adj_1.tolil()
    adj_2 = adj_2.tolil()

    features = sparse_to_tuple(features)

    return adj_norm_1, adj_norm_2, adj_1, adj_2, features, labels


def get_placeholder():
    placeholders = {
        'adj_1': tf.sparse_placeholder(tf.float32),
        'adj_2': tf.sparse_placeholder(tf.float32),
        'features': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }
    return placeholders


def construct_feed_dict(adj_1, adj_2, features, dropout, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['adj_1']: adj_1})
    feed_dict.update({placeholders['adj_2']: adj_2})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['dropout']: dropout})
    return feed_dict


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def update_adj(selected_node_id, adj_1, adj_2):

    # adj_1 = adj_1.todense()
    # adj_2 = adj_2.todense()
    adj_2[selected_node_id, :] = adj_1[selected_node_id, :]
    adj_2[:, selected_node_id] = adj_1[:, selected_node_id]
    adj_1[selected_node_id, :] = 0
    adj_1[:, selected_node_id] = 0
    adj_1[selected_node_id, selected_node_id] = 1

    # adj_1 = sp.csr_matrix(adj_1)
    # adj_2 = sp.csr_matrix(adj_2)

    adj_norm_1 = sparse_to_tuple(adj_1)
    adj_norm_2 = sparse_to_tuple(adj_2)

    return adj_norm_1, adj_norm_2, adj_1, adj_2











