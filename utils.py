import numpy as np
import scipy.sparse as sp
import torch
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


def load_data(dataset):
    data = sio.loadmat("data/{}.mat".format(dataset))
    features = data["Attributes"].todense()
    # features = data["Attributes"]
    adj = data["Network"].todense()
    labels = [label[0] for label in data['Label']]

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(np.array(labels))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels


def load_data2(dataset_source):
    data = sio.loadmat("data/{}.mat".format(dataset_source))
    features = data["Attributes"]
    adj = data["Network"]
    labels = data["Label"]

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #features = normalize(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    #adj = normalize_adj(normalize(adj + sp.eye(adj.shape[0]))+ sp.eye(adj.shape[0]))
    #features = preprocessing.normalize(features, norm='l2', axis=0)

    node_perm = np.random.permutation(labels.shape[0])
    num_train = int(0.8 * adj.shape[0])
    num_val = int(0.1 * adj.shape[0])
    idx_train = node_perm[:num_train]
    idx_train = node_perm
    idx_val = node_perm[num_train:num_train + num_val]
    idx_test = node_perm[num_train + num_val:]

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test




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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
