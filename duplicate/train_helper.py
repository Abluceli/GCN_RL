from __future__ import division
from __future__ import print_function

import torch.nn.functional as F

from duplicate.utils import *


def train_gcn(model, optimizer, features, adj):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # link_logit = torch.sigmoid(torch.mm(output, torch.transpose(output, 1, 0)))
    # adj_label = adj.to_dense()
    # adj_label = adj_label.reshape(-1, 1)
    # adj_label = torch.where(adj_label == 0, torch.zeros_like(adj_label), torch.ones_like(adj_label))
    # link_logit = link_logit.reshape(-1, 1)
    #
    # link_predict_num = torch.where(link_logit >= 0.5, torch.zeros_like(link_logit), torch.ones_like(link_logit))
    # link_predict_num = link_predict_num.detach().cpu().numpy().astype(np.int)
    # adj_label_num = adj_label.cpu().numpy().astype(np.int)
    # acc = accuracy_score(adj_label_num, link_predict_num)
    # f1 = f1_score(adj_label_num, link_predict_num)
    # one_count_pre = np.sum(link_predict_num)
    # zero_count_pre = len(link_predict_num) - np.sum(link_predict_num)
    # one_count_ac = np.sum(adj_label_num)
    # zero_count_ac = len(adj_label_num) - np.sum(adj_label_num)
    # print("Acc {}, F1 {}".format(acc, f1))
    # print("Prediction One Count {}, Zero Count {}".format(one_count_pre, zero_count_pre))
    # print("Actual One Count {}, Zero Count {}".format(one_count_ac, zero_count_ac))
    # reconstruction_loss = F.binary_cross_entropy(link_logit, adj_label)



    # reconstruction phase
    reconstruction_errors = F.mse_loss(output, features, reduction='none')
    reconstruction_errors = torch.mean(reconstruction_errors, dim=1)
    reconstruction_loss = torch.mean(reconstruction_errors)

    reconstruction_loss.backward()
    optimizer.step()

    return reconstruction_loss


def train_rl(state):

    selected_node_id = 1
    """rl model train"""
    return selected_node_id
