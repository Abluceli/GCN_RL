from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import *
from sklearn.svm import SVC

from train_helper import *


def get_reward(node_id, labels):
    return labels[node_id]


def train(args):
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data2("BlogCatalog")

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    model2 = MLP(nfeat=features.shape[1],
                nhid=args.hidden, nclass=6,
                dropout=args.dropout)
    optimizer2 = optim.Adam(model2.parameters(),
                           lr=args.lr)

    """add your rl model"""
    RL_model = None

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        model2.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Training

    for _ in range(args.embedding_step):
        ae_loss = train_gcn(model, optimizer, features, adj)
        embeddings = model.embeddings
        print("AE loss: {}".format(ae_loss))
    print("GCN training done...")
    embeddings = torch.FloatTensor(embeddings.detach().cpu().numpy()).cuda()

    for epoch in range(args.epochs):
        # GCN
        model2.train()
        optimizer2.zero_grad()
        output = model2.forward(embeddings[idx_train])
        loss_train2 = F.nll_loss(output, labels[idx_train])
        acc_train = accuracy(output, labels[idx_train])
        print("train acc: {} Loss {}".format(acc_train, loss_train2))
        loss_train2.backward()
        optimizer2.step()


    model2.eval()
    output = model2.forward(model.embeddings[idx_test])
    acc = accuracy(output, labels[idx_test])
    print("test acc: {}".format(acc_train))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--embedding_step', type=int, default=100,
                        help='Number of GCN embedding step.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train(args)


