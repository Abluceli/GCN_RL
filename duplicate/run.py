from __future__ import division
from __future__ import print_function

import argparse

import torch.optim as optim

from duplicate.models import GCN
from duplicate.train_helper import *


def get_reward(node_id, labels):
    return labels[node_id]


def train(args):
    # Load data
    adj, features, labels = load_data("BlogCatalog")

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    """add your rl model"""
    RL_model = None

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()

    # Training
    for epoch in range(args.epochs):
        # GCN
        for _ in range(args.embedding_step):
            ae_loss = train_gcn(model, optimizer, features, adj)
            print("AE loss: {}".format(ae_loss))
        print("GCN training done...")

        state = model.embeddings  # number of nodes x embedding dimension

        # RL
        """add more """
        selected_node_id = RL_model.act(state)
        reward = get_reward(selected_node_id, labels)
        RL_model.update(selected_node_id, state, reward)

        # update state



    model.gc1.reset_parameters()
    model.gc2.reset_parameters()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--embedding_step', type=int, default=16,
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


