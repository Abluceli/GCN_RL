import torch.nn as nn
import torch.nn.functional as F
from duplicate.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout
        self.embeddings = None

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        #x = F.normalize(x, p=2, dim=-1)
        self.embeddings = x
        x = self.gc2(x, adj)
        return x


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        # self.fc1 = nn.Linear(nfeat, nhid)
        self.fc1 = nn.Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.log_softmax(x, dim=1)
        return x



