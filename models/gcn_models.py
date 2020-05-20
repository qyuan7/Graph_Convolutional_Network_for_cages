import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gcn_layer import GraphConvolution


class gcn(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.5, n_class=1):
        super().__init__()
        self.pre = nn.Linear(input_dim, 64)
        self.gc1 = GraphConvolution(64, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gc3 = GraphConvolution(hidden_dim, n_class)
        self.dropout = dropout
        self.n_class = n_class

    @staticmethod
    def init_adj(adj,method='spectral'):
        if method == 'spectral':
            D = np.array(np.sum(adj, axis=0))[0]
            D_sqrt = D ** -0.5
            D_sqrt = np.diag(D_sqrt)
            adj_hat = D_sqrt * adj * D_sqrt
            adj_hat = torch.from_numpy(adj_hat).float()
        else:
            adj_hat = torch.from_numpy(adj).float() 
        adj_hat = adj_hat.clone().detach().requires_grad_(True)
        return adj_hat

    def forward(self, x, adj):
        adj = self.init_adj(adj)
        x = F.relu(self.pre(x))
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, self.training)
        #x = F.relu(self.gc2(x, adj))
        #x = F.dropout(x, self.dropout, self.training)
        x = self.gc3(x,adj)
        x = torch.sum(x, 1)
        #if self.n_class > 1:
        #    x = F.log_softmax(x, dim=1)
        return x
