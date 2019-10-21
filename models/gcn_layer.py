import math
import torch
import torch.nn as nn

from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer = nn.Linear(in_features, out_features,bias=bias)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, x, adj):
        support = self.layer(x)
        output = torch.matmul(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
