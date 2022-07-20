"""
# -*- coding:utf-8 -*-
# Version:Python3.8.3
# @Time:2022/7/20 7:39
# @Author:Lzy
# @File:B_GCNs_model.py
# @Software:PyCharm
# Code Description: 
"""

import math
import torch.nn as nn
import torch.nn.functional as F
from B_GCNs_util import *


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if use_bias:
            self.use_bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('use_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.use_bias is not None:
            self.use_bias.data.uniform_(-stdv, stdv)

    def forward(self, input_feature, adjacency,  prior_probability_tensor):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        support = torch.mm(input_feature, self.weight)
        output = torch.spmm(adjacency, support)
        if output.shape[1] == prior_probability_tensor.shape[1]:
            prior_probability_output = torch.mul(prior_probability_tensor, output)
            if self.use_bias is not None:
                # prior_probability_output += self.bias.to(device)
                return prior_probability_output + self.use_bias
            else:
                return prior_probability_output
        else:
            if self.use_bias is not None:
                return output + self.use_bias
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# ## 模型定义
class B_Gcns_Net(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(B_Gcns_Net, self).__init__()
        self.B_gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.B_gcn2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, feature, adjacency, prior_probability_tensor):
        h = F.relu(self.B_gcn1(feature, adjacency, prior_probability_tensor))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.B_gcn2(h, adjacency, prior_probability_tensor)
        return F.log_softmax(h, dim=1)
