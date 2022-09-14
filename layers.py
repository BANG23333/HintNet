import math
import torch
#from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConv(Module):

    def __init__(self, in_f_dim, out_f_dim, bias=True):
        super(GraphConv, self).__init__()
        self.in_f_dim = in_f_dim
        self.out_f_dim = out_f_dim
        self.weight = torch.nn.parameter.Parameter(torch.FloatTensor(in_f_dim, out_f_dim))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.FloatTensor(out_f_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        batch, len_x, len_y, num_f = input.shape
        input = input.reshape(batch, len_x*len_y, num_f)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        output = output.reshape((batch, len_x, len_y, num_f))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_f_dim) + ' -> ' \
            + str(self.out_f_dim) + ')'
