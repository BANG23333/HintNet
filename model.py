import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random
import seaborn as sns
import matplotlib
import pickle
from layers import GraphConv
from sklearn.model_selection import train_test_split
from H_Grid_Clustering import Grid_DBscan
from utils import *
from InitialDatabase import *

# S2 = 3
class NeuralNet(nn.Module):
    def __init__(self, num_temporal, num_spatial, num_spatial_tempoal, map_size, input_len):
        super(NeuralNet, self).__init__()
        self.num_temporal = num_temporal
        self.num_spatial = num_spatial
        self.num_spatial_tempoal = num_spatial_tempoal
        self.map_size = map_size
        self.input_len = input_len

        cnn_list = []
        # cnn_list2 = []
        # cnn_list3 = []

        for i in range(0, 7):
            cnn_list.append(GraphConv(num_spatial_tempoal, num_spatial_tempoal))
            # cnn_list2.append(GraphConvolution(num_spatial_tempoal, num_spatial_tempoal))
            # cnn_list3.append(GraphConvolution(num_spatial_tempoal, num_spatial_tempoal))

        self.cnn3d = nn.ModuleList(cnn_list)
        # self.cnn3d_2 = nn.ModuleList(cnn_list2)
        # self.cnn3d_3 = nn.ModuleList(cnn_list3)

        # self.bn1 = nn.BatchNorm2d(15)
        # self.bn2 = nn.BatchNorm2d(15)

        # self.three_d_cnn = nn.Conv3d(in_channels=num_spatial_tempoal, out_channels=num_spatial_tempoal, kernel_size = (1, S2, S2), padding="same")
        # self.two_d_cnn = nn.Conv2d(in_channels=num_spatial, out_channels=num_spatial, kernel_size = (S2, S2), padding="same")
        self.two_d_cnn = GraphConv(num_spatial, num_spatial)
        # self.two_d_cnn2 = GraphConvolution(num_spatial, num_spatial)

        self.LSTM = nn.LSTM((num_temporal + num_spatial_tempoal), hidden_size=num_temporal + num_spatial_tempoal,
                            batch_first=True)

        self.FC = nn.Linear(map_size * map_size * num_spatial_tempoal * input_len, num_spatial_tempoal * input_len)
        self.FC2 = nn.Linear(map_size * map_size * num_spatial, num_spatial)
        self.FC3 = nn.Linear(num_temporal + num_spatial + num_spatial_tempoal, 1)

    def forward(self, x, adj, future_seq=0):

        temporal_view = x[:, :, 0, 0, 0:self.num_temporal]
        spatial_view = x[:, 0, :, :, self.num_temporal:self.num_temporal + self.num_spatial]
        spatial_tempoal_view = x[:, :, :, :,
                               self.num_temporal + self.num_spatial:self.num_temporal + self.num_spatial + self.num_spatial_tempoal]

        # spatial_tempoal_view = spatial_tempoal_view.permute(0, 4, 1, 2, 3)
        # spatial_tempoal_view = self.three_d_cnn(spatial_tempoal_view).

        for i in range(7):
            # spatial_tempoal_view[:, i] = self.cnn3d[i](spatial_tempoal_view[:, i], torch.zeros((x.shape[0], 25, 25), dtype=torch.float32, device=torch.device('cuda:0')))
            spatial_tempoal_view[:, i] = self.cnn3d[i](spatial_tempoal_view[:, i], adj)
            # spatial_tempoal_view[:, i] = F.relu(spatial_tempoal_view[:, i])
            # spatial_tempoal_view[:, i] = self.bn1(spatial_tempoal_view[:, i])
            # spatial_tempoal_view[:, i] = self.cnn3d_2[i](spatial_tempoal_view[:, i], adj)
            # spatial_tempoal_view[:, i] = F.relu(spatial_tempoal_view[:, i])
            # spatial_tempoal_view[:, i] = self.cnn3d_3[i](spatial_tempoal_view[:, i], adj)
            # spatial_tempoal_view[:, i] = F.relu(spatial_tempoal_view[:, i])
            # spatial_tempoal_view[:, i] = self.bn2(spatial_tempoal_view[:, i])

        spatial_tempoal_view = F.relu(spatial_tempoal_view)

        # spatial_tempoal_view = spatial_tempoal_view.permute(0, 2, 3, 4, 1)

        # spatial_view = spatial_view.permute(0, 3, 1, 2)
        # spatial_view = self.two_d_cnn(spatial_view, torch.zeros((x.shape[0], 25, 25), dtype=torch.float32, device=torch.device('cuda:0')))
        spatial_view = self.two_d_cnn(spatial_view, adj)
        spatial_view = F.relu(spatial_view)
        # spatial_view = self.two_d_cnn2(spatial_view, adj)
        # spatial_view = F.relu(spatial_view)

        # spatial_view = spatial_view.permute(0, 2, 3, 1)

        spatial_tempoal_view = self.FC(spatial_tempoal_view.flatten(1))
        spatial_tempoal_view = torch.reshape(spatial_tempoal_view, (len(x), self.input_len, self.num_spatial_tempoal))

        merged_two_view = torch.cat((spatial_tempoal_view, temporal_view), 2)

        current, (h_n, c_n) = self.LSTM(merged_two_view)

        merged_two_view = h_n.permute(1, 0, 2).flatten(1)
        spatial_view = self.FC2(spatial_view.flatten(1))

        final_view = torch.cat((merged_two_view, spatial_view), 1)
        final_view = self.FC3(final_view)
        final_view = F.relu(final_view)

        return final_view