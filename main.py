#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from H_Grid_DBscan import Grid_DBscan
from utils import *
from InitialDatabase import *
from model import *
from train_and_test import *

torch.cuda.get_device_name()


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_printoptions(edgeitems=100)


# In[3]:


# X data shape [batch, input_len, x_len_of_grid, y_len_of_grid, features]
# Y data shape [batch, output_len (1), x_len_of_grid, y_len_of_grid]

# loading traninig and validating data.
X = np.load(open('E:/HintNet/Data/X_Iowa_2016-2017_7_1.npy', 'rb')) # Specify your data path.
Y = np.load(open('E:/HintNet/Data/Y_Iowa_2016-2017_7_1.npy', 'rb')) # Specify your data path.
X, Xv, Y, Yv = train_test_split(X, Y, test_size=0.20, random_state=11) # 31 11

x_len = X.shape[2] # x_len of your gird
y_len = X.shape[3] # y_len of your grid

mask_table = np.load(open('mask_128_64.npy', 'rb')) # if no mask available use code below to intialize a one matrix
mask = mask_table

# intialize a one matrix
"""
mask = np.ones((x_len, y_len))
mask_table = mask
"""

# loading testing data.
Xt = np.load(open('E:/HintNet/Data/X_Iowa_2018_7_1.npy', 'rb')) # Specify your data path.
Yt = np.load(open('E:/HintNet/Data/Y_Iowa_2018_7_1.npy', 'rb')) # Specify your data path.


# In[4]:


#Hyperparameter settings

itera = 10
threshold = 10
increment = 1
eps=1
min_points=1
min_density=1

acc_matrix = np.zeros((Y.shape[2], Y.shape[3]))
for b in range(len(Y)):
    acc_matrix += Y[b][0]

#Hierarchical Grid Clustering 
result_map = Grid_DBscan(eps=eps, min_points=min_points, min_density=min_density, acc_matrix=acc_matrix, mask=mask, plot=False).map_scan(itera, threshold, increment)
lables = mannual_merge_levels(result_map).astype("float")


# In[5]:


coors_matrix = []
scale = list(range(1,int(np.max(lables))+1))

for sca in scale:
    temp = []
    for x in range(128):
        for y in range(64):
            if lables[x][y] == sca and mask_table[x][y] == 1:
                temp.append([x, y])
    coors_matrix.append(temp)
coors_matrix = np.array(coors_matrix, dtype=object)

print("# of Grids in each level")
for i in range(len(scale)):
    print("level " + str(i) +": " + str(len(coors_matrix[i])))
    


# In[6]:


print("Start Model traininig ...")

model_num = ""

ctr = len(scale)
total = 0
for i in range(len(scale)-1, -1, -1):
    print("training model for level: " + str(i+1))
    total += len(coors_matrix[i])
    transfer_model = None if ctr == len(scale) else str(ctr + 1)
    sig_loss, sig_model = train_model(X, Y, Xv, Yv, device, coors_matrix[i], model_num, transfer_model)
    save_model(sig_model, str(ctr), model_num, device)
    print("Finished model_" + str(i+1) + " saved")
    ctr -= 1


# In[7]:


print("Start Model testing ...")

x_len = Xv.shape[2]
y_len = Xv.shape[3]
    
out = []

partition_map = lables

pred_map = np.full(Yv.shape, 0.0)
ctr_map = np.zeros(Yv.shape)

plot_map(partition_map)

for i in np.unique(partition_map):
    print("testing model for level: " + str(i+1))
    if i == 0:
        continue

    coors_matrix = []
    for x in range(x_len):
        for y in range(y_len):
            if partition_map[x][y] == i:
                coors_matrix.append([x, y])
    coors_matrix = np.array(coors_matrix)
    pred_map, ctr_map = test_model(Xt, Yt, device, coors_matrix, str(int(i)), pred_map, ctr_map, model_num)

ctr_map = np.where(ctr_map==0, 1, ctr_map)
final_pred = pred_map/ctr_map
Yv = np.where(mask==1, Yv, 0.0)

print("testing finished")
print("MSE: " + str(MSE_np(final_pred, Yv)))

