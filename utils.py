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


def accuracy(vector_x, vector_y):

  # torch.Size([283200])
  new_v = vector_x - vector_y
  new_v = torch.abs(new_v)
  new_v = torch.sum(new_v).data.cpu().numpy()

  return new_v/len(vector_x)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def coor_matrix_filter(coors_matrix):
    final = []
    for i in range(len(coors_matrix)):
        out = []
        for each in coors_matrix[i]:
            if each[0] < x_range[0] or each[0] >= x_range[1] or each[1] < y_range[0] or each[1] >= y_range[1]:
                continue
            out.append(each)
        final.append(out)
    final = np.array(final, dtype=object)
    return final


S = 3


def adj_generator(temp_x, temp_y, cell_x, cell_y, window_size):
    win_x = temp_x - cell_x
    win_y = temp_y - cell_y

    out = []

    for x in range(window_size):
        for y in range(window_size):
            try:
                value = adj_dict[str(temp_x) + "-" + str(temp_y)][str(win_x + x) + "-" + str(win_y + y)]
            except:
                value = 0
            out.append(value)

    return out


def generate_small_cell(x1, y1, x2, y2, S, limitx, limity):
    if x1 > S * 2 or y1 > S * 2:
        print("S error")
    if x2 >= limitx or y2 >= limity:
        print("Limit error")
    origin = [x2 - S, y2 - S]
    x1 = x1 + origin[0]
    y1 = y1 + origin[1]
    if 0 > x1 or x1 >= limitx:
        x1 = -1
    if 0 > y1 or y1 >= limity:
        y1 = -1
    return x1, y1


# temp = generate_small_cell(6, 6, 31, 31, S, 32)
# print(temp)

def generate_samples_by_cells(coors_matrix, X, Y):
    batch, period, len_x, len_y, features = X.shape

    coors = coors_matrix
    num_coor = len(coors)
    total_model = []

    adj_matrix_total = []

    for each in coors:
        x, y = each[0], each[1]
        window_size = 2 * S + 1
        single = np.zeros((batch, period, window_size, window_size, features))
        adj_matrix = np.zeros((batch, window_size ** 2, window_size ** 2))

        for cell_x in range(2 * S + 1):
            for cell_y in range(2 * S + 1):

                temp_x, temp_y = generate_small_cell(cell_x, cell_y, x, y, S, 128, 64)
                adj_matrix[:, cell_x * window_size + cell_y] = adj_generator(temp_x, temp_y, cell_x, cell_y,
                                                                             window_size)

                if temp_x < 0 or temp_y < 0:
                    continue

                single[:, :, cell_x, cell_y] = X[:, :, temp_x, temp_y]

        total_model.append(single)
        adj_matrix_total.append(adj_matrix)

    total_model = np.array(total_model)
    adj_matrix_total = np.array(adj_matrix_total)
    total_model = total_model.reshape(num_coor * batch, period, window_size, window_size, features)
    adj_matrix_total = adj_matrix_total.reshape(num_coor * batch, window_size ** 2, window_size ** 2)

    new_X = total_model

    batch, period, len_x, len_y = Y.shape

    coors = coors_matrix
    num_coor = len(coors)
    total_model = []
    for each in coors:
        x, y = each[0], each[1]
        single = np.zeros((batch, period))

        single[:, :] = Y[:, :, x, y]

        total_model.append(single)
    total_model = np.array(total_model)
    total_model = total_model.reshape(num_coor * batch, period)
    new_Y = total_model

    return new_X, new_Y, adj_matrix_total


def generate_samples_by_cells_test(coors_matrix, X, Y):
    batch, period, len_x, len_y, features = X.shape

    coors = coors_matrix
    num_coor = len(coors)
    total_model = []

    adj_matrix_total = []

    for each in coors:
        x, y = each[0], each[1]
        window_size = 2 * S + 1
        single = np.zeros((batch, period, window_size, window_size, features))
        adj_matrix = np.zeros((batch, window_size ** 2, window_size ** 2))

        for cell_x in range(2 * S + 1):
            for cell_y in range(2 * S + 1):

                temp_x, temp_y = generate_small_cell(cell_x, cell_y, x, y, S, 128, 64)
                adj_matrix[:, cell_x * window_size + cell_y] = adj_generator(temp_x, temp_y, cell_x, cell_y,
                                                                             window_size)

                if temp_x < 0 or temp_y < 0:
                    continue

                single[:, :, cell_x, cell_y] = X[:, :, temp_x, temp_y]

        total_model.append(single)
        adj_matrix_total.append(adj_matrix)

    total_model = np.array(total_model)
    adj_matrix_total = np.array(adj_matrix_total)
    total_model = total_model.reshape(num_coor * batch, period, window_size, window_size, features)
    adj_matrix_total = adj_matrix_total.reshape(num_coor * batch, window_size ** 2, window_size ** 2)

    new_X = total_model

    batch, period, len_x, len_y = Y.shape

    coors = coors_matrix
    num_coor = len(coors)
    total_model = []
    total_xy = []

    for each in coors:
        x, y = each[0], each[1]
        single = np.zeros((batch, period))
        single_xy = np.zeros((batch, period, 2))

        single[:, :] = Y[:, :, x, y]
        single_xy[:, :, 0] = x
        single_xy[:, :, 1] = y

        total_model.append(single)
        total_xy.append(single_xy)

    total_model = np.array(total_model)
    total_xy = np.array(total_xy)
    total_model = total_model.reshape(num_coor * batch, period)
    total_xy = total_xy.reshape(num_coor * batch, period, 2)

    new_Y = total_model

    return new_X, new_Y, adj_matrix_total, total_xy


def plot_loss(train_loss_arr, valid_loss_arr):
    fig, ax1 = plt.subplots(figsize=(20, 10))

    ax1.plot(train_loss_arr, 'k', label='training loss')
    ax1.plot(valid_loss_arr, 'g', label='validation loss')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()

    ax2.legend(loc=2)
    plt.show()
    plt.clf()


def MSE_torch(prediction, true_value):
    prediction = prediction.flatten(0)
    true_value = true_value.flatten(0)

    # prediction = torch.round(prediction)

    mse = torch.sum(torch.square(prediction - true_value) / len(prediction))

    return mse


def RMSE_torch(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()

    # prediction = np.round(prediction)

    rmse = torch.sqrt(torch.sum(torch.square(prediction - true_value) / len(prediction)))

    return rmse


def MAE_torch(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()
    mae = torch.abs(prediction - true_value)
    return torch.sum(mae) / len(prediction)


def accuracy(vector_x, vector_y):
    vector_x = vector_x.flatten()
    vector_y = vector_y.flatten()
    new_v = vector_x - vector_y
    new_v = np.abs(new_v)
    new_v = np.sum(new_v)
    return 1 - new_v / len(vector_x)


def MSE_np(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()

    # prediction = np.round(prediction)

    mse = np.sum(np.square(prediction - true_value)) / len(prediction)

    return mse


def MAE_np(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()
    mae = np.abs(prediction - true_value)
    return np.sum(mae) / len(prediction)


def RMSE_np(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()

    # prediction = np.round(prediction)

    rmse = np.sqrt(np.sum(np.square(prediction - true_value) / len(prediction)))

    return rmse


def precision(predicted, true_y):
    TruePositives = 0
    FalsePositives = 0
    for i in range(true_y.shape[0]):
        for t in range(true_y.shape[1]):
            true = true_y[i][t]
            pred = predicted[i][t]
            for x in range(len(true)):
                for y in range(len(true[0])):
                    if pred[x][y] == 0 and true[x][y] == 0:
                        TruePositives += 1
                    elif pred[x][y] == 0 and true[x][y] == 1:
                        FalsePositives += 1
    precision = TruePositives / (TruePositives + FalsePositives)
    return precision


def recall(predicted, true_y):
    # Recall Recall = TruePositives / (TruePositives + FalseNegatives)
    TruePositives = 0
    FalseNegatives = 0
    for i in range(true_y.shape[0]):
        for t in range(true_y.shape[1]):
            true = true_y[i][t]
            pred = predicted[i][t]
            for x in range(len(true)):
                for y in range(len(true[0])):
                    if pred[x][y] == 0 and true[x][y] == 0:
                        TruePositives += 1
                    elif pred[x][y] == 1 and true[x][y] == 0:
                        FalseNegatives += 1
    recall = TruePositives / (TruePositives + FalseNegatives)
    return recall


def remove_from_arr_to_arr(a, b):
    indices = np.argwhere(np.isin(a, b))
    a = np.delete(a, indices)
    return a


def data_split(x_arrange, num_X_bag, X, Y):
    if len(x_arrange) < num_X_bag:
        return np.NaN, X[x_arrange], Y[x_arrange]

    idx = np.random.choice(x_arrange, num_X_bag, replace=False)
    x_sample, y_sample = X[idx], Y[idx]
    x_arrange = remove_from_arr_to_arr(x_arrange, idx)

    return x_arrange, x_sample, y_sample


def data_split_no_random(x_arrange, num_X_bag, X, Y):
    if len(x_arrange) < num_X_bag:
        return np.NaN, X[x_arrange], Y[x_arrange]

    idx = x_arrange[:num_X_bag]
    x_sample, y_sample = X[idx], Y[idx]
    x_arrange = remove_from_arr_to_arr(x_arrange, idx)

    return x_arrange, x_sample, y_sample


# model = NeuralNet()
# model.cuda()

def save_model(model, path, rd):
    PATH = "E:/HintNet/models/model" + str(rd) + "_" + str(path) + ".pt"
    torch.save(model.state_dict(), PATH)


def load_model(model, path, rd):
    PATH = "E:/HintNet/models/model" + str(rd) + "_" + str(path) + ".pt"
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
    model.to(device)
    return model


def speak(text):
    from win32com.client import Dispatch

    speak = Dispatch("SAPI.SpVoice").Speak

    speak(text)

def assign_scale(num):
    if num < 150:
        return 1
    elif num <300:
        return 1
    elif num <500:
        return 1
    elif num <1000:
        return 1
    else:
        return 1


def flip_num_matrix(target):
    arr = np.unique(target)
    arr = np.sort(arr)
    out = np.zeros(target.shape)

    temp_dict = {}

    for i in range(len(arr)):
        temp_dict[arr[i]] = arr[len(arr) - i - 1]

    for x in range(target.shape[0]):
        for y in range(target.shape[1]):
            out[x][y] = temp_dict[target[x][y]]

    return out


def plot_map(out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.heatmap(out.transpose(), robust=False, annot=False, cbar=True)
    ax.invert_yaxis()
    plt.axis("off")
    plt.show()


def mannual_merge_levels(result_map):

    result_map = np.where(result_map==1, 1, result_map)
    result_map = np.where(result_map==2, 1, result_map)
    result_map = np.where(result_map==3, 2, result_map)
    result_map = np.where(result_map==4, 2, result_map)
    result_map = np.where(result_map==5, 3, result_map)
    result_map = np.where(result_map==6, 3, result_map)
    result_map = np.where(result_map==7, 4, result_map)
    result_map = np.where(result_map==8, 4, result_map)
    result_map = np.where(result_map==9, 5, result_map)
    result_map = np.where(result_map==10, 5, result_map)
    result_map = np.where(result_map==11, 6, result_map)

    return result_map