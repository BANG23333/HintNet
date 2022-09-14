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
from model import *

torch.cuda.get_device_name()


def train_model(X, Y, Xv, Yv, device, coors_matrix, k, transfer_model=None):
    iterations = assign_scale(len(coors_matrix))
    batch_size = 1024
    criterion = nn.MSELoss()
    model = NeuralNet(5, 29, 13, S * 2 + 1, 7).to(device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_arr = []
    valid_loss_arr = []

    if transfer_model != None:
        model = load_model(model, transfer_model, k, device)

    all_results = {}
    best_result = math.inf
    best_ctr = 0
    best_model = model

    for echo in range(100):

        avg_train_loss = []
        avg_valid_loss = []
        ctr = 0
        ctrv = 0
        x_arrange = np.arange(len(X))
        xv_arrange = np.arange(len(Xv))

        num_X_bag = int(len(X) / iterations)
        num_Xv_bag = int(len(Xv) / iterations)

        for i in range(iterations):

            if i == iterations - 1:
                num_Xv_bag = np.inf
                num_X_bag = np.inf

            x_arrange, x_sample, y_sample = data_split(x_arrange, num_X_bag, X, Y)
            xv_arrange, xv_sample, yv_sample = data_split(xv_arrange, num_Xv_bag, Xv, Yv)

            cur_X, cur_Y, cur_adj = generate_samples_by_cells(coors_matrix, x_sample, y_sample)
            cur_Xv, cur_Yv, cur_adjv = generate_samples_by_cells(coors_matrix, xv_sample, yv_sample)

            train_dataset = DM_Dataset(cur_X, cur_Y, cur_adj)
            training_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            validation_dataset = DM_Dataset(cur_Xv, cur_Yv, cur_adjv)
            validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

            model.train()

            for local_batch, local_labels, local_adj in training_generator:
                local_batch, local_labels, local_adj = local_batch.to(device), local_labels.to(device), local_adj.to(
                    device)

                outputs = model(local_batch, local_adj, 1)

                outputs = torch.flatten(outputs)
                true_y = torch.flatten(local_labels)
                train_loss = criterion(outputs, true_y)

                avg_train_loss.append(train_loss.cpu().data * len(local_batch))
                ctr += len(local_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            model.eval()

            with torch.no_grad():
                for local_batch, local_labels, local_adj in validation_generator:
                    local_batch, local_labels, local_adj = local_batch.to(device), local_labels.to(
                        device), local_adj.to(device)

                    Voutputs = model(local_batch, local_adj, 1)
                    Voutputs = torch.flatten(Voutputs)

                    V_true_y = torch.flatten(local_labels)
                    v_loss = criterion(Voutputs, V_true_y)

                    avg_valid_loss.append(v_loss.cpu().data * len(local_batch))
                    ctrv += len(local_batch)

        train_loss_arr.append(sum(avg_train_loss) / ctr)
        valid_loss_arr.append(sum(avg_valid_loss) / ctrv)

        if best_result <= float(valid_loss_arr[-1].item()):
            best_ctr += 1
        else:
            best_ctr = 0
            # save_model(model, out_path, k)
            best_model = model
            # print("epochs: " + str(echo))
            # print(float(valid_loss_arr[-1].item()))

        # print("echo: " + str(echo))
        # print("train_loss: "+str(train_loss_arr[-1].item())  + "||" + "v_loss: " + str(valid_loss_arr[-1].item()))

        best_result = min(best_result, valid_loss_arr[-1].item())

        if best_ctr > 5:
            # print("early stop")
            # plot_loss(train_loss_arr, valid_loss_arr)
            # print("best mse: " + str(best_result))
            break

        # if best_result < best_arr[num]:
        #    break
    return best_result, best_model

def convert_to_pred(Xv, y_out_total, x_out_total, xy_out_total, pred_map, ctr_map):

    x_len = Xv.shape[2]
    y_len = Xv.shape[3]

    final_dict = {}

    for j in range(
            len(y_out_total)):
        temp_y = y_out_total[j]
        temp_x = x_out_total[j]
        temp_xy = xy_out_total[j]

        for fk in range(len(temp_y)):
            calendar = temp_x[fk, 0, 3, 3, :5]

            # calendar = calendar*(minus_vector) + min_vector
            # calendar = np.rint(calendar)
            year = float(calendar[0])
            month = float(calendar[1])
            day = float(calendar[2])

            key = str(year) + "-" + str(month) + "-" + str(day) + "-" + str(int(temp_xy[fk][0][0])) + "-" + str(
                int(temp_xy[fk][0][1]))

            final_dict[key] = temp_y[fk]

    for t in range(len(Xv)):
        calendar = Xv[t][0][0][0][:5]
        # calendar = calendar*(minus_vector) + min_vector

        year = float(calendar[0])
        month = float(calendar[1])
        day = float(calendar[2])

        for x in range(x_len):
            for y in range(y_len):

                key = str(year) + "-" + str(month) + "-" + str(day) + "-" + str(x) + "-" + str(y)
                if final_dict.get(key) is not None:
                    pred_map[t][0][x][y] += final_dict[key]
                    ctr_map[t][0][x][y] += 1

    return pred_map, ctr_map


def test_model(Xv, Yv, device, coors_matrix, path, pred_map, ctr_map, model_num):
    iterations = assign_scale(len(coors_matrix))
    batch_size = 1024
    criterion = nn.MSELoss()
    model = NeuralNet(5, 29, 13, S * 2 + 1, 7).to(device)

    model = load_model(model, path, model_num, device)  # !!!!!!!!!!!!!!!!!!!!!!!!!!

    num_Xv_bag = int(len(Xv) / iterations)
    xv_arrange = np.arange(len(Xv))

    valid_loss_arr = []
    ctr = 0
    avg_valid_loss = []

    y_out_total = []
    x_out_total = []
    xy_out_total = []

    for i in range(iterations):

        if i == iterations - 1:
            num_Xv_bag = np.inf

        xv_arrange, xv_sample, yv_sample = data_split_no_random(xv_arrange, num_Xv_bag, Xv, Yv)
        cur_Xv, cur_Yv, cur_adjv, total_xy = generate_samples_by_cells_test(coors_matrix, xv_sample, yv_sample)

        validation_dataset = DM_Dataset_test(cur_Xv, cur_Yv, cur_adjv, total_xy)
        validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        model.eval()

        with torch.no_grad():
            for local_batch, local_labels, local_adj, total_xy in validation_generator:
                local_batch, local_labels, local_adj = local_batch.to(device), local_labels.to(device), local_adj.to(
                    device)

                Voutputs = model(local_batch, local_adj, 1)
                Voutputs = torch.flatten(Voutputs)

                y_out_total.append(Voutputs.cpu().detach().numpy())
                x_out_total.append(local_batch[:, :, :, :, :5].cpu().detach().numpy())
                xy_out_total.append(total_xy.cpu().detach().numpy())

                V_true_y = torch.flatten(local_labels)
                v_loss = criterion(Voutputs, V_true_y)

                avg_valid_loss.append(v_loss.cpu().data * len(local_batch))
                ctr += len(local_batch)

                valid_loss_arr.append(sum(avg_valid_loss) / ctr)


    return convert_to_pred(Xv, y_out_total, x_out_total, xy_out_total, pred_map, ctr_map)

def save_model(model, path, rd, device):
    PATH = "model" + str(rd) + "_" + str(path) + ".pt"
    torch.save(model.state_dict(), PATH)


def load_model(model, path, rd, device):
    PATH = "model" + str(rd) + "_" + str(path) + ".pt"
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
    model.to(device)
    return model