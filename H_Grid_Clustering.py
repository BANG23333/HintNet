import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, date
import numpy as np
import pandas as pd
import math
import os
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib
import sys

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

class Grid_DBscan():

    def __init__(self, eps=1, min_points=5, min_density=0, acc_matrix=None, mask=None, plot=False):
        self.eps = eps
        self.min_points = min_points
        self.cluster_label = 1
        self.noise = 0
        self.unclassified = -1
        self.min_density = min_density
        self.plot = plot
        self.min_neibours = 3
        self.acc_matrix = acc_matrix
        self.mask = mask

    def generate_small_cell(self, x1, y1, x2, y2, S, x_limit, y_limit):
        if x1 > S * 2 or y1 > S * 2:
            print("S error")
        if x2 >= x_limit or y2 >= y_limit:
            print("Limit error")
        origin = [x2 - S, y2 - S]
        x1 = x1 + origin[0]
        y1 = y1 + origin[1]
        if 0 > x1 or x1 >= x_limit:
            x1 = -1
        if 0 > y1 or y1 >= y_limit:
            y1 = -1
        return x1, y1


    def neighbour_dict_generator(self, len_x, len_y):

        neighbour_dict = {}

        for x in range(len_x):
            for y in range(len_y):
                neighbour_dict[str(x)+"|"+str(y)] = []

        for x in range(len_x):
            for y in range(len_y):
                key = str(x)+"|"+str(y)

                for cell_x in range(2 * self.eps + 1):
                    for cell_y in range(2 * self.eps + 1):

                        temp_x, temp_y = self.generate_small_cell(cell_x, cell_y, x, y, self.eps, len_x, len_y)
                        if temp_x < 0 or temp_y < 0:
                            continue

                        neighbour_dict[key].append(str(temp_x)+"|"+str(temp_y))

        return neighbour_dict

    def neighbour_counter(self, matrix, neighbour_array, origin):
        ctr = 0
        for coor in neighbour_array:
            x, y = int(coor.split("|")[0]), int(coor.split("|")[1])
            if int(matrix[x][y]) > self.min_points:
                ctr += 1
        x, y = int(origin.split("|")[0]), int(origin.split("|")[1])
        if int(matrix[x][y]) > self.min_points:
            ctr += 1
        return ctr

    def expand_cluster(self, x, y, classifications, neighbour_dict, matrix):
        key = str(x) + "|" + str(y)
        seeds = neighbour_dict[key]
        if self.neighbour_counter(matrix, seeds, key) < self.min_neibours:
            classifications[x][y] = self.noise
            return False
        else:
            classifications[x][y] = self.cluster_label
            for seed in seeds:
                x, y = int(seed.split("|")[0]), int(seed.split("|")[1])
                if matrix[x][y] >= self.min_density:
                    classifications[x][y] = self.cluster_label
                else:
                    classifications[x][y] = self.noise

            while len(seeds) > 0:
                current = seeds[0]
                current_neighbour = neighbour_dict[current]
                if self.neighbour_counter(matrix, current_neighbour, current) >= self.min_neibours:
                    for nei in current_neighbour:
                        x, y = int(nei.split("|")[0]), int(nei.split("|")[1])
                        if classifications[x][y] == self.unclassified or classifications[x][y] == self.noise:
                            if classifications[x][y] == self.unclassified:
                                seeds.append(nei)
                            if matrix[x][y] >= self.min_density:
                                classifications[x][y] = self.cluster_label
                            else:
                                classifications[x][y] = self.noise
                seeds = seeds[1:]
            return True

    def helper_plot(self, classifications):
        print(classifications)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax = sns.heatmap(classifications.transpose(), robust=True, annot=False, cbar=False, linewidths=.25, linecolor="black")
        ax.invert_yaxis()
        plt.title("eps: " + str(self.eps) + " | " + "min_points: " + str(self.min_points))
        plt.show()

    def fit_predict(self, matrix):
        # matrix = np.where(matrix <= 10, 0, matrix)

        len_x = matrix.shape[0]
        len_y = matrix.shape[1]
        neighbour_dict = self.neighbour_dict_generator(len_x, len_y)

        classifications = np.full((len_x, len_y), self.unclassified)

        for x in range(len_x):
            for y in range(len_y):
                if classifications[x][y] == -1:
                    if self.expand_cluster(x, y, classifications, neighbour_dict, matrix):
                        self.cluster_label += 1

        classifications = np.where(self.mask == 1, classifications, 0)

        if self.plot:
            self.helper_plot(classifications)

        return classifications

    def count_acc(self, coor_list, acc_matrix):
        sum = 0
        for coor in coor_list:
            sum += acc_matrix[coor[0]][coor[1]]
        return sum


    def write_in_map(self, target, coor_list, matrix):
        out = matrix.copy()
        for coor in coor_list:
            out[coor[0]][coor[1]] = target
        return out


    def coor_filter(self, gla_fix_map, coor_list):
        out = []
        for coor in coor_list:
            if gla_fix_map[coor[0]][coor[1]] == 0:
                out.append(coor)
        return out

    def count_num(self, matrix, target):
        ctr = 0
        for x in matrix:
            for y in x:
                if target == y:
                    ctr += 1
        return ctr

    def map_scan(self, iter, threshold, increment):

        pred_map = np.zeros((iter, self.acc_matrix.shape[0], self.acc_matrix.shape[1]))

        for i in range(iter):
            self.min_neibours = i
            labels = self.fit_predict(self.acc_matrix)
            pred_map[i] = labels

        gla_fix_map = np.full((pred_map.shape[1], pred_map.shape[2]), 0)
        result_map = np.full((pred_map.shape[1], pred_map.shape[2]), -increment)

        for i in range(0, iter, increment):
            if i == iter - increment:
                threshold = np.inf

            for target in np.unique(pred_map[i]):

                coor_list = []

                for x in range(pred_map.shape[1]):
                    for y in range(pred_map.shape[2]):

                        if pred_map[i][x][y] == target:
                            coor_list.append([x, y])

                if target == 0:
                    coor_list = self.coor_filter(gla_fix_map, coor_list)

                    gla_fix_map = self.write_in_map(1, coor_list, gla_fix_map)
                    result_map = self.write_in_map(i-increment, coor_list, result_map)

                    continue

                cur_acc = self.count_acc(coor_list, self.acc_matrix)

                if cur_acc <= threshold:
                    coor_list = self.coor_filter(gla_fix_map, coor_list)
                    gla_fix_map = self.write_in_map(1, coor_list, gla_fix_map)
                    result_map = self.write_in_map(i, coor_list, result_map)

        result_map += increment
        result_map = np.where(self.mask == 1, result_map, -1)
        result_map += 1
        return result_map

