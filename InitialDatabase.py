from torch.utils.data import Dataset
from torch.autograd import Variable
import torch

class DM_Dataset(Dataset):
    def __init__(self, X_input, Y_input, adj_input):
        self.X_input = Variable(torch.Tensor(X_input).float())
        self.Y_input = Variable(torch.Tensor(Y_input).float())
        self.adj_input = Variable(torch.Tensor(adj_input).float())

    def __len__(self):
        return len(self.X_input)

    def __getitem__(self, idx):
        return self.X_input[idx], self.Y_input[idx], self.adj_input[idx]


class DM_Dataset_test(Dataset):
    def __init__(self, X_input, Y_input, adj_input, total_xy):
        self.X_input = Variable(torch.Tensor(X_input).float())
        self.Y_input = Variable(torch.Tensor(Y_input).float())
        self.adj_input = Variable(torch.Tensor(adj_input).float())
        self.total_xy = Variable(torch.Tensor(total_xy).float())

    def __len__(self):
        return len(self.X_input)

    def __getitem__(self, idx):
        return self.X_input[idx], self.Y_input[idx], self.adj_input[idx], self.total_xy[idx]