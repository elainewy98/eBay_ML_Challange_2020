from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
class Sequence(Dataset):
    def __init__(self, df, label, batch_size=128):
        self.batch_size = batch_size
        self.df = df
        self.label = label

    def __getitem__(self, i):
        return self.df[i, :].astype(np.int_), self.label[i].astype(np.int_)

    def __len__(self):
        return len(self.label)

class eBay_Model(nn.Module):
    def __init__(self, convo_layers=None, dense_layers=None, dropoutrate=0.3, batch=256, input_height=1024,input_width=1024):
        super(eBay_Model, self).__init__()
        self.batch = batch
        self.cl1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.cl2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.cl3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5,stride=2)
        self.cl4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cl5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.cl6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,stride=2)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.cl4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.avgpool = nn.AvgPool2d(kernel_size=2)

        # self.cl4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.convo_layers = torch.nn.Sequential(self.cl1, nn.Dropout(0.4), self.cl2,nn.Dropout(0.4),
                                                self.cl3, nn.Dropout(0.4),self.cl4, nn.Dropout(0.4),self.cl5, self.cl6)

        self.ll1 = nn.Linear(in_features=128, out_features=64)
        self.relu1 = nn.ReLU()
        self.ll2 = nn.Linear(in_features=64, out_features=16)
        # self.relu2 = nn.ReLU()
        # self.ll3 = nn.Linear(in_features=32, out_features=16)
        # self.ll4 = nn.Linear(in_features=16, out_features=dense_layers[0])
        self.dense_layers = torch.nn.Sequential(self.ll1, self.relu1, self.ll2)
        # self.relu2, self.ll3)


    def forward(self, x):
        x = self.convo_layers(x)

        x = x.view(-1, 512)
        x = self.dense_layers(x)
        V = torch.clamp(x, -100, 100)
        return V

    # def loss(self, V, y):

    #     inner_mat = V@V.T
    #     mag_vec = 1/(torch.diagonal(inner_mat))**0.5
    #     cos_mat = ((inner_mat * mag_vec).T * mag_vec +1)/2
    #     n = len(y)
    #     i_index, j_index = np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    #     def check_group(x1, x2):
    #         return y[x1]==y[x2]
    #     v_check_group = np.vectorize(check_group)
    #     mask = torch.tensor(v_check_group(i_index, j_index).astype(int)*2-1)
    #     loss = torch.sum(cos_mat*mask)
    #     return -loss
    def pairwise_distances(self, x, y=None):
        '''
        Input: x is a Nxd matrix
              y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)

    def loss(self, V,y):
        # print(V[:5,:5])
        dist = self.pairwise_distances(V)
        n = len(y)
        i_index, j_index = np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
        def check_group(x1, x2):
            return y[x1]==y[x2]
        v_check_group = np.vectorize(check_group)
        mask = torch.tensor(v_check_group(i_index, j_index).astype(int))
        mask = mask*(n**2/torch.sum(mask)-1)-1
        mag_vec = torch.norm(V, dim=1)
        loss = torch.sum(dist*mask)
        # + torch.sum(mag_vec)*n
        # print(torch.sum(mag_vec)*n)
        # print(torch.sum(dist*mask))
        return loss/n/n

