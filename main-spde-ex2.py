#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from scipy.special import gamma
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import torch
import argparse

# set random seed
np.random.seed(42)


def g(x):
    '''
    inputs
        :param x: float, x value
    return
        concrete example functions value
    e.x.
    ===================
    x + 1
    '''
    return x + 1


def d(x, i, alpha):
    '''
    \Gamma function
    '''
    if i < np.ceil(alpha):
        return 0.0
    return math.gamma(i + 1) / math.gamma(i + 1 - alpha) * x ** (i - alpha)


def f(x, a_value, n=3):
    '''
    left formula, Df(x) : spde

    inputs
        :param x: float, x value
        :param a_value: train args
        :param n:  a_{j}, j=1,2,...,n
    '''
    global lambda_v
    # artificially set parameters
    lambda_v = 0.88

    error = torch.zeros(1)
    for idx, xx in enumerate(x):
        total_value = 0
        for i in range(n+1):
            total_value += a_value[i] * d(xx, i, 2)

        for i in range(n+1):
            total_value += a_value[i] * d(xx, i, 1.5)

        for i in range(n+1):
            total_value += a_value[i] * xx ** i


        g_result = g(xx)

        er = (total_value - g_result) ** 2 + lambda_v * (a_value[0] - 1.0) ** 2
        error += er

    return error / (len(x) + 1)


class DataSet(object):
    def __init__(self, sample_list):
        self.len = len(sample_list)
        self.data = torch.from_numpy(np.array(sample_list, np.float32))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


# ANN
class MyNet(nn.Module):
    def __init__(self, input_param, hidden_num, output_param):
        super(MyNet, self).__init__()
        self.linear0 = nn.Linear(input_param, hidden_num, bias=True)
        self.linear1 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.linear2 = nn.Linear(hidden_num, output_param, bias=True)

    def forward(self, x):
        x = self.linear0(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.dropout(x, p=0.01)
        x = F.relu(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=int, default=10, help="data sample nums")
    parser.add_argument("max_iter", type=int, default=5000, help="iter max value")
    parser.add_argument("hidden_layer", type=int, default=30, help="hidden nums")
    parser.add_argument("j_value", type=int, help="a_{j}")
    parser.add_argument("lr", type=float, help="learning rate")
    # args = get_arg()
    args = parser.parse_args()

    # axis
    data_nums = args.data
    # learning rate
    lr = args.lr

    input_param = data_nums // 2
    hidden_num = args.hidden_layer
    output_param = args.j_value

    # train nums
    n_iter_max = args.max_iter

    x = np.linspace(0, 1, data_nums)
    y = g(x)
    # array2tensor
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)

    dataset = DataSet(x)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=data_nums // 2,
        shuffle=True,
        num_workers=0)

    model = MyNet(input_param, hidden_num, output_param)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for iter_n in range(n_iter_max):
        for i, data in enumerate(train_loader, 0):
            a_value = model(data)
            loss = f(data, a_value, n=output_param - 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (iter_n) % 30 == 0:
            print("iter-->",iter_n, "loss-->", loss.data)

