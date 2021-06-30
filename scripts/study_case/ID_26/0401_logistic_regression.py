import myutil as mu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더
from torch.utils.data import Dataset

import numpy as np  # 넘파이 사용


def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
mu.log("hypothesis", hypothesis)
mu.log("y_train", y_train)

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
mu.log("hypothesis", hypothesis)
mu.log("y_train", y_train)

losses = -(y_train * torch.log(hypothesis)) + (1 - y_train) * torch.log(1 - hypothesis)
cost = losses.mean()
mu.log("losses", losses)
mu.log("cost", cost)

loss = F.binary_cross_entropy(hypothesis, y_train)
mu.log("loss.item()", loss.item())

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=1)
nb_epoches = 1000
mu.plt_init()

'''inserted code'''
import sys
sys.path.append("/data")
from scripts.utils.torch_utils import TorchScheduler
scheduler = TorchScheduler(name="PyTorchDeepLearningStart.0401_logistic_regression")
'''inserted code'''

while True:
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()
    accuracy = mu.get_regression_accuracy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    '''inserted code'''
    scheduler.loss_checker(cost)
    scheduler.check_time()
    '''inserted code'''

mu.plt_show()
mu.log("W", W)
mu.log("b", b)

prediction = hypothesis >= torch.FloatTensor([0.5])
mu.log("prediction", prediction)
mu.log("y_data", y_data)
