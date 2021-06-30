import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2 * x)


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5 + x)
y2 = sigmoid(1 + x)
y3 = sigmoid(1.5 + x)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
cost = losses.mean()

# %%
F.binary_cross_entropy(hypothesis, y_train)

# %%

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# %%
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

'''inserted code'''
import sys
sys.path.append("/data")
from scripts.utils.torch_utils import GradientSearcher
gradient_search = GradientSearcher(name="start_beginner_kaggle.01_grist", switch_data=False)
gradient_search.build(batch_size=1)
'''inserted code'''

# nb_epochs = 1000
# for epoch in range(nb_epochs + 1):
while True:
    # Cost 계산
    x_train = torch.autograd.Variable(x_train, requires_grad=True)
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    obj_var = hypothesis
    obj_function = torch.min(torch.abs(obj_var))
    obj_grads = torch.autograd.grad(obj_function, x_train, retain_graph=True, allow_unused=True)[0]

    cost = -(y_train * torch.log(hypothesis) +
             (1 - y_train) * torch.log(1 - hypothesis)).mean()


    """inserted code"""
    monitor_vars = {'loss': cost, 'obj_function': obj_function, 'obj_grad': obj_grads}
    x_train, _ = gradient_search.update_batch_data(monitor_var=monitor_vars, input_data=x_train, clip=False)
    """inserted code"""

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    # if epoch % 100 == 0:
    #     print('Epoch {:4d}/{} Cost: {:.6f}'.format(
    #         epoch, nb_epochs, cost.item()
    #     ))

# %%
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

# %%
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)

# %%
print(W)
print(b)
