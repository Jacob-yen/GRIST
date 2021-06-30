import myutil as mu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

sys.path.append("/data")
torch.manual_seed(1)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]

y_train = [2, 2, 2, 1, 1, 1, 0, 0]

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

mu.log("x_train", x_train)
mu.log("y_train", y_train)

y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(dim=1, index=y_train.unsqueeze(dim=1), value=1)
mu.log("y_one_hot", y_one_hot)

W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=0.1)
nb_epoches = 2000
mu.plt_init()

'''inserted code'''
import sys
sys.path.append("/data")
from scripts.utils.torch_utils import GradientSearcher
gradient_search = GradientSearcher(name="PyTorchDeepLearningStart.0504_softmax_regression_grist", switch_data=False)
gradient_search.build(batch_size=1)
'''inserted code'''

while True:
    x_train = torch.autograd.Variable(x_train, requires_grad=True)
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)

    obj_var = hypothesis
    obj_function = torch.min(torch.abs(obj_var))
    obj_grads = torch.autograd.grad(obj_function, x_train, retain_graph=True, allow_unused=True)[0]

    cost = (y_one_hot * -torch.log(hypothesis)).sum().mean()
    optimizer.zero_grad()

    """inserted code"""
    monitor_vars = {'loss': cost, 'obj_function': obj_function, 'obj_grad': obj_grads}
    x_train, _ = gradient_search.update_batch_data(monitor_var=monitor_vars, input_data=x_train, clip=False)
    """inserted code"""

    cost.backward()
    optimizer.step()

mu.plt_show()
mu.log("W", W)
mu.log("b", b)
