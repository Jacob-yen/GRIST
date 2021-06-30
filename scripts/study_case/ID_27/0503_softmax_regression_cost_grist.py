import myutil as mu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


torch.manual_seed(1)

z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)
mu.log("z", z)
mu.log("hypothesis", hypothesis)
mu.log("hypothesis.sum()", hypothesis.sum())

z = torch.rand(3, 5, requires_grad=True)
mu.log("z", z)
hypothesis = F.softmax(z, dim=1)
mu.log("hypothesis", hypothesis)
mu.log("hypothesis.sum(dim=1)", hypothesis.sum(dim=1))

y = torch.randint(5, (3,))
mu.log("y", y)
hypothesis = F.softmax(z, dim=1)
mu.log("hypothesis", hypothesis)
y_one_hot = torch.zeros_like(hypothesis)
mu.log("y_one_hot", y_one_hot)
mu.log("y.unsqueeze(1)", y.unsqueeze(1))
mu.log("y_one_hot.scatter_(dim=1, y.unsqueeze(dim=1), index=1)", y_one_hot.scatter_(1, y.unsqueeze(1), 1))


cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
mu.log("z", z)
mu.log("y", y)
mu.log("y_one_hot", y_one_hot)
mu.log("hypothesis", hypothesis)
mu.log("torch.log(hypothesis)", torch.log(hypothesis))
mu.log("y_one_hot * -torch.log(hypothesis)", y_one_hot * -torch.log(hypothesis))
mu.log("(y_one_hot * -torch.log(hypothesis)).sum(dim=1)", (y_one_hot * -torch.log(hypothesis)).sum(dim=1))
mu.log("(y_one_hot * -torch.log(hypothesis)).sum()", (y_one_hot * -torch.log(hypothesis)).sum())
mu.log("cost", cost)


z = torch.rand(3, 5, requires_grad=True)
mu.log("z", z)
y = torch.randint(5, (3,))
mu.log("y", y)
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
mu.log("y_one_hot", y_one_hot)
hypothesis = F.softmax(z, dim=1)
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
mu.log("cost low level", cost)

'''inserted code'''
import sys
sys.path.append("/data")
from scripts.utils.torch_utils import GradientSearcher
gradient_search = GradientSearcher(name="PyTorchDeepLearningStart.0503_softmax_regression_cost_grist", switch_data=False)
gradient_search.build(batch_size=1)
'''inserted code'''

while True:
    z = torch.autograd.Variable(z, requires_grad=True)
    hypothesis = F.softmax(z, dim=1)
    y = torch.randint(5, (3,)).long()
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    obj_var = hypothesis
    loss = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
    obj_function = torch.min(torch.abs(obj_var))
    obj_grads = torch.autograd.grad(obj_function, z, retain_graph=True, allow_unused=True)[0]

    """inserted code"""
    monitor_vars = {'loss': loss, 'obj_function': obj_function, 'obj_grad': obj_grads}
    z, _ = gradient_search.update_batch_data(monitor_var=monitor_vars, input_data=z, clip=False)
    """inserted code"""


cost = (y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean()
mu.log("cost log_softmax", cost)

cost = F.nll_loss(F.log_softmax(z, dim=1), y)
mu.log("cost nll_loss log_softmax", cost)

cost = F.cross_entropy(z, y)
mu.log("cost cross_entropy", cost)
