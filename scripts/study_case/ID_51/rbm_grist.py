import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import sys

sys.path.append("/data")


class RBM(nn.Module):
    def __init__(self,
                 n_vis=784,
                 n_hin=500,
                 k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
        self.obj_function = None

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def v_to_h(self, v):
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        pre_h1, h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(self.k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)

        return v, v_

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        self.obj_function = -1 * torch.max(wx_b)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()


batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=batch_size)

"inserted code"
from scripts.utils.torch_utils import GradientSearcher
gradient_search = GradientSearcher(name="RBM_grist")
dataloader_iter = iter(train_loader)
data, target = next(dataloader_iter)
data = Variable(data, requires_grad=True)
max_val, min_val = data.max().item(), data.min().item()
gradient_search.build(batch_size=batch_size, min_val=min_val, max_val=max_val)

"inserted code"

rbm = RBM(k=1)
train_op = optim.SGD(rbm.parameters(), 0.1)
loss_ = []
while True:
    data = Variable(data.view(-1, 784), requires_grad=True)
    # sample_data = data.bernoulli()
    sample_data = Variable(data, requires_grad=True)

    v, v1 = rbm(sample_data)
    output1 = rbm.free_energy(v)
    output2 = rbm.free_energy(v1)

    """insert code"""

    obj_grads = torch.autograd.grad(rbm.obj_function, sample_data, retain_graph=True, allow_unused=True)[0]
    """insert code"""

    loss = output1 - output2
    loss_.append(loss.data.item())
    train_op.zero_grad()

    """inserted code"""
    monitor_vars = {'loss': loss, 'obj_function': rbm.obj_function, 'obj_grad': obj_grads}
    data, scores_rank = gradient_search.update_batch_data(monitor_var=monitor_vars, input_data=data)
    """inserted code"""

    loss.backward()
    train_op.step()

    """inserted code"""
    try:
        new_batch_xs, new_batch_ys = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(train_loader)
        new_batch_xs, new_batch_ys = next(dataloader_iter)
    new_batch_xs = Variable(new_batch_xs.view(-1, 784), requires_grad=True)
    new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
    old_data_dict = {'x': data, 'y': target}
    data, target = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                   old_data_dict=old_data_dict,
                                                   scores_rank=scores_rank)

    data = Variable(data, requires_grad=True)
    """inserted code"""

    '''inserted code'''
    gradient_search.check_time()
    '''inserted code'''

    # print(np.mean(loss_))
