import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid , save_image
import sys

sys.path.append("/data")
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('My_RBM', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])), batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('My_RBM', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])), batch_size=batch_size)

class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, k):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.k = k
        self.obj_function = None

    def vis_hid(self, v):
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = F.relu(torch.sign(p_h - Variable(torch.rand(p_h.size()))))
        return p_h, sample_h

    def hid_vis(self, h):
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = F.relu(torch.sign(p_v - Variable(torch.rand(p_v.size()))))
        return p_v, sample_v

    def forward(self, v):
        h0, h_ = self.vis_hid(v)
        for _ in range(self.k):
            v0_, v_ = self.hid_vis(h_)
            h0_, h_ = self.vis_hid(v_)
        return v, v_

    def free_energy(self, v):
        wx_b = F.linear(v, self.W, self.h_bias)
        vbias_term = v.mv(self.v_bias)
        self.obj_function = -1 * torch.max(wx_b)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()


"inserted code"
from scripts.utils.torch_utils import GradientSearcher
gradient_search = GradientSearcher(name="git1_rbm_grist")
dataloader_iter = iter(train_loader)
data, target = next(dataloader_iter)
data = Variable(data, requires_grad=True)
max_val, min_val = data.max().item(), data.min().item()
gradient_search.build(batch_size=batch_size, min_val=min_val, max_val=max_val)
"inserted code"

rbm = RBM(n_vis=784, n_hid=500, k=1)
train_op = optim.SGD(rbm.parameters(), 0.1)

loss_ = []
while True:
    # sample_data = Variable(data.view(-1, 784)).bernoulli()
    data = Variable(data.view(-1, 784), requires_grad=True)
    sample_data = Variable(data, requires_grad=True)
    v, v1 = rbm(sample_data)
    loss = rbm.free_energy(v) - rbm.free_energy(v1)
    loss_.append(loss.item())
    train_op.zero_grad()

    """insert code"""
    obj_grads = torch.autograd.grad(rbm.obj_function, sample_data, retain_graph=True, allow_unused=True)[0]
    monitor_vars = {'loss': loss, 'obj_function': rbm.obj_function, 'obj_grad': obj_grads}
    data, scores_rank = gradient_search.update_batch_data(monitor_var=monitor_vars, input_data=data)
    """insert code"""

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
    gradient_search.check_time()
    '''inserted code'''
