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
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()


rbm = RBM(n_vis=784, n_hid=500, k=1)
train_op = optim.SGD(rbm.parameters(), 0.1)

'''inserted code'''
from scripts.utils.torch_utils import TorchScheduler
scheduler = TorchScheduler(name="git1_rbm")
'''inserted code'''

while True:
    loss_ = []
    for _, (data, target) in enumerate(train_loader):
        sample_data = Variable(data.view(-1, 784)).bernoulli()
        v, v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)

        '''inserted code'''
        scheduler.loss_checker(loss)
        '''inserted code'''

        loss_.append(loss.item())
        train_op.zero_grad()
        loss.backward()
        train_op.step()

        '''inserted code'''
        scheduler.check_time()
        '''inserted code'''
