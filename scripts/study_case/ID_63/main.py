import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
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
        p_ = p - Variable(torch.rand(p.size()))
        p_sign = torch.sign(p_)
        return F.relu(p_sign)

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
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
        temp = torch.log(
            torch.exp(wx_b) + 1
        )
        hidden_term = torch.sum(temp, dim=1)

        return (-hidden_term - vbias_term).mean()


batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=batch_size)

'''inserted code'''
from scripts.utils.torch_utils import TorchScheduler
scheduler = TorchScheduler(name="gongdols_main")
'''inserted code'''

rbm = RBM(k=1)
train_op = optim.Adam(rbm.parameters(), 0.005)


while True:
    loss_ = []
    for _, (data, target) in enumerate(train_loader):
        data = Variable(data.view(-1, 784), requires_grad=True)
        sample_data = Variable(data, requires_grad=True)
        # data = Variable(data.view(-1, 784))
        # sample_data = data.bernoulli()

        v, v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)

        '''inserted code'''
        scheduler.loss_checker(loss)
        '''inserted code'''

        loss_.append(loss.data.item())
        train_op.zero_grad()

        loss.backward()
        train_op.step()

        '''inserted code'''
        scheduler.check_time()
        '''inserted code'''
