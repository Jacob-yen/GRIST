import torch
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.FloatTensor([1, 2, 3])

hypothesis = F.softmax(z, dim=0)
print(hypothesis)
hypothesis.sum()

z = torch.rand(3, 5, requires_grad=True)

hypothesis = F.softmax(z, dim=1)
print(hypothesis)

y = torch.randint(5, (3,)).long()

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y.unsqueeze(1))

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

'''inserted code'''
import sys
sys.path.append("/data")
from scripts.utils.torch_utils import TorchScheduler
scheduler = TorchScheduler(name="pytorch_tu")
'''inserted code'''

while True:
    z = torch.autograd.Variable(z, requires_grad=True)
    hypothesis = F.softmax(z, dim=1)
    y = torch.randint(5, (3,)).long()
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    loss = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    '''inserted code'''
    scheduler.loss_checker(loss)
    scheduler.check_time()
    '''inserted code'''

F.nll_loss(F.log_softmax(z, dim=1), y)
F.cross_entropy(z, y)
