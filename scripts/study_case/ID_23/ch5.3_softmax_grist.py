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
from scripts.utils.torch_utils import GradientSearcher
gradient_search = GradientSearcher(name="pytorch_tu_grist", switch_data=False)
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

F.nll_loss(F.log_softmax(z, dim=1), y)
F.cross_entropy(z, y)
