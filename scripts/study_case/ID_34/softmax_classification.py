import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

z = torch.FloatTensor([1, 2, 3])

# https://bskyvision.com/427
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
print(hypothesis.sum())

# Cross entropy loss (low-level)
# For multi-class classification, we use the cross entropy loss
z = torch.rand(3, 5, requires_grad=True)  # a uniform distribution on the interval [0, 1)
hypothesis = F.softmax(z, dim=1)
print(z)
print(hypothesis)
print(hypothesis[:1, :].sum())

# Returns a tensor filled with random integers generated uniformly between low(inclusive) and high(exclusive)
y = torch.randint(5, (3,)).long()  # params; low=0, high, size
print(y)

y_one_hot = torch.zeros_like(hypothesis)  # Returns a tensor filled with the scalar value 0, with the same size as input
print(y_one_hot)
print(y.unsqueeze(1))
# https://discuss.pytorch.org/t/what-does-the-scatter-function-do-in-layman-terms/28037/2
# Writes all values from the tensor src into self at the indices specified in the index tensor.
# params; dim, index, src
y_one_hot.scatter_(1, y.unsqueeze(1), 1)  # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_
print(y_one_hot)

# Cross entropy loss
# https://de-novo.org/tag/cross-entropy-loss/
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

# Cross entropy loss with torch.nn.functional
# low level
print(torch.log(F.softmax(z, dim=1)))

# high level
print(F.log_softmax(z, dim=1))

# PyTorch also has F.nll_loss() function that computes the negative loss likelihood
# low level
nll_loss = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(nll_loss)

print(F.nll_loss(F.log_softmax(z, dim=1), y))

# PyTorch also has F.cross_entropy that combines F.log_softmax() and F.nll_loss()
print(F.cross_entropy(z, y))

# Training with low-level cross entropy loss
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]  # 8 x 4
y_train = [2, 2, 2, 1, 1, 1, 0, 0]  # 1 x 8
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# Model initialization
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# Optimizer
optimizer = optim.SGD([W, b], lr=0.1)

'''inserted code'''
import sys
sys.path.append("/data")
from scripts.utils.torch_utils import TorchScheduler
scheduler = TorchScheduler(name="TodayILearned.softmax_classification")
'''inserted code'''

print("Training with low-level cross entropy loss")
# nb_epochs = 1000
# for epoch in range(nb_epochs + 1):
while True:
    # cost 1
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    '''inserted code'''
    scheduler.loss_checker(cost)
    scheduler.check_time()
    '''inserted code'''

    # if epoch % 100 == 0:
    #     print('Epoch {:4d}/{} Cost: {:.6f}'.format(
    #         epoch, nb_epochs, cost.item()))

# Model initialization
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer
optimizer = optim.SGD([W, b], lr=0.1)

print("Training with F.cross_entropy")
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # cost 2
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()))


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


model = SoftmaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Training with nn.Module")
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x)
    prediction = model(x_train)

    # cost
    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
