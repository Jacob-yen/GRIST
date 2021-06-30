import torch
import torch.optim as optim
import torch.nn.functional as F

# For reproducibility
torch.manual_seed(1)

# Training data
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]  # 6 x 2
y_data = [[0], [0], [0], [1], [1], [1]]  # 6 x 1

# Given the number of hours each student spent watching the lecture and working in the code lab,
# predict whether the student passed or failed a course.

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)

# Computing the hypothesis
print("e^1 equals: {}".format(torch.exp(torch.FloatTensor([1]))))

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))

print(hypothesis)
print(hypothesis.shape)

print("1/(1+e^(-1)) equals: {}".format(torch.sigmoid(torch.FloatTensor([1]))))

hypothesis = torch.sigmoid(x_train.matmul(W) + b)

print(hypothesis)
print(hypothesis.shape)

# Computing the cost function (low-level)
# BCS(Binary Cross-Entropy)
# We want to measure the difference between hypothesis and y_train

# For one element, the loss can be computed as follows
single_loss = -(y_train[0] * torch.log(hypothesis[0])) + (1 - y_train[0]) * torch.log(1 - hypothesis[0])
print(single_loss)

losses = -(y_train * torch.log(hypothesis)) + (1 - y_train) * torch.log(1 - hypothesis)
print(losses)

cost = losses.mean()
print(cost)

# Computing the cost function with F.binary_cross_entropy
print(F.binary_cross_entropy(hypothesis, y_train))

# Training with low-level binary cross entropy loss
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1)

'''inserted code'''
import sys
sys.path.append("/data")
from scripts.utils.torch_utils import GradientSearcher
gradient_search = GradientSearcher(name="TodayILearned.logistic_classification_grist", switch_data=False)
gradient_search.build(batch_size=1)
'''inserted code'''

# nb_epochs = 1000
# for epoch in range(nb_epochs + 1):
while True:
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

    # fitting model with cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # if epoch % 100 == 0:
    #     print('Epoch {:4d}/{} Cost: {:.6f}'.format(
    #         epoch, nb_epochs, cost.item()))

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()))
