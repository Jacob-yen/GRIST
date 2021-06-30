import torch
from torch.autograd import Variable
import numpy as np

torch.manual_seed(777)  # for reproducibility

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = Variable(torch.from_numpy(x_data))
Y = Variable(torch.from_numpy(y_data))

# Hypothesis using sigmoid
linear = torch.nn.Linear(2, 1, bias=True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear, sigmoid)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

'''inserted code'''
import sys
sys.path.append("/data")
from scripts.utils.torch_utils import GradientSearcher
gradient_search = GradientSearcher(name="MachineLearning.lab09_grist", switch_data=False)
gradient_search.build(batch_size=1)
'''inserted code'''

# for step in range(10001):
while True:
    X = torch.autograd.Variable(X, requires_grad=True)
    optimizer.zero_grad()
    hypothesis = model(X)

    obj_var = hypothesis
    obj_function = torch.min(torch.abs(obj_var))
    obj_grads = torch.autograd.grad(obj_function, X, retain_graph=True, allow_unused=True)[0]

    # cost/loss function
    cost = -(Y * torch.log(hypothesis) + (1 - Y)
             * torch.log(1 - hypothesis)).mean()

    """inserted code"""
    monitor_vars = {'loss': cost, 'obj_function': obj_function, 'obj_grad': obj_grads}
    X, _ = gradient_search.update_batch_data(monitor_var=monitor_vars, input_data=X, clip=False)
    """inserted code"""

    cost.backward()
    optimizer.step()

    # if step % 100 == 0:
    #     print(step, cost.data.numpy())

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = (model(X).data > 0.5).float()
accuracy = (predicted == Y.data).float().mean()
print("\nHypothesis: ", hypothesis.data.numpy(), "\nCorrect: ", predicted.numpy(), "\nAccuracy: ", accuracy)
