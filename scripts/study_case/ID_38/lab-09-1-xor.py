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
from scripts.utils.torch_utils import TorchScheduler
scheduler = TorchScheduler(name="MachineLearning.lab09")
'''inserted code'''

# for step in range(10001):
while True:
    optimizer.zero_grad()
    hypothesis = model(X)
    # cost/loss function
    cost = -(Y * torch.log(hypothesis) + (1 - Y)
             * torch.log(1 - hypothesis)).mean()
    cost.backward()
    optimizer.step()

    '''inserted code'''
    scheduler.loss_checker(cost)
    scheduler.check_time()
    '''inserted code'''

    # if step % 100 == 0:
    #     print(step, cost.data.numpy())

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = (model(X).data > 0.5).float()
accuracy = (predicted == Y.data).float().mean()
print("\nHypothesis: ", hypothesis.data.numpy(), "\nCorrect: ", predicted.numpy(), "\nAccuracy: ", accuracy)
