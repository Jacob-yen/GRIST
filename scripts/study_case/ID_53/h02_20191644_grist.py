import torch

x_train = torch.FloatTensor([[1], [2], [3], [4], [5], [2.5], [3.5], [0], [3.1], [2.7], [2.8], [2.9]])
y_train = torch.FloatTensor([[1], [1], [1], [0], [0], [0], [0], [1], [0], [1], [1], [1]])

# W, b 초기화
# Learning Rate 설정

W = torch.zeros(1, 1)  # 1 x 1
b = torch.zeros(1, 1)  # 1 x 1

lr = 1.0

# 반복횟수 설정
# W, b의 requires_grad를 T로

'''inserted code'''
import sys
sys.path.append("/data")
from scripts.utils.torch_utils import GradientSearcher
gradient_search = GradientSearcher(name="DataScience.h02_grist", switch_data=False)
gradient_search.build(batch_size=1)
'''inserted code'''

while True:
# for epoch in range(3001):
    x_train = torch.autograd.Variable(x_train, requires_grad=True)
    W.requires_grad_(True)
    b.requires_grad_(True)
    # Hypothesis, cost 설정

    hypothesis = torch.sigmoid(torch.mm(x_train, W) + b)

    obj_var = hypothesis
    obj_function = torch.min(torch.abs(obj_var))
    obj_grads = torch.autograd.grad(obj_function, x_train, retain_graph=True, allow_unused=True)[0]

    cost = torch.mean(-y_train * torch.log(hypothesis) - (1 - y_train) * torch.log(1 - hypothesis))

    # 경사 계산
    # W, b 업데이트

    """inserted code"""
    monitor_vars = {'loss': cost, 'obj_function': obj_function, 'obj_grad': obj_grads}
    x_train, _ = gradient_search.update_batch_data(monitor_var=monitor_vars, input_data=x_train, clip=False)
    """inserted code"""

    cost.backward()
    with torch.no_grad() as grd:
        W = W - lr * W.grad
        b = b - lr * b.grad
    # if epoch % 100 == 0:
    #     print('epoch: {}, cost : {:.6f}, W : {:.6f}, b : {:.6f}'.format(epoch, cost.item(), W.squeeze(), b.squeeze()))

# Test

x_test = torch.FloatTensor([[4.5], [1.1]])
test_result = torch.sigmoid(torch.mm(x_test, W) + b)
print(torch.round(test_result))

# Optimizer
# Matplotlib

'''
optimizer = torch.optim.SGD([W, b], lr = 1.0)
optimizer.zero_grad()
cost.backward()
optimizer.step()
'''
import matplotlib.pyplot as plt

W.requires_grad_(False)
b.requires_grad_(False)

plt.scatter(x_train, y_train, c='black', label='Trainingdata')  # point?

X = torch.linspace(0, 5, 100).unsqueeze(1)  # unsqueeze -> T?
# print(X)
Y = torch.sigmoid(torch.mm(X, W) + b)
# print(Y)
# print(W)

plt.plot(X, Y, c='#ff0000', label='Fitting line')  # line

plt.ylabel("Probability of 1(Y)")
plt.xlabel("Input (X)")
plt.legend()
plt.show()
