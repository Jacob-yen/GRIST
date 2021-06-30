import torch

x_T = torch.FloatTensor(
    [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]])
y_T = torch.FloatTensor([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

# W, b 초기화
# Optimizer 생성

W = torch.zeros(4, 3, requires_grad=True)
b = torch.zeros(1, 3, requires_grad=True)

optimizer = torch.optim.Adam([W, b], lr=0.1)

# 반복횟수, 가설 및 비용 설정
# H(x) = S(x^T*w + b)

'''inserted code'''
import sys
sys.path.append("/data")
from scripts.utils.torch_utils import TorchScheduler
scheduler = TorchScheduler(name="DataScience.h03")
'''inserted code'''

# for epoch in range(3001):
while True:
    hypothesis = torch.softmax(torch.mm(x_T, W) + b, dim=1)
    cost = -torch.mean(torch.sum(y_T * torch.log(hypothesis), dim=1))

    # Optimizer를 이용한 경사 계산 및 W, b 업데이트

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    '''inserted code'''
    scheduler.loss_checker(cost)
    scheduler.check_time()
    '''inserted code'''

    # Test

    # if epoch % 300 == 0:
    #     print("Epoch : {}, cost : {:.6f}".format(epoch, cost.item()))

print("H : {}, Cost : {}".format(hypothesis, cost))

# x = [1, 11, 10, 9], [1, 3, 4, 3], [1, 1, 0, 1] ~> y =??

W.requires_grad_(False)
b.requires_grad_(False)

x_T = torch.FloatTensor([[1, 11, 10, 9], [1, 3, 4, 3], [1, 1, 0, 1]])
test_all = torch.softmax(torch.mm(x_T, W) + b, dim=1)
print(test_all)
print(torch.argmax(test_all, dim=1))

# 조금 더 깔끔하게 softMax

import torch.nn.functional as F
import torch.nn as nn

x_T = torch.FloatTensor(
    [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]])
y_T = torch.LongTensor([2, 2, 2, 1, 1, 1, 0,
                        0])  # == [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# W, b 초기화
# Optimizer 생성

W = torch.zeros(4, 3, requires_grad=True)
b = torch.zeros(1, 3, requires_grad=True)

model = nn.Linear(4, 3)
optimizer = torch.optim.Adam(model.parameters(), lr=1)  # optimizer = torch.optim.Adam([W, b], lr=0.1)

# 반복횟수, 가설 및 비용 설정
# H(x) = S(x^T*w + b)

for epoch in range(3001):
    # z = torch.mm(x_T, W) + b
    z = model(x_T)
    cost = F.cross_entropy(z,
                           y_T)  # == hypothesis = toch.softmax(torch.mm(x_T, W) + b, dim = 1) 와 cost = -torch.mean(torch.sum(y_T * torch.log(hypothesis), dim = 1))

    # Optimizer를 이용한 경사 계산 및 W, b 업데이트

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # Test

    if epoch % 300 == 0:
        print("Epoch : {}, cost : {:.6f}".format(epoch, cost.item()))

print("H : {}, Cost : {}".format(hypothesis, cost.item()))
