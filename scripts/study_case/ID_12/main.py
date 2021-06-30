import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
mnist = fetch_openml('mnist_784', cache=False)
print(mnist.data.shape)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')
X /= 255.0
print(X.min(), X.max())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
assert(X_train.shape[0] + X_test.shape[0] == mnist.data.shape[0])
print(X_train.shape, y_train.shape)
import sys
sys.path.append("/data")
from scripts.study_case.ID_12.skorch.classifier import NeuralNetClassifier

torch.manual_seed(0)

from torch import nn
import torch.nn.functional as F

mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim/8)
output_dim = len(np.unique(mnist.target))
print(mnist_dim, hidden_dim, output_dim)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class ClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=mnist_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=100,
    lr=0.1,
    device=device,
)
net.fit(X_train, y_train)