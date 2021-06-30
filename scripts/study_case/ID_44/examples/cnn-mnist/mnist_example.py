# 103300 / 30min (not found)
__author__ = 'hadyelsahar'

import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append('../../')
sys.path.append("/data")
from CNN import *

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_train = np.reshape(mnist.train.images, [-1, 28, 28, 1])

y_train = mnist.train.labels
y_train = [np.where(i == 1)[0][0] for i in y_train]

classes = np.unique(y_train)
cnn = CNN(input_shape=[28, 28, 1], classes=classes, conv_shape=[5, 5])

cnn.fit(x_train, y_train)

x_test = np.reshape(mnist.test.images, [-1, 28, 28, 1])
y_pred = cnn.predict(x_test)

y_true = mnist.test.labels
y_true = [list(i).index(1) for i in y_true]
