'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import sys
import sys
sys.path.append("/data")
from datetime import datetime
import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 250
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax
#MUTATION#
# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
obj_var = tf.reduce_min(tf.abs(pred))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler

scheduler = TensorFlowScheduler(name="tensorflow_examples")
'''inserted code'''

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'tensorflow_examples.pbtxt')
    sess.run(init)

    while True:
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, loss, obj_var_val = sess.run([optimizer, cost, obj_var], feed_dict={x: batch_xs,
                                                                                y: batch_ys})

            '''inserted code'''
            scheduler.loss_checker(loss)
            '''inserted code'''

            '''inserted code'''
            scheduler.check_time()
            '''inserted code'''
