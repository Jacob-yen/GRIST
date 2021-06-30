'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
sys.path.append("/data")
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

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

"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="tensorflow_examples_grist")
obj_function = tf.reduce_min(tf.abs(pred))
obj_grads = tf.gradients(obj_function, x)[0]
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
max_val, min_val = np.max(batch_xs), np.min(batch_xs)
gradient_search.build(batch_size=batch_size, min_val=min_val, max_val=max_val)
"""insert code"""

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    while True:
        """inserted code"""
        monitor_vars = {'loss': cost, 'obj_function': obj_function, 'obj_grad': obj_grads}
        feed_dict = {x: batch_xs, y: batch_ys}
        batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                  feed_dict=feed_dict, input_data=batch_xs,
                                                                  )
        """inserted code"""

        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})

        """inserted code"""
        new_batch_xs, new_batch_ys = mnist.train.next_batch(batch_size)
        new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
        old_data_dict = {'x': batch_xs, 'y': batch_ys}
        batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                             old_data_dict=old_data_dict, scores_rank=scores_rank)
        gradient_search.check_time()
        """inserted code"""
