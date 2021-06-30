#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append("/data")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

obj_var = y
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="Imprtant_Projects_grist")
obj_function = tf.reduce_min(tf.abs(obj_var))
obj_grads = tf.gradients(obj_function, x)[0]
batch_xs, batch_ys = mnist.train.next_batch(100)
max_val, min_val = np.max(batch_xs), np.min(batch_xs)
gradient_search.build(batch_size=100, min_val=min_val, max_val=max_val)
"""insert code"""

while True:

    """inserted code"""
    monitor_vars = {'loss': cross_entropy, 'obj_function': obj_function, 'obj_grad': obj_grads}
    feed_dict = {x: batch_xs, y_: batch_ys}
    batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                              feed_dict=feed_dict, input_data=batch_xs, )
    """inserted code"""

    _, loss = sess.run([train_step, cross_entropy], feed_dict=feed_dict)

    """inserted code"""
    new_batch_xs, new_batch_ys = mnist.train.next_batch(100)
    new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
    old_data_dict = {'x': batch_xs, 'y': batch_ys}
    batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                         old_data_dict=old_data_dict,
                                                         scores_rank=scores_rank)
    gradient_search.check_time()
    """inserted code"""

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print("cross_entropy_train:", sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}), end=",")
print("cross_entropy_test:", sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
