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

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'ImportantProjects.pbtxt')
tf.global_variables_initializer().run()

'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="Imprtant_Projects")
'''inserted code'''

while True:
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})

    '''inserted code'''
    scheduler.loss_checker(loss)
    scheduler.check_time()
    '''inserted code'''

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print("cross_entropy_train:", sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}), end=",")
print("cross_entropy_test:", sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
