# -*- encoding=utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append("/data")
mnist = input_data.read_data_sets(r"MNIST_data", one_hot=True)

learnin_rate = 0.1
trainin_epochs = 25
batch_size = 100
display_step = 1

X = tf.placeholder(tf.float32, [None, 784])  # 28*28
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(X, W) + b)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(X, [-1, 28, 28, 1])  # 28 * 28

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size: 28 * 28 * 32
h_pool1 = max_pool_2x2(h_conv1)  # output size: 14 * 14 * 32

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size: 14 * 14 * 64
h_pool2 = max_pool_2x2(h_conv2)  # output size: 7 * 7 * 64

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction), reduction_indices=1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'practice.pbtxt')

sess.run(tf.initialize_all_variables())

'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="practice.logistic_regression")
'''inserted code'''

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    while True:
        batch = mnist.train.next_batch(50)
        batch_xs, batch_ys = batch[0], batch[1]

        _, loss = sess.run([train_step, cross_entropy], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5})

        '''inserted code'''
        scheduler.loss_checker(loss)
        scheduler.check_time()
        '''inserted code'''
