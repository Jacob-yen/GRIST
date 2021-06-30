#-*-coding=utf-8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import sys
sys.path.append("/data")

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# log_dir='/home/dmrf/tensorflow_gesture_data/Log'

# Variables
batch_size = 100
total_steps = 5000
dropout_keep_prob = 0.5
steps_per_test = 100


def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding)


def max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)

# Initial
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784],name='x-input')
    y_label = tf.placeholder(tf.float32, shape=[None, 10],name='y-input')

with tf.name_scope('input_reshape'):
    x_reshape = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input',x_reshape,10)

# Layer1
w_conv1 = weight([5, 5, 1, 32])
b_conv1 = bias([32])
h_conv1 = tf.nn.relu(conv2d(x_reshape, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

# Layer2
w_conv2 = weight([5, 5, 32, 64])
b_conv2 = bias([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

# Layer3
w_fc1 = weight([7 * 7 * 64, 1024])
b_fc1 = bias([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)


# Softmax
w_fc2 = weight([1024, 10])
b_fc1 = bias([10])
y = tf.nn.softmax(tf.matmul(h_fc1,w_fc2)+b_fc1)

#MUTATION#
# Loss
cross_entropy = -tf.reduce_sum(y_label * tf.log(y))

train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Prediction
correct_prediction = tf.equal(tf.argmax(y_label, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="Tensorflow_gesture_grist")
obj_function = tf.reduce_min(tf.abs(y))
obj_grads = tf.gradients(obj_function, x)[0]
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
max_val, min_val = np.max(batch_xs), np.min(batch_xs)
gradient_search.build(batch_size=batch_size,min_val=min_val,max_val=max_val)
"""insert code"""

step = 0
# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
        """inserted code"""
        monitor_vars = {'loss': cross_entropy, 'obj_function': obj_function, 'obj_grad': obj_grads}
        feed_dict = {x: batch_xs, y_label: batch_ys}
        batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                  feed_dict=feed_dict, input_data=batch_xs,
                                                                  )
        """inserted code"""

        sess.run(train, feed_dict={x: batch_xs, y_label: batch_ys})

        obj_grads_val = sess.run([obj_grads], feed_dict=feed_dict)

        """inserted code"""
        new_batch_xs, new_batch_ys = mnist.train.next_batch(batch_size)
        new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
        old_data_dict = {'x': batch_xs, 'y': batch_ys}
        batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict, old_data_dict=old_data_dict,
                                                             scores_rank=scores_rank)
        gradient_search.check_time()
        """inserted code"""
        step += 1
