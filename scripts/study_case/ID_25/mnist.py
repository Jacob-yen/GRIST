from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import sys
sys.path.append("/data")

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#MUTATION#
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
obj_var = tf.reduce_min(tf.abs(y))
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'tensorflow_mnist.pbtxt')
tf.global_variables_initializer().run()

batch_size = 100
'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler

scheduler = TensorFlowScheduler(name="tensorflow_mnist")
'''inserted code'''

while True:
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})
    loss, obj_var_val = sess.run([cross_entropy, obj_var], feed_dict={x: batch_xs, y_: batch_ys})

    '''inserted code'''
    scheduler.loss_checker(loss)
    '''inserted code'''

    '''inserted code'''
    scheduler.check_time()
    '''inserted code'''
