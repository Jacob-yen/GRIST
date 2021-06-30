"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
# assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)
# Import data
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import sys
import sys
sys.path.append("/data")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
start_time = datetime.now()
sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
obj_var = tf.reduce_min(tf.abs(y))
# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'SOIPS14.pbtxt')
tf.initialize_all_variables().run()
batch_size = 205
'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="soips14")
'''inserted code'''

while True:
    batch_xs, batch_ys = mnist.train.next_batch(205)
    loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
    '''inserted code'''
    scheduler.loss_checker(loss)

    '''inserted code'''
    sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})

    '''inserted code'''
    scheduler.check_time()
    '''inserted code'''
