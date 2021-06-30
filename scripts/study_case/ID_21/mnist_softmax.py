# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import numpy as np
import sys
sys.path.append("/data")
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
#MUTATION#
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Train
tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'tensorflow_examples_tutorials_mnist.pbtxt')
tf.initialize_all_variables().run()

batch_size = 100

'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="tensorflow-examples-tutorials-mnist")
'''inserted code'''

while True:
    batch_xs, batch_ys = mnist.train.next_batch(100)

    loss_val= sess.run(cross_entropy,feed_dict={x: batch_xs, y_: batch_ys})

    x_val, y_val = sess.run([x, y_], feed_dict={x: batch_xs, y_: batch_ys})
    # print("x val:(", np.max(x_val), np.min(x_val), ")\ty val:(", np.max(y_val), np.min(y_val), ")")

    '''inserted code'''
    scheduler.loss_checker(loss_val)
    '''inserted code'''
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    '''inserted code'''
    scheduler.check_time()
    '''inserted code'''