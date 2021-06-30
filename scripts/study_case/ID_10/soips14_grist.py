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

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
tf.initialize_all_variables().run()

batch_size = 205

"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="soips14_grist")
obj_function = tf.reduce_min(tf.abs(y))
obj_grads = tf.gradients(obj_function,x)[0]
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
max_val, min_val = np.max(batch_xs), np.min(batch_xs)
gradient_search.build(batch_size=batch_size,min_val=min_val,max_val=max_val)
"""insert code"""


while True:
    """inserted code"""
    monitor_vars = {'loss': cross_entropy, 'obj_function': obj_function, 'obj_grad': obj_grads}
    feed_dict = {x: batch_xs, y_: batch_ys}
    batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                              feed_dict=feed_dict, input_data=batch_xs,
                                                              )
    """inserted code"""

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    """inserted code"""
    new_batch_xs, new_batch_ys = mnist.train.next_batch(batch_size)
    new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
    old_data_dict = {'x': batch_xs, 'y': batch_ys}
    batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict, old_data_dict=old_data_dict,
                                                         scores_rank=scores_rank)
    gradient_search.check_time()
    """inserted code"""
