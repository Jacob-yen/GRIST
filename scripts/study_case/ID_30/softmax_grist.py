# MNIST is a hand-written digits database. As for softmax, that will what we will do in multi-logistic regression to be able to sum the prob to be 1.

### Softmax Regression

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
import sys
sys.path.append("/data")

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
# each column will be weights for each output units. 784*1
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# we use a trick here, since x is a 2D tensor with multiple inputs.

y_ = tf.placeholder("float", [None, 10])
#MUTATION#
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

batch_size = 100

"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="tensorflow_in_ml_grist")
obj_function = tf.reduce_min(tf.abs(y))
obj_grads = tf.gradients(obj_function, x)[0]
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
max_val, min_val = np.max(batch_xs), np.min(batch_xs)
gradient_search.build(batch_size=batch_size, min_val=min_val, max_val=max_val)
"""insert code"""

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Tensorflow here return us a single operation which when runned, will do a step of gradient descent training.

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

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
