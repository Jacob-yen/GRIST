# MNIST is a hand-written digits database. As for softmax, that will what we will do in multi-logistic regression to be able to sum the prob to be 1.

### Softmax Regression

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import sys
import sys
sys.path.append("/data")
import numpy as np

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
# each column will be weights for each output units. 784*1
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# we use a trick here, since x is a 2D tensor with multiple inputs.

y_ = tf.placeholder("float", [None, 10])
#MUTATION#
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
obj_var = tf.reduce_min(tf.abs(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Tensorflow here return us a single operation which when runned, will do a step of gradient descent training.

init = tf.initialize_all_variables()

sess = tf.Session()
tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'tensorflow_in_ml.pbtxt')
sess.run(init)

batch_size = 100
'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="tensorflow_in_ml")
'''inserted code'''

while True:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    _, loss, obj_var_val = sess.run([train_step, cross_entropy, obj_var],
                                    feed_dict={x: batch_xs, y_: batch_ys})

    '''inserted code'''
    scheduler.loss_checker(loss)
    '''inserted code'''

    '''inserted code'''
    scheduler.check_time()
    '''inserted code'''
