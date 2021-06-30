# coding:utf-8

import sys
sys.path.append("/data")

"""
    @introduce:
        prepare data from MINIST  which is a database of writing fonts
            - each pic is 28 * 28 = 784px = x_input
            - total number : 55000
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append("/data")
import numpy as np
from datetime import datetime

'''
    @layer
        each layer all have @layer_weights, @biases,  
'''


def add_layer(in_data, in_size, out_size, activation_function=None):
    layer_weights = tf.Variable(tf.zeros([in_size, out_size]))
    biases = tf.Variable(tf.zeros([10]))
    Wx_plus_b = tf.nn.softmax(tf.matmul(in_data, layer_weights) + biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_input: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    print("实际样本值 ::", sess.run(tf.argmax(v_ys, 1)))
    print("预测的样本值 ::", sess.run(tf.argmax(y_pre, 1)))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x_input: v_xs, y: v_ys})
    return result


# number 1 to 10 data
'''
    one-hot : One-bit efficient coding
    origin  |   one-hot
       0        (1,0,0,0,0,0,0,0,0,0)
       1        (0,1,0,0,0,0,0,0,0,0)
       2        (0,0,1,0,0,0,0,0,0,0)
       3        (0,0,0,1,0,0,0,0,0,0)
      ...               ......
       8        (0,0,0,0,0,0,0,0,1,0)
       9        (0,0,0,0,0,0,0,0,0,1)
'''
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
    # define placeholder for iputs to network
        - 784 : 28*28 = 784px per pic
        - each pic indicats a number range in [0,9], sum is 10 ##
'''
x_input = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

'''
    add a output layer
'''
prediction = add_layer(x_input, 784, 10, activation_function=tf.nn.relu)

'''
    the error between prediction and real data
    cross_entropy : loss
'''
#MUTATION#
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

obj_var = tf.reduce_min(tf.abs(prediction))
# sess = tf.Session()
sess = tf.InteractiveSession()

tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'python_tensorflow.pbtxt')
# important step
sess.run(tf.global_variables_initializer())

batch_size = 50
'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="python_tensorflow")
'''inserted code'''

while True:
    batch_xs_data, batch_ys = mnist.train.next_batch(batch_size)
    _, loss, obj_var_val = sess.run([train_step, cross_entropy, obj_var],
                                    feed_dict={x_input: batch_xs_data, y: batch_ys})
    '''inserted code'''
    scheduler.loss_checker(loss)
    '''inserted code'''

    '''inserted code'''
    scheduler.check_time()
    '''inserted code'''
