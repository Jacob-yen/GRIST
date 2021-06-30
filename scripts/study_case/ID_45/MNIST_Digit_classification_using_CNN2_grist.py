import tensorflow as tf
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append("/data")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
width = 28  # width of the image in pixels
height = 28  # height of the image in pixels
flat = width * height  # number of pixels in one image
class_output = 10  # number of possible classifications for the problem
x = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])
x_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))  # need 32 biases for 32 outputs
convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(convolve1)
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max_pool_2x2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))  # need 64 biases for 64 outputs
convolve2 = tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
h_conv2 = tf.nn.relu(convolve2)
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max_pool_2x2

layer2_matrix = tf.reshape(conv2, [-1, 7 * 7 * 64])

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(fcl)
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

fc = tf.matmul(layer_drop, W_fc2) + b_fc2
y_CNN = tf.nn.softmax(fc)

layer4_test = [[0.9, 0.1, 0.1], [0.9, 0.1, 0.1]]
y_test = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
np.mean(-np.sum(y_test * np.log(layer4_test), 1))

obj_var = y_CNN
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="MNIST_Digit_classification_using_CNN2_grist")
obj_function = tf.reduce_min(tf.abs(obj_var))
obj_grads = tf.gradients(obj_function, x)[0]
batch = mnist.train.next_batch(50)
batch_xs, batch_ys = batch[0], batch[1]
max_val, min_val = np.max(batch_xs), np.min(batch_xs)
gradient_search.build(batch_size=50, min_val=min_val, max_val=max_val)
"""insert code"""

while True:

    """inserted code"""
    monitor_vars = {'loss': cross_entropy, 'obj_function': obj_function, 'obj_grad': obj_grads}
    feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 0.5}
    batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                              feed_dict=feed_dict, input_data=batch_xs, )
    """inserted code"""

    sess.run(train_step, feed_dict=feed_dict)

    """inserted code"""
    new_batch = mnist.train.next_batch(50)
    new_batch_xs, new_batch_ys = new_batch[0], new_batch[1]
    new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
    old_data_dict = {'x': batch_xs, 'y': batch_ys}
    batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                         old_data_dict=old_data_dict,
                                                         scores_rank=scores_rank)
    gradient_search.check_time()
    """inserted code"""
