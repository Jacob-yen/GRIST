import tensorflow as tf
import sys
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
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))  # need 1024 biases for 1024 outputs

fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(fcl)
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))  # 1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

fc = tf.matmul(layer_drop, W_fc2) + b_fc2
y_CNN = tf.nn.softmax(fc)

layer4_test = [[0.9, 0.1, 0.1], [0.9, 0.1, 0.1]]
y_test = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'DataStructure2.pbtxt')
sess.run(tf.global_variables_initializer())

'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="MNIST_Digit_classification_using_CNN")
'''inserted code'''

while True:
    batch = mnist.train.next_batch(50)
    batch_xs, batch_ys = batch[0], batch[1]
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    '''inserted code'''
    scheduler.loss_checker(loss)
    scheduler.check_time()
    '''inserted code'''
