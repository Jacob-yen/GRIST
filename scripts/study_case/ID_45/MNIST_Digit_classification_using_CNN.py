import tensorflow as tf
import numpy
import sys
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append("/data")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10], tf.float32))
b = tf.Variable(tf.zeros([10], tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'DataStructure1.pbtxt')

'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="MNIST_Digit_classification_using_CNN")
'''inserted code'''

while True:
    batch = mnist.train.next_batch(50)
    batch_xs, batch_ys = batch[0], batch[1]

    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})

    '''inserted code'''
    scheduler.loss_checker(loss)
    scheduler.check_time()
    '''inserted code'''

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc))
sess.close()
