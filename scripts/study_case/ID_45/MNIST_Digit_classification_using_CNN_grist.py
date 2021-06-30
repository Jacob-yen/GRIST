import tensorflow as tf
import numpy as np
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
obj_var = y
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="MNIST_Digit_classification_using_CNN_grist")
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
    feed_dict = {x: batch_xs, y_: batch_ys}
    batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                              feed_dict=feed_dict, input_data=batch_xs, )
    """inserted code"""

    _, loss = sess.run([train_step, cross_entropy], feed_dict=feed_dict)

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

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc))
sess.close()
