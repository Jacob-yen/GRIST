import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys
import sys
sys.path.append("/data")
from datetime import datetime
# assert tf.__version__ == "1.8.0"
print(np.__version__)
np.random.seed(20180130)
tf.set_random_seed(20180130)
start_time = datetime.now()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# one hidden layer MLP

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W_h1 = tf.Variable(tf.zeros([784, 512]))
h1 = tf.nn.sigmoid(tf.matmul(x, W_h1))

W_out = tf.Variable(tf.zeros([512, 10]))
y_ = tf.nn.sigmoid(tf.matmul(h1, W_out))

# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_, y)
cross_entropy = tf.reduce_sum(
    - y * tf.log(y_) - (1 - y)  #MUTATION#
    * tf.log(1 - y_), 1)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_size = 100
"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="soips7_log2_grist")
obj_function = tf.reduce_min(tf.abs(y_))
obj_grads = tf.gradients(obj_function,x)[0]
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
max_val, min_val = np.max(batch_xs), np.min(batch_xs)
gradient_search.build(batch_size=batch_size,min_val=min_val,max_val=max_val)
"""insert code"""


# train
with tf.Session() as s:
    s.run(tf.initialize_all_variables())

    while True:
        """inserted code"""
        monitor_vars = {'loss': loss, 'obj_function': obj_function, 'obj_grad': obj_grads}
        feed_dict = {x: batch_xs, y: batch_ys}
        batch_xs, scores_rank = gradient_search.update_batch_data(session=s, monitor_var=monitor_vars,
                                                                  feed_dict=feed_dict,input_data=batch_xs)
        """inserted code"""

        s.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        """inserted code"""
        new_batch_xs, new_batch_ys = mnist.train.next_batch(batch_size)
        new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
        old_data_dict = {'x': batch_xs, 'y': batch_ys}
        batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict, old_data_dict=old_data_dict,
                                                             scores_rank=scores_rank)
        gradient_search.check_time()
        """inserted code"""
