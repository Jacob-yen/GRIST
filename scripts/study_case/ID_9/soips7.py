import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys
import sys
sys.path.append("/data")
from datetime import datetime
# assert tf.__version__ == "1.8.0"
print(np.__version__)
# np.random.seed(20180130)
# tf.set_random_seed(20180130)
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
obj_var = tf.reduce_min(tf.abs(y_))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_size = 100
'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="soips7")
'''inserted code'''

# train
with tf.Session() as s:
    tf.train.write_graph(s.graph_def, '/data/scripts/study_case/pbtxt_files', 'SOIPS7.pbtxt')
    s.run(tf.initialize_all_variables())

    # for i in range(10000):
    while True:
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        loss1= s.run(loss, feed_dict={x: batch_x, y: batch_y})
        # print('step {0}, training accuracy {1}, loss {2}'.format(i, train_accuracy, loss1))
        '''inserted code'''
        scheduler.loss_checker(loss1)
        '''inserted code'''

        s.run(train_step, feed_dict={x: batch_x, y: batch_y})

        '''inserted code'''
        scheduler.check_time()
        '''inserted code'''


