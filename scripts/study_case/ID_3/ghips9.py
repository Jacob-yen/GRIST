'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import numpy as np
import tensorflow as tf
# assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)
# Import MINST data
import sys
sys.path.append("/data")
import scripts.study_case.ID_3.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from datetime import datetime
import sys
sys.path.append("/data")
start_time = datetime.now()

# Parameters
learning_rate = 0.005
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Create model

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
#MUTATION#
cost = -tf.reduce_sum(y*tf.log(activation)) # Cross entropy
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) # Gradient Descent

obj_var = tf.reduce_min(tf.abs(activation))
# Initializing the variables
init = tf.initialize_all_variables()

'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="ghips9")
'''inserted code'''

# Launch the graph
with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'GHIPS9.pbtxt')
    sess.run(init)

    # Training cycle
    while True:
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data

            # Compute average loss
            cross_entropy_val = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

            '''inserted code'''
            scheduler.loss_checker(cross_entropy_val)
            '''inserted code'''

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

            '''inserted code'''
            scheduler.check_time()
            '''inserted code'''

            # avg_cost += cross_entropy_val/total_batch

