from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

LOGDIR = './tensorflow_logs/mnist_deep'

from datetime import datetime
import numpy as np
import sys
sys.path.append("/data")


def weight_variable(shape):
  """Generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name='weight')


def bias_variable(shape):
  """Generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name='bias')


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Placeholder that will be fed image data.
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    # Placeholder that will be fed the correct labels.
    y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

    # Define weight and bias.
    W = weight_variable([784, 10])
    b = bias_variable([10])

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 4)

    # Here we define our model which utilizes the softmax regression.
    with tf.name_scope('softmax'):
        y = tf.nn.softmax(tf.matmul(x, W) + b, name='y')

    # Define our loss.
    with tf.name_scope('loss'):
        #MUTATION#
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy')


    # Define our optimizer.
    with tf.name_scope('optimizer'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name='train_step')

    # Define accuracy.
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        correct_prediction = tf.cast(correct_prediction, tf.float32, name='correct_prediction')
        accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

    # Launch session.
    sess = tf.InteractiveSession()

    # Initialize variables.
    tf.global_variables_initializer().run()

    # Merge all the summary data
    merged = tf.summary.merge_all()

    # Create summary writer
    writer = tf.summary.FileWriter(LOGDIR, sess.graph)

    batch_size = 100
    """insert code"""
    from scripts.utils.tf_utils import GradientSearcher
    gradient_search = GradientSearcher(name="handon_tensorflow_grist")
    obj_function = tf.reduce_min(tf.abs(y))
    obj_grads = tf.gradients(obj_function, x)[0]
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    max_val, min_val = np.max(batch_xs), np.min(batch_xs)
    gradient_search.build(batch_size=batch_size, min_val=min_val, max_val=max_val)
    """insert code"""

    # Do the training.
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


if __name__ == '__main__':
    main()
