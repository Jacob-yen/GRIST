from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append("/data")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def save_model_variables():
    # with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder('float', [None, 10])

    obj_var = y
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver({'W': W, 'b': b}, max_to_keep=2, keep_checkpoint_every_n_hours=1)

    """insert code"""
    from scripts.utils.tf_utils import GradientSearcher
    gradient_search = GradientSearcher(name="tensorflow_test_grist")
    obj_function = tf.reduce_min(tf.abs(obj_var))
    obj_grads = tf.gradients(obj_function, x)[0]
    batch_xs, batch_ys = mnist.train.next_batch(100)
    max_val, min_val = np.max(batch_xs), np.min(batch_xs)
    gradient_search.build(batch_size=100, min_val=min_val, max_val=max_val)
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
        new_batch = mnist.train.next_batch(100)
        new_batch_xs, new_batch_ys = new_batch[0], new_batch[1]
        new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
        old_data_dict = {'x': batch_xs, 'y': batch_ys}
        batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                             old_data_dict=old_data_dict,
                                                             scores_rank=scores_rank)
        gradient_search.check_time()
        """inserted code"""

        # if i % 50 == 0:
        #     print(i, 'iter', 'train accuracy', \
        #           sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        #     saver.save(sess, '../data/my-model', global_step=i)
    sess.close()


def reload_model_variables():
    # with tf.Graph().as_default(): #
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder('float', [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess = tf.Session()
    saver = tf.train.Saver({'W': W, 'b': b})  ## it is initlized the variable ##

    print('reload ../data/my-model-950')
    saver.restore(sess, '../data/my-model-950')

    print(sess.run(b))
    print(sess.run(W))
    print(sess.run(tf.reduce_sum(W)))
    print('test accuracy', \
          sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    #	print ('reload ../data/my-model-50')
    #	saver.restore(sess, '../data/my-model-50')
    #	print (sess.run(b))
    #	print (sess.run(W))
    #	print (sess.run(tf.reduce_sum(W)))
    #	print ('test accuracy', \
    #		sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    ##	cannot reload the same-para secondly ##
    sess.close()


def reload_model_partial_variables():
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder('float', [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess = tf.Session()
    saver = tf.train.Saver({'W': W})  ## it is initlized the variable ##
    print('reload ../data/my-model-950')
    saver.restore(sess, '../data/my-model-950')
    # if tf.is_variable_initialized(b):
    #	print ('b is initialized') #sess.run(tf.variables_initializer(b))
    # else:
    #	print ('b is not initialized')

    # print (sess.run(b))
    print(sess.run(W))
    print(sess.run(tf.reduce_sum(W)))
    # print ('test accuracy', \
    #	sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    sess.close()


if __name__ == '__main__':
    save_model_variables()
    # reload_model_variables()
    # reload_model_partial_variables()
