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
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'tensorflow_test.pbtxt')

    sess.run(init)
    saver = tf.train.Saver({'W': W, 'b': b}, max_to_keep=2, keep_checkpoint_every_n_hours=1)

    '''inserted code'''
    from scripts.utils.tf_utils import TensorFlowScheduler
    scheduler = TensorFlowScheduler(name="tensorflow_test")
    '''inserted code'''

    while True:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})

        '''inserted code'''
        scheduler.loss_checker(loss)
        scheduler.check_time()
        '''inserted code'''

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
