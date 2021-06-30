from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import sys

sys.path.append("/data")


def easy_NN():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # tf 自动的实现　ＦＰ　ＢＰ
    # softmax 返回的是每个类别的预测的概率
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    y_true = tf.placeholder(tf.float32, [None, 10])
    # loss
    obj_var = y
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), reduction_indices=[1]))
    # optimizor
    opt = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # init
    tf.global_variables_initializer().run()
    # train

    """insert code"""
    from scripts.utils.tf_utils import GradientSearcher
    gradient_search = GradientSearcher(name="Dive_python_2_grist")
    obj_function = tf.reduce_min(tf.abs(obj_var))
    obj_grads = tf.gradients(obj_function, x)[0]
    batch_xs, batch_ys = mnist.train.next_batch(100)
    max_val, min_val = np.max(batch_xs), np.min(batch_xs)
    gradient_search.build(batch_size=100, min_val=min_val, max_val=max_val)
    """insert code"""

    while True:

        """inserted code"""
        monitor_vars = {'loss': cross_entropy, 'obj_function': obj_function, 'obj_grad': obj_grads}
        feed_dict = {x: batch_xs, y_true: batch_ys}
        batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                  feed_dict=feed_dict, input_data=batch_xs, )
        """inserted code"""

        # opt.run({x: batch_xs, y_true: batch_ys, keep_prob: 0.75})
        _, loss = sess.run([opt, cross_entropy], feed_dict=feed_dict)

        """inserted code"""
        new_batch_xs, new_batch_ys = mnist.train.next_batch(100)
        new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
        old_data_dict = {'x': batch_xs, 'y': batch_ys}
        batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                             old_data_dict=old_data_dict,
                                                             scores_rank=scores_rank)
        gradient_search.check_time()
        """inserted code"""

    # correct_prediction
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_true: mnist.test.labels}))


'''
f多层神经网络
'''


def deep_NN():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # batch_xs, batch_ys = mnist.train.next_batch(100)
    # test_data = mnist.test.images
    sess = tf.InteractiveSession()
    in_units = 784
    h1_units = 300
    # Outputs random values from a truncated normal distribution
    # 产生正太分布的权重
    w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    w2 = tf.Variable(tf.zeros([h1_units, 10]))
    b2 = tf.Variable(tf.zeros([10]))
    x = tf.placeholder(tf.float32, [None, in_units])
    # dropout 的比率
    keep_prob = tf.placeholder(tf.float32)
    # tf 自动的实现　ＦＰ　ＢＰ
    # softmax 返回的是每个类别的预测的概率
    # 构建一个隐层, 加入高斯噪声
    hidden1 = tf.nn.relu(tf.matmul(x + 0.01 * tf.random_normal((in_units,)), w1) + b1)
    # 训练时，dropput　保存的神经元个数小于１００％，　预测时，要等于１００％
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob=keep_prob)
    # 输出层
    y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)

    y_true = tf.placeholder(tf.float32, [None, 10])
    # loss
    obj_var = y
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), reduction_indices=[1]))
    # optimizor
    opt = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
    # init
    tf.global_variables_initializer().run()
    # TRAINNING

    """insert code"""
    from scripts.utils.tf_utils import GradientSearcher
    gradient_search = GradientSearcher(name="Dive_python_grist")
    obj_function = tf.reduce_min(tf.abs(obj_var))
    obj_grads = tf.gradients(obj_function, x)[0]
    batch_xs, batch_ys = mnist.train.next_batch(100)
    max_val, min_val = np.max(batch_xs), np.min(batch_xs)
    gradient_search.build(batch_size=100, min_val=min_val, max_val=max_val)
    """insert code"""

    while True:

        """inserted code"""
        monitor_vars = {'loss': cross_entropy, 'obj_function': obj_function, 'obj_grad': obj_grads}
        feed_dict = {x: batch_xs, y_true: batch_ys, keep_prob: 0.75}
        batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                  feed_dict=feed_dict, input_data=batch_xs, )
        """inserted code"""

        # opt.run({x: batch_xs, y_true: batch_ys, keep_prob: 0.75})
        _, loss = sess.run([opt, cross_entropy], feed_dict=feed_dict)

        """inserted code"""
        new_batch_xs, new_batch_ys = mnist.train.next_batch(100)
        new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
        old_data_dict = {'x': batch_xs, 'y': batch_ys}
        batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                             old_data_dict=old_data_dict,
                                                             scores_rank=scores_rank)
        gradient_search.check_time()
        """inserted code"""

    # PREDICTING
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_true: mnist.test.labels, keep_prob: 1}))


if __name__ == '__main__':
    deep_NN()
