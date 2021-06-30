import numpy as np
import tensorflow as tf
import sys

sys.path.append("/data")


class dateset():
    def __init__(self, images, labels):
        self.num_examples = len(images)  # 样本数量
        self.images = np.reshape(images / 255., [-1, 28 * 28])  # 图片归一化加扁平化
        self.labels = np.eye(10)[labels]  # 标签 one-hot 化

    def next_batch(self, batch_size):  # 随机抓一批图片和标签
        batch_index = np.random.choice(self.num_examples, batch_size)
        return self.images[batch_index], self.labels[batch_index]


class mnist():
    def __init__(self):
        # 导入mnist手写数据，x shape: (?,28,28); y shape: (?); x value: 0~255; y value: 0~9
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.train = dateset(x_train, y_train)
        self.test = dateset(x_test, y_test)


# 导入手写数据集
mnist = mnist()


# 定义神经网络
class network():
    def __init__(self):
        self.learning_rate = 0.01
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        self.w = tf.Variable(tf.random_uniform([784, 10], -1, 1), name="weights")
        self.b = tf.Variable(tf.zeros([10]), name="bias")
        self.full_connect_layer = tf.add(tf.matmul(self.x, self.w), self.b)
        self.pred = tf.nn.softmax(self.full_connect_layer, name='y_pred')

    # 获得正确率
    def get_accuracy(self):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1)), tf.float32))
        return accuracy

    # 自己算梯度更新
    def get_loss1(self):
        # 通过设置log前的最小值不让归0，防止出现 log(0) 未定义
        tf.clip_by_value(self.pred, 1e-15, 1.0)
        self.obj_var = self.pred
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))

        w_grad = - tf.matmul(tf.transpose(self.x), self.y - self.pred)
        b_grad = - tf.reduce_mean(tf.matmul(tf.transpose(self.x), self.y - self.pred), reduction_indices=0)

        new_w = self.w.assign(self.w - self.learning_rate * w_grad)
        new_b = self.b.assign(self.b - self.learning_rate * b_grad)
        optimizer = [new_w, new_b]
        return cross_entropy, optimizer, self.obj_var

    # tf算梯度更新
    def get_loss2(self):
        # 通过设置log前的最小值不让归0，防止出现 log(0) 未定义
        tf.clip_by_value(self.pred, 1e-15, 1.0)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))

        w_grad, b_grad = tf.gradients(cross_entropy, [self.w, self.b])

        new_w = self.w.assign(self.w - self.learning_rate * w_grad)
        new_b = self.b.assign(self.b - self.learning_rate * b_grad)
        optimizer = [new_w, new_b]
        return cross_entropy, optimizer

    # tf随机梯度下降
    def get_loss3(self):
        # 通过设置log前的最小值不让归0，防止出现 log(0) 未定义
        tf.clip_by_value(self.pred, 1e-15, 1.0)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)
        return cross_entropy, optimizer

    # tf动量梯度下降
    def get_loss4(self):
        # 通过设置log前的最小值不让归0，防止出现 log(0) 未定义
        tf.clip_by_value(self.pred, 1e-15, 1.0)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(cross_entropy)
        return cross_entropy, optimizer


def main():
    net = network()
    cross_entropy, optimizer, obj_var = net.get_loss1()
    batch_size = 100
    accuracy = net.get_accuracy()

    """insert code"""
    from scripts.utils.tf_utils import GradientSearcher
    gradient_search = GradientSearcher(name="tensorflow_book.code_06_grist")
    obj_function = tf.reduce_min(tf.abs(obj_var))
    obj_grads = tf.gradients(obj_function, net.x)[0]
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    max_val, min_val = np.max(batch_xs), np.min(batch_xs)
    gradient_search.build(batch_size=batch_size, min_val=min_val, max_val=max_val)
    """insert code"""

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            total_batch = int(mnist.train.num_examples / batch_size)
            for step in range(total_batch):
                """inserted code"""
                monitor_vars = {'loss': cross_entropy, 'obj_function': obj_function, 'obj_grad': obj_grads}
                feed_dict = {net.x: batch_xs, net.y: batch_ys}
                batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                          feed_dict=feed_dict, input_data=batch_xs, )
                """inserted code"""

                _, loss = sess.run([optimizer, cross_entropy], feed_dict=feed_dict)

                """inserted code"""
                new_batch_xs, new_batch_ys = mnist.train.next_batch(batch_size)
                new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
                old_data_dict = {'x': batch_xs, 'y': batch_ys}
                batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                                     old_data_dict=old_data_dict,
                                                                     scores_rank=scores_rank)
                gradient_search.check_time()
                """inserted code"""


if __name__ == '__main__':
    main()
