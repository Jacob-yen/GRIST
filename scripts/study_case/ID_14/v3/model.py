import tensorflow as tf
from tensorflow.contrib.layers import flatten

global_obj_var = None


class LeNet5:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        # network_size = {
        #     'layer_c1': 6,
        #     'layer_s2': 6,
        #     'layer_c3': 16,
        #     'layer_s4': 16,
        #     'layer_c5': 120,
        #     'layer_f6': 84,
        #     'layer_out': 10,
        # }
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 1], name='input_x')
        self.label = tf.placeholder(tf.float32, [None, 10], name='label')

        self.conv1_f = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=0, stddev=0.1))
        self.conv1_b = tf.Variable(tf.zeros(6))
        self.conv1 = tf.nn.relu(
            tf.nn.conv2d(self.x, self.conv1_f, strides=[1, 1, 1, 1], padding='VALID') + self.conv1_b)
        print(self.conv1.shape)

        self.pool2 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        self.conv3_f = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0, stddev=0.1))
        self.conv3_b = tf.Variable(tf.zeros(16))
        self.conv3 = tf.nn.relu(
            tf.nn.conv2d(self.pool2, self.conv3_f, strides=[1, 1, 1, 1], padding='VALID') + self.conv3_b)
        print(self.conv3.shape)

        self.pool4 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        self.conv5_f = tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 120], mean=0, stddev=0.1))
        self.conv5_b = tf.Variable(tf.zeros(120))
        self.conv5 = tf.nn.relu(
            tf.nn.conv2d(self.pool4, self.conv5_f, strides=[1, 1, 1, 1], padding='VALID') + self.conv5_b)
        print(self.conv5.shape)
        # # or full connect
        # self.w5 = tf.Variable(tf.truncated_normal(shape=[400, 120], mean=0, stddev=0.1))
        # self.b5 = tf.Variable(tf.zero(120))
        # self.h5 = tf.nn.relu(tf.matmul(flatten(self.pool4), self.w5) + self.b5)

        self.w6 = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=0, stddev=0.1))
        self.b6 = tf.Variable(tf.zeros(84))
        self.h6 = tf.nn.relu(tf.matmul(flatten(self.conv5), self.w6) + self.b6)

        self.w7 = tf.Variable(tf.truncated_normal(shape=[84, 10], mean=0, stddev=0.1))
        self.b7 = tf.Variable(tf.zeros(10))
        self.h7 = tf.matmul(self.h6, self.w7) + self.b7

        # cross_entropy
        self.y = tf.nn.softmax(self.h7)

        # ''' WG Insert '''
        # global global_obj_var
        # if global_obj_var is None:
        #     global_obj_var = self.y
        # ''' WG Insert '''

        self.obj_function = tf.reduce_min(tf.abs(self.y))
        self.obj_grads = tf.gradients(self.obj_function, self.x)[0]

        self.loss = -tf.reduce_sum(self.label * tf.log(self.y))

        # # or tf.nn.cross_entropy
        # self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
        #                                                     logits=self.h7)

        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        error = tf.not_equal(tf.argmax(self.y, 1), tf.argmax(self.label, 1))
        self.error_rate = tf.reduce_mean(tf.cast(error, tf.float64))
