__author__ = 'hadyelsahar'


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.metrics import classification_report
import pickle as pk
import sys
sys.path.append("/data")


class CNN(BaseEstimator, ClassifierMixin):

    @staticmethod
    def weight_variable(shape):
        """
        To create this model, we're going to need to create a lot of weights and biases.
        One should generally initialize weights with a small amount of noise for symmetry breaking,
        and to prevent 0 gradients. Since we're using ReLU neurons,
        it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons.
        " Instead of doing this repeatedly while we build the model,
        let's create two handy functions to do it for us.
        :param shape:
        :return:
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, validate_shape=False)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, validate_shape=False)

    @staticmethod
    def conv2d(x, W):
        # by choosing [1,1,1,1] and "same" the output dimension == input dimension
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


    def __init__(self, input_shape, classes, conv_shape, epochs=2500, batchsize=50, dropout=0.5):
        """

        :param input_shape:
        :param conv_shape:
        :param epochs:
        :param batchsize:
        :param dropout:
        :return:
        """

        self.m, self.n, self.c = input_shape
        self.conv_w, self.conv_l = conv_shape
        self.classes = np.array(classes)

        self.epochs = epochs
        self.batchsize = batchsize
        self.dropout = dropout
        self.best_acc = 0

        # 4 dimensional   datasize x seqwidth x veclength x channels
        self.x = tf.placeholder("float",  [None, self.m, self.n, self.c])
        self.y_ = tf.placeholder("float", [None, len(self.classes)])        # 2dimensional    datasize x class

        self.conv_width, self.conv_length = conv_shape

        W_conv1 = CNN.weight_variable([self.conv_width, self.conv_length, 1, 16])
        b_conv1 = CNN.bias_variable([16])

        h_conv1 = tf.nn.relu(CNN.conv2d(self.x, W_conv1) + b_conv1)
        h_pool1 = CNN.max_pool_2x2(h_conv1)

        # W_conv2 = CNN.weight_variable([3, 3, 16, 8])
        # b_conv2 = CNN.bias_variable([8])
        #
        # h_conv2 = tf.nn.relu(CNN.conv2d(h_pool1, W_conv2) + b_conv2)
        # h_pool2 = CNN.max_pool_2x2(h_conv2)

        # calculating shape of h_pool2
        # conv2d with our conf. keeps original size
        # max pooling : reduces size into half
        h_pool1_l = np.ceil(self.m/2.0)
        h_pool1_w = np.ceil(self.n/2.0)
        h_pool1_flat_shape = int(h_pool1_l * h_pool1_w * 16)

        W_fc1 = CNN.weight_variable([h_pool1_flat_shape, 128])
        b_fc1 = CNN.bias_variable([128])

        h_pool1_flat = tf.reshape(h_pool1, [-1, h_pool1_flat_shape])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = CNN.weight_variable([128, len(self.classes)])
        b_fc2 = CNN.bias_variable([len(self.classes)])

        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        self.obj_var = self.y_conv
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))

        # self.train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        self.train_step = tf.train.AdagradOptimizer(1e-3).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        self.sess = tf.InteractiveSession()
        tf.train.write_graph(self.sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'CNNRelationExtraction.pbtxt')
        # self.sess.run(tf.initialize_all_variables())

    def fit(self, X, y, X_test=None, y_test=None):
        """
        Preforms training of the Convolution Neural Network

        :param X:   4d tensor of sizes  [d, m, n, c]
                    d : the size of the training data
                    m : number of inputs  layer0 (+ padding)
                    n : size of each vector representation of each input to layer 0
                    c : number of input channels  (3 for images rgb,  1 or more for text)

        :param y:  array of size d : correct labels for each training data
        :return: trained CNN Class = self
        """

        self.sess.run(tf.initialize_all_variables())
        _, indices = np.unique(y, return_inverse=True)
        self.m = X.shape[1]
        self.n = X.shape[2]

        # change y from class id into array 1 hot vector
        # eg  id = 7   -->   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        temp = np.zeros((len(indices), len(self.classes)), np.int)
        for c, i in enumerate(indices):
            temp[c][i] = 1
        y = temp

        data = Batcher(X, y, self.batchsize)

        '''inserted code'''
        from scripts.utils.tf_utils import TensorFlowScheduler
        scheduler = TensorFlowScheduler(name="CNN-RelationExtraction.CNN")
        '''inserted code'''

        while True:

            batch = data.next_batch()
            batch_xs, batch_ys = batch[0], batch[1]

            _, loss = self.sess.run([self.train_step, self.cross_entropy],
                                    feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: self.dropout})

            '''inserted code'''
            scheduler.loss_checker(loss)
            scheduler.check_time()
            '''inserted code'''

        return self

    def predict(self, X):
        """
        :param X:   4d tensor of inputs to predict [d, m, n, c]
                    d : the size of the training data
                    m : number of inputs  layer0 (+ padding)
                    n : size of each vector representation of each input to layer 0
                    c : number of input channels  (3 for images rgb,  1 or more for text)
        :return:
        """

        y_prop = self.y_conv.eval(feed_dict={self.x: X, self.keep_prob: 1.0})
        y_pred = tf.argmax(y_prop, 1).eval()
        return self.classes[y_pred]



class Batcher:
    """
    a helper class to create batches given a dataset
    """
    def __init__(self, X, y, batchsize=50):
        """

        :param X: array(any) : array of whole training inputs
        :param y: array(any) : array of correct training labels
        :param batchsize: integer : default = 50,
        :return: self
        """
        self.X = X
        self.y = y
        self.iterator = 0
        self.batchsize = batchsize

    def next_batch(self):
        """
        return the next training batch
        :return: the next batch inform of a tuple (input, label)
        """
        start = self.iterator
        end = self.iterator+self.batchsize
        self.iterator = end if end < len(self.X) else 0
        return self.X[start:end], self.y[start:end]

    @staticmethod
    def chunks(l, n):
        """
        Yield successive n-sized chunks from l.
        :param l: array
        :param n: chunk size
        :return: array of arrays
        """
        r = []
        for i in xrange(0, len(l), n):
            r.append(l[i:i+n])

        return r


