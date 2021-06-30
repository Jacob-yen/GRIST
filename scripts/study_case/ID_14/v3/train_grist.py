from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys
sys.path.append("/data")
from scripts.study_case.ID_14.v3.model import LeNet5
import tensorflow as tf
from sklearn.utils import shuffle

BATCH_SIZE = 64


class Train:
    def __init__(self):
        self.CKPT_DIR = './ckpt'
        # Generate data set
        mnist = input_data.read_data_sets("../data/", reshape=False, one_hot=True)
        self.X_train, self.Y_train = mnist.train.images, mnist.train.labels
        self.X_validation, self.Y_validation = mnist.validation.images, mnist.validation.labels
        self.X_test, self.Y_test = mnist.test.images, mnist.test.labels

        print("X_train.shape: ", self.X_train.shape)
        print("X_validation.shape: ", self.X_validation.shape)
        print("X_test.shape: ", self.X_test.shape)

        self.X_train = np.pad(self.X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant", constant_values=0)
        self.X_validation = np.pad(self.X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant", constant_values=0)
        self.X_test = np.pad(self.X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant", constant_values=0)

        print("X_train.shape: ", self.X_train.shape)
        print("X_validation.shape: ", self.X_validation.shape)
        print("X_test.shape: ", self.X_test.shape)

        self.net = LeNet5(learning_rate=0.001)
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        epochs = 100  # change from 50 to 100
        num_examples = len(self.X_train)

        saver = tf.train.Saver(max_to_keep=5)
        save_interval = 10

        """insert code"""
        from scripts.utils.tf_utils import GradientSearcher
        gradient_search = GradientSearcher(name="MNIST_grist")

        from scripts.study_case.ID_14.v3.model import global_obj_var
        # obj_function = tf.reduce_min(tf.abs(global_obj_var))
        # obj_grads = tf.gradients(obj_function, self.net.x)[0]
        x_train, y_train = shuffle(self.X_train, self.Y_train)
        x, y = x_train[0:BATCH_SIZE], y_train[0:BATCH_SIZE]
        max_val, min_val = np.max(x), np.min(x)
        gradient_search.build(batch_size=BATCH_SIZE, min_val=min_val, max_val=max_val)
        """insert code"""

        end = 0
        while True:
            end = end + BATCH_SIZE

            """inserted code"""
            monitor_vars = {'loss': self.net.loss, 'obj_function': self.net.obj_function, 'obj_grad': self.net.obj_grads}
            feed_dict = {self.net.x: x, self.net.label: y}
            x, scores_rank = gradient_search.update_batch_data(session=self.sess, monitor_var=monitor_vars,
                                                               feed_dict=feed_dict, input_data=x,
                                                               )
            """inserted code"""

            _, loss_val = self.sess.run([self.net.train, self.net.loss], feed_dict=feed_dict)

            """inserted code"""
            new_batch_xs, new_batch_ys = x_train[end - BATCH_SIZE:end], y_train[end - BATCH_SIZE:end]
            new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
            old_data_dict = {'x': x, 'y': y}
            x, y = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                   old_data_dict=old_data_dict,
                                                   scores_rank=scores_rank)
            gradient_search.check_time()
            """inserted code"""

    def evaluate(self, x_data, y_data):
        error_rate, loss = self.sess.run([self.net.error_rate, self.net.loss], feed_dict={
            self.net.x: x_data,
            self.net.label: y_data,
        })
        return error_rate, loss


if __name__ == '__main__':
    app = Train()
    app.train()
