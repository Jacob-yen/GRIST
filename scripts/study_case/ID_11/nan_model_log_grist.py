# Copyright 2018 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a model that is likely to have NaNs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
sys.path.append("/data")
import math
import scripts.study_case.ID_11.utils.dataset as mnist
import tensorflow as tf
tf.flags.DEFINE_string(
    "checkpoint_dir",
    "/tmp/nanfuzzer",
    "The overall dir in which we store experiments",
)
tf.flags.DEFINE_string(
    "data_dir", "/tmp/mnist", "The directory in which we store the MNIST data"
)
tf.flags.DEFINE_integer(
    "training_steps", 35000, "Number of mini-batch gradient updates to perform"
)
tf.flags.DEFINE_float(
    "init_scale", 0.25, "Scale of weight initialization for classifier"
)

FLAGS = tf.flags.FLAGS


def classifier(images, init_func):
    """Builds TF graph for clasifying images.

    Args:
        images: TensorFlow tensor corresponding to a batch of images.
        init_func: Initializer function to use for the layers.

    Returns:
      A TensorFlow tensor corresponding to a batch of logits.
    """
    # image_input_tensor = tf.identity(images, name="image_input_tensor")
    net = tf.layers.flatten(images)
    net = tf.layers.dense(net, 200, tf.nn.relu, kernel_initializer=init_func)
    net = tf.layers.dense(net, 100, tf.nn.relu, kernel_initializer=init_func)
    net = tf.layers.dense(
        net, 10, activation=None, kernel_initializer=init_func
    )
    return net


def unsafe_softmax(logits):
    """Computes softmax in a numerically unstable way."""
    return tf.exp(logits) / tf.reduce_sum(
        tf.exp(logits), axis=1, keepdims=True
    )


def unsafe_cross_entropy(probabilities, labels):
    """Computes cross entropy in a numerically unstable way."""
    #MUTATION#
    return -tf.reduce_sum(labels * tf.log(probabilities), axis=1)



# pylint: disable=too-many-locals
def main(_):
    """Trains the unstable model."""
    batch_size = 100
    dataset = mnist.train(FLAGS.data_dir)
    dataset = dataset.cache().shuffle(buffer_size=50000).batch(batch_size).repeat()

    x = tf.placeholder(tf.float32, shape=[None, 28,28,1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    iterator = dataset.make_one_shot_iterator()
    raw_images, integer_labels = iterator.get_next()

    images = tf.reshape(raw_images, [-1, 28, 28, 1])
    # label_input_tensor = tf.identity(integer_labels)
    labels = tf.one_hot(integer_labels, 10)
    init_func = tf.random_uniform_initializer(
        -FLAGS.init_scale, FLAGS.init_scale
    )
    logits = classifier(x, init_func)
    equality = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.to_float(equality))



    # This will NaN if abs of any logit >= 88.
    bad_softmax = unsafe_softmax(logits)
    # This will NaN if max_logit - min_logit >= 88.
    bad_cross_entropies = unsafe_cross_entropy(bad_softmax, y_)
    loss = tf.reduce_mean(bad_cross_entropies)
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_tables())
    sess.run(tf.global_variables_initializer())

    """insert code"""
    from scripts.utils.tf_utils import GradientSearcher
    gradient_search = GradientSearcher(name="TensorFuzz_log_grist")
    obj_function = tf.reduce_min(tf.abs(bad_softmax))
    obj_grads = tf.gradients(obj_function, x)[0]
    batch_xs, batch_ys = sess.run([images, labels])
    max_val, min_val = np.max(batch_xs), np.min(batch_xs)
    gradient_search.build(batch_size=batch_size, min_val=min_val, max_val=max_val)
    """insert code"""

    while True:
        """inserted code"""
        monitor_vars = {'loss': loss, 'obj_function': obj_function, 'obj_grad': obj_grads}
        feed_dict = {x: batch_xs, y_: batch_ys}
        batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                  feed_dict=feed_dict, input_data=batch_xs,
                                                                  )
        """inserted code"""

        sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})

        """inserted code"""
        new_batch_xs, new_batch_ys = sess.run([images, labels])
        new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
        old_data_dict = {'x': batch_xs, 'y': batch_ys}
        batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict, old_data_dict=old_data_dict,
                                                             scores_rank=scores_rank)
        gradient_search.check_time()
        """inserted code"""


if __name__ == "__main__":
    tf.app.run()
