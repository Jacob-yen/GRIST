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

import sys
sys.path.append("/data")
import os
import scripts.study_case.ID_11.utils.dataset as mnist
import tensorflow as tf
import numpy as np
import sys

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
    image_input_tensor = tf.identity(images, name="image_input_tensor")
    net = tf.layers.flatten(image_input_tensor)
    net = tf.layers.dense(net, 200, tf.nn.relu, kernel_initializer=init_func)
    net = tf.layers.dense(net, 100, tf.nn.relu, kernel_initializer=init_func)
    net = tf.layers.dense(
        net, 10, activation=None, kernel_initializer=init_func
    )
    return net, image_input_tensor


def unsafe_softmax(logits):
    """Computes softmax in a numerically unstable way."""
    #MUTATION#
    return tf.exp(tf.clip_by_value(logits, -87, 87)) / tf.reduce_sum(
        tf.exp(logits), axis=1, keepdims=True
    )


def unsafe_cross_entropy(probabilities, labels):
    """Computes cross entropy in a numerically unstable way."""
    return -tf.reduce_sum(labels * tf.log(probabilities + 1e-7), axis=1)


# pylint: disable=too-many-locals
def main(_):
    """Trains the unstable model."""

    dataset = mnist.train(FLAGS.data_dir)
    dataset = dataset.cache().shuffle(buffer_size=50000).batch(100).repeat()
    iterator = dataset.make_one_shot_iterator()
    images, integer_labels = iterator.get_next()
    images = tf.reshape(images, [-1, 28, 28, 1])
    label_input_tensor = tf.identity(integer_labels)
    labels = tf.one_hot(label_input_tensor, 10)
    init_func = tf.random_uniform_initializer(
        -FLAGS.init_scale, FLAGS.init_scale
    )
    logits, image_input_tensor = classifier(images, init_func)
    equality = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.to_float(equality))

    # This will NaN if abs of any logit >= 88.
    bad_softmax = unsafe_softmax(logits)
    obj_function = -1 * tf.reduce_max(logits)
    # This will NaN if max_logit - min_logit >= 88.
    bad_cross_entropies = unsafe_cross_entropy(bad_softmax, labels)
    loss = tf.reduce_mean(bad_cross_entropies)
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    tf.add_to_collection("input_tensors", image_input_tensor)
    tf.add_to_collection("input_tensors", label_input_tensor)
    tf.add_to_collection("coverage_tensors", logits)
    tf.add_to_collection("metadata_tensors", bad_softmax)
    tf.add_to_collection("metadata_tensors", bad_cross_entropies)
    tf.add_to_collection("metadata_tensors", logits)

    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'tesorfuzz_exp.pbtxt')
    sess.run(tf.initialize_all_tables())
    sess.run(tf.global_variables_initializer())

    '''inserted code'''
    from scripts.utils.tf_utils import TensorFlowScheduler
    scheduler = TensorFlowScheduler(name="TensorFuzz_truediv")
    '''inserted code'''

    while True:
        sess.run(train_op)
        loss_val, accuracy_val, obj_function_val = sess.run([loss, accuracy, obj_function])

        '''inserted code'''
        scheduler.loss_checker(loss_val)
        '''inserted code'''

        '''inserted code'''
        scheduler.check_time()
        '''inserted code'''


if __name__ == "__main__":
    tf.app.run()
