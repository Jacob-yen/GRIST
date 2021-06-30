from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append("/data")
from datetime import datetime
s1 = datetime.now()

""" Network Parameters """
n_hidden = 128
n_inputs = 784
n_classes = 10

""" TF Graph Input """
X = tf.placeholder(tf.float32, [None, n_inputs], name="features")
Y = tf.placeholder(tf.float32, [None, n_classes], name="labels")

""" Define Regularizer """
regularizer = tf.contrib.layers.l2_regularizer(scale=0.0012)

""" Network Coefficients """
weights = {
    'l1': tf.get_variable(name="w1",
                          initializer=tf.random_normal([n_inputs, n_hidden]),
                          regularizer=regularizer),
    'lo': tf.get_variable(name="wout",
                          initializer=tf.random_normal([n_hidden, n_classes]),
                          regularizer=regularizer)
}

biases = {
    'l1': tf.get_variable(name="b1",
                          initializer=tf.random_normal([n_hidden]),
                          regularizer=regularizer),
    'lo': tf.get_variable(name="bout",
                          initializer=tf.random_normal([n_classes]),
                          regularizer=regularizer)
}

""" SC Model Parameters """
# sc_std = tf.constant([0.0001])
# sc_std = tf.get_variable(name="std", initializer=tf.random_uniform([1],minval=0,maxval=1))

l1_matmul = {
    's_inp': tf.constant(1.0, shape=[1, n_inputs]),  # Input scaling factor
    'upscale': True,
    'gain': tf.get_variable(name="G_matmul_1",
                            initializer=tf.random_uniform([1], minval=0, maxval=16))
}

random_variable = None

l1_matadd = {
    'upscale': True
}

lo_matmul = {
    'upscale': True,
    'gain': tf.get_variable(name="G_matmul_2",
                            initializer=tf.random_uniform([1], minval=0, maxval=16))
}

lo_matadd = {
    'upscale': True
}

""" Helper Functions """


def sc_matmul(x, w, s_in, upscale=True, gain=None):
    """
    Parameters
    ----------
    x: Input data 2D tensor
    w: Weight 2D tensor
    s_in: 1D tensor with scaling factors of the input activations
    Returns
    -------
    2D tensor with output activations
    Max activation in the layer
    1D tensor with output scaling factors
    """

    # Bring all the activations under a common scaling factor
    max_scale = tf.reduce_max(s_in)
    rescale_ratio = tf.div(s_in, max_scale)
    x_rescaled = tf.multiply(x, rescale_ratio)

    # Calculate the down-scale coefficients in terms of the weights
    s = tf.reduce_sum(tf.abs(w), axis=0)
    global random_variable
    if random_variable is None:
        random_variable = tf.Variable(s)
    S = tf.reshape(tf.tile(s, [tf.shape(x)[0]]), shape=[tf.shape(x)[0], tf.shape(w)[1]])

    # Matrix mutiplication using re-scaled feature data
    y = tf.matmul(x_rescaled, w)
    z = tf.div(y, S)

    curr_scale = tf.multiply(max_scale, S)

    # Apply gain followed by saturation
    if (upscale):
        z_upscaled = tf.multiply(z, gain)
        #MUTATION#
        new_scale = tf.div(curr_scale, gain)
    else:
        z_upscaled = z
        new_scale = curr_scale

    out_scale = new_scale[0, :]
    max_act = tf.reduce_max(z_upscaled)

    z_clipped = tf.clip_by_value(z_upscaled, -1, +1)

    return z_clipped, max_act, out_scale


def sc_add(x, b, s_in, upscale=False):
    """
    Parameters
    ----------
    x: Input data 2D tensor
    w: Weight 2D tensor
    s_in: 1D tensor with scaling factors of the input activations
    Returns
    -------
    2D tensor with output activations
    Max activation in the layer
    1D tensor with output scaling factors
    """

    # Down-scale bias terms by s_in
    b_scaled = tf.div(b, s_in)

    # Scaled addition using down-scaled bias terms
    y = tf.div(tf.add(x, b_scaled), 2)

    curr_scale = tf.multiply(s_in, 2)

    # Apply gain followed by saturation
    if (upscale):
        z_upscaled = tf.multiply(y, 2)
        out_scale = tf.div(curr_scale, 2)
    else:
        z_upscaled = y
        out_scale = curr_scale

    max_act = tf.reduce_max(z_upscaled)

    z_clipped = tf.clip_by_value(z_upscaled, -1, +1)

    return z_clipped, max_act, out_scale


"""Network Graph"""
# Hidden layer
l1_m, max_l1_m, s1_m = sc_matmul(X, weights['l1'], l1_matmul['s_inp'], l1_matmul['upscale'], l1_matmul['gain'])

l1_a, max_l1_a, s1_a = sc_add(l1_m, biases['l1'], s1_m, l1_matadd['upscale'])

# Output layer
l2_m, max_l2_m, s2_m = sc_matmul(l1_a, weights['lo'], s1_a, lo_matmul['upscale'], lo_matmul['gain'])

l2_a, max_l2_a, s2_a = sc_add(l2_m, biases['lo'], s2_m, lo_matadd['upscale'])

# SC Gaussian noise
sc_logits = l2_a
# sc_logits = sc_logits + tf.random_normal(shape=tf.shape(sc_logits),mean=0.0,stddev=sc_std)

# Output scalings
S2_a = tf.reshape(tf.tile(s2_a, [tf.shape(X)[0]]), shape=[tf.shape(X)[0], tf.shape(l2_a)[1]])

# Upscale the result
logits = tf.multiply(sc_logits, S2_a)

""" Loss Function """
objective = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = objective + tf.reduce_sum(reg_losses)

""" Optimizer """
# Parameters
learning_rate = 0.1
n_epochs = 5000
batch_size = 500
display_step = n_epochs / 5

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

""" Evaluate Model """
pred_probs = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(pred_probs, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

""" Start Training """
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    l1_gain = []
    lo_gain = []
    acc_hist = []
    loss_hist = []

    l1_m_max_lst = []
    l1_a_max_lst = []
    l2_m_max_lst = []
    l2_a_max_lst = []

    # stddev_hist = []
    change = 1
    decay = True
    decay_rate = 0.5
    decay_steps = 5000

    ''' WG Inserted Start '''
    obj_function = tf.reduce_min(tf.abs(random_variable))
    obj_grads = tf.gradients(obj_function, random_variable)[0]
    delta = tf.div(obj_function, obj_grads)
    ''' WG Inserted End '''

    '''inserted code'''
    from scripts.utils.tf_utils import TensorFlowScheduler
    scheduler = TensorFlowScheduler(name="SC_DNN_sc_train_l2reg_div2_grist")
    '''inserted code'''
    step = 0
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    while True:
        # Run the optimizer
        _, loss_val, random_variable_val, obj_function_val, obj_grads_val, delta_val = sess.run(
            [train_op, loss, random_variable, obj_function, obj_grads, delta],
            feed_dict={X: batch_x, Y: batch_y})

        scheduler.loss_checker(loss_val)

        random_variable_val = random_variable_val - delta_val
        # random_variable_val = np.clip(random_variable_val, 0, 16)

        # # if it's a variable
        random_variable = tf.assign(random_variable, random_variable_val)

        loss_hist.append(sess.run(loss, feed_dict={X: batch_x, Y: batch_y}))
        acc_hist.append(sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y}))

        l1_m_max_lst.append(sess.run(max_l1_m, feed_dict={X: batch_x, Y: batch_y}))
        l1_a_max_lst.append(sess.run(max_l1_a, feed_dict={X: batch_x, Y: batch_y}))
        l2_m_max_lst.append(sess.run(max_l2_m, feed_dict={X: batch_x, Y: batch_y}))
        l2_a_max_lst.append(sess.run(max_l2_a, feed_dict={X: batch_x, Y: batch_y}))

        step += 1

        '''inserted code'''
        scheduler.check_time()
        '''inserted code'''
