from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import numpy as np
import sys
sys.path.append("/data")
import tensorflow as tf



def regularizer(x):
    a = tf.abs(x)
    a_clipped = tf.clip_by_value(a, -1, +1)
    y = 2 * (a - a_clipped)

    return y

from datetime import datetime
s1 = datetime.now()

""" Network Parameters """
n_hidden = 128
n_inputs = 784
n_classes = 10

""" TF Graph Input """
X = tf.placeholder(tf.float32, [None, n_inputs], name="features")
Y = tf.placeholder(tf.float32, [None, n_classes], name="labels")

weights = {
    'l1': tf.get_variable(name="w1",
                          initializer=tf.random_normal([n_inputs, n_hidden])),
    'lo': tf.get_variable(name="wout",
                          initializer=tf.random_normal([n_hidden, n_classes]))
}

biases = {
    'l1': tf.get_variable(name="b1",
                          initializer=tf.random_normal([n_hidden])),
    'lo': tf.get_variable(name="bout",
                          initializer=tf.random_normal([n_classes]))
}

""" SC Model Parameters """
# sc_std = tf.constant([0.0001])
# sc_std = tf.get_variable(name="std", initializer=tf.random_uniform([1],minval=0,maxval=1))

l1_matmul = {
    's_inp': tf.constant(1.0, shape=[1, n_inputs]),  # Input scaling factor
    'upscale': True,
    'gain': tf.get_variable(name="G_matmul_1",
                            initializer=tf.random_uniform([n_hidden], minval=0, maxval=16))
}

l1_matadd = {
    'upscale': True
}

lo_matmul = {
    'upscale': True,
    'gain': tf.get_variable(name="G_matmul_2",
                            initializer=tf.random_uniform([n_classes], minval=0, maxval=16))
}

lo_matadd = {
    'upscale': True
}

""" Helper Functions """
obj_var = None


def sc_matmul(x, w, s_in, upscale=False, gain=None):
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

    rescale_ratio = tf.div(s_in, max_scale, name="mydiv1")
    x_rescaled = tf.multiply(x, rescale_ratio)

    # Calculate the down-scale coefficients in terms of the weights
    s = tf.reduce_sum(tf.abs(w), axis=0)
    S = tf.reshape(tf.tile(s, [tf.shape(x)[0]]),
                   shape=[tf.shape(x)[0], tf.shape(w)[1]])

    # Matrix mutiplication using re-scaled feature data
    y = tf.matmul(x_rescaled, w)
    z = tf.div(y, S, name="mydiv2")

    curr_scale = tf.multiply(max_scale, S)

    # Apply gain followed by saturation
    if (upscale):
        z_upscaled = tf.multiply(z, gain)
        #MUTATION#
        new_scale = tf.div(curr_scale, gain, name="mydiv3")
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
    global obj_var
    if obj_var is None:
        obj_var = s_in
    b_scaled = tf.div(b, s_in, name="mydiv4")

    # Scaled addition using down-scaled bias terms
    y = tf.div(tf.add(x, b_scaled), 2, name="mydiv5")

    curr_scale = tf.multiply(s_in, 2)

    # Apply gain followed by saturation
    if (upscale):
        z_upscaled = tf.multiply(y, 2)
        out_scale = tf.div(curr_scale, 2, name="mydiv6")
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
objective = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

# Regularization Losses
reg_scale = tf.constant(0.25)
reg_loss_1 = tf.reduce_sum(regularizer(weights['l1']))
reg_loss_2 = tf.reduce_sum(regularizer(weights['lo']))
reg_loss_3 = tf.reduce_sum(regularizer(biases['l1']))
reg_loss_4 = tf.reduce_sum(regularizer(biases['lo']))

loss = objective + reg_scale * tf.reduce_sum(reg_loss_1 + reg_loss_2 + reg_loss_3 + reg_loss_4)

""" Optimizer """
# Parameters
learning_rate = 0.1
n_epochs = 50000
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
    # tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'SC_DNN.pbtxt')
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
    step = 0

    """insert code"""
    from scripts.utils.tf_utils import GradientSearcher
    gradient_search = GradientSearcher(name="SC_DNN.sc_train_creg_div4_grist")
    obj_function = tf.reduce_min(tf.abs(obj_var))
    obj_grads = tf.gradients(obj_function, X)[0]
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    max_val, min_val = np.max(batch_x), np.min(batch_x)
    gradient_search.build(batch_size=batch_size, min_val=min_val, max_val=max_val)
    """insert code"""

    while True:
        """inserted code"""
        monitor_vars = {'loss': loss, 'obj_function': obj_function, 'obj_grad': obj_grads}
        feed_dict = {X: batch_x, Y: batch_y}
        batch_x, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                 feed_dict=feed_dict, input_data=batch_x)
        """inserted code"""

        loss_val = sess.run(loss, feed_dict=feed_dict)

        sess.run(train_op, feed_dict=feed_dict)

        loss_hist.append(loss_val)
        acc_hist.append(sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y}))

        l1_gain.append(l1_matmul['gain'].eval())
        lo_gain.append(lo_matmul['gain'].eval())

        l1_m_max_lst.append(sess.run(max_l1_m, feed_dict={X: batch_x, Y: batch_y}))
        l1_a_max_lst.append(sess.run(max_l1_a, feed_dict={X: batch_x, Y: batch_y}))
        l2_m_max_lst.append(sess.run(max_l2_m, feed_dict={X: batch_x, Y: batch_y}))
        l2_a_max_lst.append(sess.run(max_l2_a, feed_dict={X: batch_x, Y: batch_y}))

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            b_loss, b_acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})

        step += 1

        """inserted code"""
        new_batch_xs, new_batch_ys = mnist.train.next_batch(batch_size)
        new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
        old_data_dict = {'x': batch_x, 'y': batch_y}
        batch_x, batch_y = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                           old_data_dict=old_data_dict,
                                                           scores_rank=scores_rank)
        gradient_search.check_time()
        """inserted code"""
