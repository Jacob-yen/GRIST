import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append("/data")

image_size = 28 * 28

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


w, n_input, n_z = {}, image_size, 2  # 20
n_hidden_recog_1, n_hidden_recog_2 = 500, 500
n_hidden_gener_1, n_hidden_gener_2 = 500, 500
w['w_recog'] = {
    'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
    'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
    'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
    'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
w['b_recog'] = {
    'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
    'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
    'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
    'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
w['w_gener'] = {
    'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
    'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
    'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
    'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
w['b_gener'] = {
    'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
    'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
    'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
    'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}

l_rate = 0.001
batch_size = 100

x = tf.placeholder(tf.float32, [None, n_input])
enc_layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, w["w_recog"]['h1']), w["b_recog"]['b1']))
enc_layer_2 = tf.nn.softplus(tf.add(tf.matmul(enc_layer_1, w["w_recog"]['h2']), w["b_recog"]['b2']))
z_mean = tf.add(tf.matmul(enc_layer_2, w["w_recog"]['out_mean']), w["b_recog"]['out_mean'])
z_log_sigma_sq = tf.add(tf.matmul(enc_layer_2, w["w_recog"]['out_log_sigma']), w["b_recog"]['out_log_sigma'])

# eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32)
eps = tf.placeholder(tf.float32, [None, n_z])

suspect_func = z_log_sigma_sq
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

dec_layer_1 = tf.nn.softplus(tf.add(tf.matmul(z, w["w_gener"]['h1']), w["b_gener"]['b1']))
dec_layer_2 = tf.nn.softplus(tf.add(tf.matmul(dec_layer_1, w["w_gener"]['h2']), w["b_gener"]['b2']))
x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(dec_layer_2, w["w_gener"]['out_mean']), w["b_gener"]['out_mean']))

#MUTATION#
reconstr_loss = -tf.reduce_sum(x * tf.log(x_reconstr_mean + 1e-8) + (1 - x) * tf.log(1 - x_reconstr_mean + 1e-8), 1)

latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(reconstr_loss + latent_loss)
reconstr_loss_mean = tf.reduce_mean(reconstr_loss)
latent_loss_mean = tf.reduce_mean(latent_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)

show_step = [20]
xs_show, _ = mnist.test.next_batch(5000)


def train(sess, batch_size=100, training_epochs=10):
    """insert code"""
    from scripts.utils.tf_utils import GradientSearcher
    gradient_search = GradientSearcher(name="ch10_04_03_Pic_10_05_exp_grist")
    obj_function = -1 * tf.reduce_max(suspect_func)
    obj_grads = tf.gradients(obj_function, x)[0]
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    max_val, min_val = np.max(batch_xs), np.min(batch_xs)
    gradient_search.build(batch_size=batch_size, min_val=min_val, max_val=max_val)
    """insert code"""
    while True:
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):

            """inserted code"""
            monitor_vars = {'loss': cost, 'obj_function': obj_function, 'obj_grad': obj_grads}
            feed_dict = {x: batch_xs, eps: np.random.normal(loc=0.0, scale=1.0, size=(batch_size, n_z))}
            batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                      feed_dict=feed_dict, input_data=batch_xs, )
            """inserted code"""

            _, loss_val, r_loss, l_loss = sess.run((optimizer, cost, reconstr_loss_mean, latent_loss_mean),
                                                   feed_dict=feed_dict)

            """inserted code"""
            new_batch_xs, new_batch_ys = mnist.train.next_batch(batch_size)
            new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
            old_data_dict = {'x': batch_xs, 'y': batch_ys}
            batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                                 old_data_dict=old_data_dict,
                                                                 scores_rank=scores_rank)
            gradient_search.check_time()
            """inserted code"""


init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
train(sess, training_epochs=200, batch_size=batch_size)

sess.close()
