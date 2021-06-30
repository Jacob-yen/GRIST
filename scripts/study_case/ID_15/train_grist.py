import time
import numpy as np
import tensorflow as tf
import sys
sys.path.append("/data")
from scripts.study_case.ID_15.data import *
from scripts.study_case.ID_15.model import *
from scripts.study_case.ID_15.utils import *

np.random.seed(0)

# Data
tf.app.flags.DEFINE_string('input', '/data/scripts/study_case/ID_15/data/gridworld_8.mat', 'Path to data')
tf.app.flags.DEFINE_integer('imsize', 8, 'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs', 100, 'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k', 10, 'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i', 2, 'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h', 150, 'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q', 10, 'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize', 12, 'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 10,
                            'Number of state inputs for each sample (real number, technically is k+1)')
tf.app.flags.DEFINE_boolean('untied_weights', False, 'Untie weights of VI network')
# Misc.
tf.app.flags.DEFINE_integer('display_step', 1, 'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log', False, 'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir', 'log/', 'Directory to store tensorboard summary')

config = tf.app.flags.FLAGS

# symbolic input image tensor where typically first channel is image, second is the reward prior
X = tf.placeholder(tf.float32, name="X", shape=[None, config.imsize, config.imsize, config.ch_i])
# symbolic input batches of vertical positions
S1 = tf.placeholder(tf.int32, name="S1", shape=[None, config.statebatchsize])
# symbolic input batches of horizontal positions
S2 = tf.placeholder(tf.int32, name="S2", shape=[None, config.statebatchsize])
y = tf.placeholder(tf.int32, name="y", shape=[None])

# Construct model (Value Iteration Network)
if (config.untied_weights):
    nn = VI_Untied_Block(X, S1, S2, config)
else:
    nn = VI_Block(X, S1, S2, config)

# Define loss and optimizer
dim = tf.shape(y)[0]
cost_idx = tf.concat([tf.reshape(tf.range(dim), [dim, 1]), tf.reshape(y, [dim, 1])], 1)

''' WG Insert Start '''
global_obj_var = nn
''' WG Insert End '''
#MUTATION#
cost = -tf.reduce_mean(tf.gather_nd(tf.log(nn), [cost_idx]))
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr, epsilon=1e-6, centered=True).minimize(cost)

# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast(tf.not_equal(cp, y), dtype=tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(input=config.input,
                                                                                        imsize=config.imsize)
"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="tensorflow_value_iteration_networks_v1_grist")
obj_function = tf.reduce_min(tf.abs(global_obj_var))
obj_grads = tf.gradients(obj_function, X)[0]
batch_xs = Xtrain[0:config.batchsize]
batch_ys = ytrain[0:config.batchsize * config.statebatchsize]
max_val, min_val = np.max(batch_xs), np.min(batch_xs)
gradient_search.build(batch_size=config.batchsize, min_val=min_val, max_val=max_val)
"""insert code"""

flag = 0

with tf.Session() as sess:
    if config.log:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
    sess.run(init)

    batch_size = config.batchsize
    print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))

    i = 0
    j = 0
    while True:
        for i in range(0, Xtrain.shape[0], batch_size):
            j = i + batch_size
            if j <= Xtrain.shape[0]:
                """inserted code"""
                monitor_vars = {'loss': cost, 'obj_function': obj_function, 'obj_grad': obj_grads}
                feed_dict = {X: batch_xs, S1: S1train[i:j], S2: S2train[i:j], y: batch_ys}
                batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                          feed_dict=feed_dict, input_data=batch_xs,
                                                                          )
                """inserted code"""

                _, e_, c_ = sess.run([optimizer, err, cost], feed_dict=feed_dict)

                """inserted code"""
                new_batch_xs = Xtrain[j:j+batch_size]
                new_batch_ys = ytrain[j * config.statebatchsize:(j + batch_size) * config.statebatchsize]
                new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
                old_data_dict = {'x': batch_xs, 'y': batch_ys}
                batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                                     old_data_dict=old_data_dict,
                                                                     scores_rank=scores_rank)
                gradient_search.check_time()
                """inserted code"""
