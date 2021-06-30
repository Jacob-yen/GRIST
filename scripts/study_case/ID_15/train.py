import time
import numpy as np
import sys
sys.path.append("/data")
import tensorflow as tf
from scripts.study_case.ID_15.data import *
from scripts.study_case.ID_15.model import *
from scripts.study_case.ID_15.utils import *

np.random.seed(0)

# Data
tf.app.flags.DEFINE_string('input', '/data/scripts/study_case/ID_15/data/gridworld_8.mat',
                           'Path to data')
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
#MUTATION#
cost = -tf.reduce_mean(tf.gather_nd(tf.log(nn), [cost_idx]))
# cost = -tf.reduce_mean(tf.log(nn))
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr, epsilon=1e-6, centered=True).minimize(cost)

obj_function = tf.reduce_min(tf.abs(nn))

# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast(tf.not_equal(cp, y), dtype=tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(input=config.input,
                                                                                        imsize=config.imsize)

# Launch the graph
with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'tensorflow_value_iteration_networks.pbtxt')
    if config.log:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
    sess.run(init)

    batch_size = config.batchsize
    print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))

    '''inserted code'''
    from scripts.utils.tf_utils import TensorFlowScheduler
    scheduler = TensorFlowScheduler(name="tensorflow_value_iteration_networks_v1")
    '''inserted code'''

    epoch = 1
    while True:
        for i in range(0, Xtrain.shape[0], batch_size):
            j = i + batch_size
            if j <= Xtrain.shape[0]:
                fd = {X: Xtrain[i:j], S1: S1train[i:j], S2: S2train[i:j],
                      y: ytrain[i * config.statebatchsize:j * config.statebatchsize]}
                _, e_, loss, obj_function_val = sess.run([optimizer, err, cost, obj_function], feed_dict=fd)
                x_val, S1_val, S2_val, y_val = sess.run([X, S1, S2, y], feed_dict=fd)
                print(f"X ({np.max(x_val)}, {np.min(x_val)}), S1 ({np.max(S1_val)}, {np.min(S1_val)}), S2 ({np.max(S2_val)}, {np.min(S2_val)}), y ({np.max(y_val)}, {np.min(y_val)})")

                '''inserted code'''
                scheduler.loss_checker(loss)
                '''inserted code'''

                '''inserted code'''
                scheduler.check_time()
                '''inserted code'''
