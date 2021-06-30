import pandas as pd
import numpy as np
import tensorflow as tf
import sys

sys.path.append("/data")
csv = pd.read_csv("bmi.csv")
csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 100
bclass = {"thin": [1, 0, 0], "normal": [0, 1, 0], "fat": [0, 0, 1]}
csv["label_pat"] = csv["label"].apply(lambda x: np.array(bclass[x]))

test_csv = csv[15000:20000]
test_pat = test_csv[["weight", "height"]]
test_ans = list(test_csv["label_pat"])

x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.zeros([2, 3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

sess = tf.Session()
tw = tf.summary.FileWriter("log_dir", graph=sess.graph)
tf.train.write_graph(sess.graph_def, '/data/scripts/study_case/pbtxt_files', 'ml_plactice_tbbmi.pbtxt')

sess.run(tf.initialize_all_variables())

'''inserted code'''
from scripts.utils.tf_utils import TensorFlowScheduler
scheduler = TensorFlowScheduler(name="ml_plactice.ch5.tb-bmi")
'''inserted code'''

step = 0
while True:
    i = (step * 100) % 14000
    rows = csv[1 + i: 1 + i + 100]
    x_pat = rows[["weight", "height"]]
    y_ans = list(rows["label_pat"])
    fd = {x: x_pat, y_: y_ans}
    _, loss = sess.run([train, cross_entropy], feed_dict=fd)
    if step % 500 == 0:
        cre = sess.run(cross_entropy, feed_dict=fd)
        acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
        # print("step=", step, "cre=", cre, "acc=", acc)
    step += 1

    '''inserted code'''
    scheduler.loss_checker(loss)
    scheduler.check_time()
    '''inserted code'''
