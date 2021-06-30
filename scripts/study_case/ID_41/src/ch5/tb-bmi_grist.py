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

obj_var = y
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

sess = tf.Session()
tw = tf.summary.FileWriter("log_dir", graph=sess.graph)

sess.run(tf.initialize_all_variables())

"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="ml_plactice.ch5.tb-bmi_grist")
obj_function = tf.reduce_min(tf.abs(obj_var))
obj_grads = tf.gradients(obj_function, x)[0]
rows = csv[1: 1 + 100]
x_pat = np.array(rows[["weight", "height"]])
y_ans = list(np.array(rows["label_pat"]))
max_val, min_val = np.max(x_pat), np.min(x_pat)
gradient_search.build(batch_size=100, min_val=min_val, max_val=max_val)
"""insert code"""

step = 0
while True:
    i = (step * 100) % 14000

    """inserted code"""
    monitor_vars = {'loss': cross_entropy, 'obj_function': obj_function, 'obj_grad': obj_grads}
    feed_dict = {x: x_pat, y_: y_ans}
    x_pat, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                           feed_dict=feed_dict, input_data=x_pat, )
    """inserted code"""

    _, loss = sess.run([train, cross_entropy], feed_dict=feed_dict)

    """inserted code"""
    new_rows = csv[1 + i: 1 + i + 100]
    new_x_pat = np.array(rows[["weight", "height"]])
    new_y_ans = list(np.array(rows["label_pat"]))
    new_data_dict = {'x': new_x_pat, 'y': new_y_ans}
    old_data_dict = {'x': x_pat, 'y': y_ans}
    x_pat, _ = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                               old_data_dict=old_data_dict,
                                               scores_rank=scores_rank)
    gradient_search.check_time()
    """inserted code"""

    if step % 500 == 0:
        cre = sess.run(cross_entropy, feed_dict=feed_dict)
        acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
        # print("step=", step, "cre=", cre, "acc=", acc)
    step += 1
