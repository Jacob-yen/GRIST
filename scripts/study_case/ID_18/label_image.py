#!/usr/bin/env python

"""
This script will classify an image using your retrained model.
Usage: ./label_image.py ./path/to/image.jpg ./model-dir

./model-dir should contain the retrain_labels.txt and retrained_graph.pb files.

"""

import tensorflow as tf
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Path of image you want to classify.
image_path = sys.argv[1]

#Directory containing graph file and labels file
base = sys.argv[2]

base = os.path.join(base, '')

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_list = [line.rstrip() for line
                in tf.gfile.GFile(base + "retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile(base + "retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of predictions by confidence
    sorted_nodes = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node in sorted_nodes:
        label = label_list[node]
        score = predictions[0][node]
        print('%s (score = %.5f)' % (label, score))
