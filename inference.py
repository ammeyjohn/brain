from os import path
import numpy as np
import tensorflow as tf
from PIL import Image

from model import deepnn

labels = []
with open('./images/meters/labels.txt', 'r') as f:
    for line in f.readlines():
        labels.append(line.strip())
num_classes = len(labels) + 1
labels = dict(zip(range(num_classes-1), labels))

base_dir = "./images/meters/test/others"

im = Image.open(path.join(base_dir, '38913314_13668eecefd15c7afe7a4d7d3104105b.jpg'))
im = im.resize([28, 28])

x = tf.placeholder(tf.float32, [None, 28, 28, 3])
y_conv, keep_prob = deepnn(x, num_classes)


saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # read the previously saved network.
    saver.restore(sess, './model/meters.ckpt-4000')

    output = sess.run(y_conv, feed_dict={ x: [np.array(im)], keep_prob: 1.0 })
    label_index = np.argmax(output, 1)
    label = labels[label_index[0]]
    print(output, label_index, label)