from os import path
import numpy as np
import tensorflow as tf
from PIL import Image

from model import deepnn

base_dir = "F:\\Workspace\\image_retraining\\classified\\invalid"

im = Image.open(path.join(base_dir, '00008R.png'))
im = im.resize([28, 28])

x = tf.placeholder(tf.float32, [None, 28, 28, 3])
y_conv, keep_prob = deepnn(x, 5)


saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # read the previously saved network.
    saver.restore(sess, './model/' + 'model')

    output = sess.run(y_conv, feed_dict={ x: [np.array(im)], keep_prob: 1.0 })
    print(output, np.argmax(output, 1))