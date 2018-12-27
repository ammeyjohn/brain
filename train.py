import time

import numpy as np
import tensorflow as tf

from dataset import load_data

# from alexnet import get_shape, inference
from vgg16 import get_shape, inference

input_shape = get_shape()
print('input', '\t', input_shape)

batch_size = 32
num_channels = 3
num_classes = 5
learning_rate = 0.001
num_epochs = 10000

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], num_channels])
y = tf.placeholder(tf.int64, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

logits = inference(x, keep_prob, num_classes)

# 定义损失函数和优化器
# 这里定义损失函数时调用tf.nn.softmax_cross_entropy_with_logits() 函数必须使用参数命名的方式来调用 (logits=pred, labels=y)不然会报错。
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) 
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估函数
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

iterator = load_data('./datasets/train-00000-of-00001.tfrecord', \
                     image_shape=(input_shape[0], input_shape[1]), \
                     batch_size=batch_size)
train_images, train_labels = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    step_count = 0.0
    for i in range(num_epochs):
        sess.run(iterator.initializer)
        start_time = time.time()        
        while True:
            try:
                batch_images, batch_labels = \
                    sess.run([train_images, train_labels])
                sess.run(train_op, feed_dict={ x: batch_images, y: batch_labels, keep_prob: 0.5 })
                step_count += 1

                if step_count % 10 == 0:
                    loss, acc = sess.run([cost, accuracy], \
                        feed_dict={ x: batch_images, y: batch_labels, keep_prob: 1.0 })
                    time_cost = time.time() - start_time
                    print('step=%d, loss=%f, acc=%f (%f sec/batch)' % (step_count, loss, acc, time_cost))
                    start_time = time.time()
            except tf.errors.OutOfRangeError:
                break


        
    