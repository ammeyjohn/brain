import time

import numpy as np
import tensorflow as tf

from dataset import load_data
from model import inference


batch_size = 32
num_classes = 4


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

pred = inference(x, y, keep_prob)

learning_rate = 0.0001

# 定义损失函数和优化器
# 这里定义损失函数时调用tf.nn.softmax_cross_entropy_with_logits() 函数必须使用参数命名的方式来调用 (logits=pred, labels=y)不然会报错。
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)) 
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估函数
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

num_epochs = 10000

iterator = load_data('./datasets/train-00000-of-00001.tfrecord', \
                     image_shape=(227, 227), \
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
                sess.run(train_op, feed_dict={ x: batch_images, y: batch_labels, keep_prob: 0.8 })
                step_count += 1

                if step_count % 10 == 0:
                    loss, acc = sess.run([cost, accuracy], \
                        feed_dict={ x: batch_images, y: batch_labels, keep_prob: 1.0 })
                    time_cost = time.time() - start_time
                    print('step=%d, loss=%f, acc=%f (%f sec/batch)' % (step_count, loss, acc, time_cost))
                    start_time = time.time()
            except tf.errors.OutOfRangeError:
                break


        
    