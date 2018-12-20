import numpy as np
import tensorflow as tf


class Model():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def _print_activations(self, tensor):
        '''Print tensor name and shape.'''
        print(tensor.op.name, ' ', tensor.get_shape().as_list())

    def conv2d(self, input, w, b, stride, padding='SAME', name=None):
        '''convolution layer'''
        x = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x, name=name)    

    def max_pool(self, input, ksize, stride, padding='VALID', name=None):
        '''Max pooling'''
        return tf.nn.max_pool(input, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

# 定义所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 3, 48])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 48, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 192])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 192, 192])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 192, 128])),
    'wd1': tf.Variable(tf.random_normal([3*3*128, 2048])),
    'wd2': tf.Variable(tf.random_normal([2048, 2048])),
    'out': tf.Variable(tf.random_normal([2048, num_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([48])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([192])),
    'bc4': tf.Variable(tf.random_normal([192])),
    'bc5': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([2048])),
    'bd2': tf.Variable(tf.random_normal([2048])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def inference(images, labels, keep_prob):
    conv1 = conv2d('conv1', images, weights['wc1'], biases['bc1'], stride=4)
    pool1 = max_pool('pool1', conv1, k=3, stride=2)
    norm1 = norm('norm1', pool1, size=5)
    _print_activations(norm1)

    # 卷积层2
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'], stride=1, padding="VALID")
    pool2 = max_pool('pool2', conv2, k=3, stride=2)
    norm2 = norm('norm2', pool2, size=5)
    _print_activations(norm2)

    # 卷积层3 padding=1
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'], stride=1, padding="VALID")
    _print_activations(conv3)

    # 卷积层4 padding=1
    conv4 = conv2d('conv4', conv3, weights['wc4'], biases['bc4'], stride=1, padding="VALID")
    _print_activations(conv4)

    # 卷积层5 padding=1
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'], stride=1, padding="VALID")
    pool5 = max_pool('pool5', conv5, k=3, stride=2)
    _print_activations(pool5)

    # 全连接层1
    # 先把特征图转为向量
    fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1, name='fc1')
    # Dropout
    drop1 = tf.nn.dropout(fc1, keep_prob)
    _print_activations(drop1)

    # 全连接层2
    fc2 = tf.add(tf.matmul(drop1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2, name='fc2')
    # Dropout
    drop2 = tf.nn.dropout(fc2, keep_prob)
    _print_activations(drop2)

    # out
    out = tf.add(tf.matmul(drop2, weights['out']), biases['out'], name='logits')
    _print_activations(out)

    return out
