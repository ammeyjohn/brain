import numpy as np
import tensorflow as tf


def _print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

# 卷积层
def conv2d(name, input, w, b, stride, padding='SAME'):
    # 测试
    x = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)    


# 最大下采样
def max_pool(name, input, k, stride):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)


# 归一化操作 ToDo 正则方式待修改
def norm(name, input, size=4):
    return tf.nn.lrn(input, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

num_classes = 6

# 定义所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 512])),
    'out': tf.Variable(tf.random_normal([512, num_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([512])),
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
