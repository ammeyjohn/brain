import numpy as np
import tensorflow as tf


class Model():

    def __init__(self, num_classes):        
        self.num_classes = num_classes

    def conv(self, name, x, shape):
        """build a full function conv2d layer."""
        with tf.name_scope(name):
            W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=shape[-1:]))
            conv2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
            h_conv = tf.nn.relu(conv2d + b)
            self.print_tensor(h_conv)
            return h_conv, W, b

    def fc(self, name, x, shape):
        """build a full connection layer."""
        with tf.name_scope(name):
            W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=shape[-1:]))
            h_fc = tf.nn.relu(tf.matmul(x, W) + b)
            self.print_tensor(h_fc)
            return h_fc, W, b

    def dropout(self, name, x, keep_prob=None):
        """return a dropout layer."""
        with tf.name_scope(name):
            if keep_prob is None:
                keep_prob = tf.placeholder(tf.float32)
            h_fc = tf.nn.dropout(x, keep_prob)
            self.print_tensor(h_fc)
            return h_fc, keep_prob

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, name, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        with tf.name_scope(name):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')    

    def weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def print_tensor(self, tensor):
        """print tensor name and shape."""
        print(tensor.op.name, '\t', tensor.get_shape().as_list())    

    def get_shape(self):
        raise NotImplementedError()

    def inference(self, x):
        raise NotImplementedError()

    def train(self, logits, labels, learning_rate=1e-4, name='train'):
        raise NotImplementedError()

    def validate(self, logits, labels, name='val'):
        raise NotImplementedError()