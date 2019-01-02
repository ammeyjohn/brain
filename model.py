import numpy as np
import tensorflow as tf


class Model():

    def __init__(self, num_classes):        
        self.num_classes = num_classes
        print(self.num_classes)

    def conv(self, name, x, shape, stride=1, padding='SAME'):
        """build a full function conv2d layer."""
        with tf.name_scope(name):
            W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=shape[-1:]))
            h_conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
            h_conv = tf.nn.relu(h_conv + b)
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

    def dropout(self, name, x, keep_prob):
        """return a dropout layer."""
        with tf.name_scope(name):
            h_fc = tf.nn.dropout(x, keep_prob)
            self.print_tensor(h_fc)
            return h_fc

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool(self, name, x, ksize=2, stride=2, padding='SAME'):
        """max_pool_2x2 downsamples a feature map by 2X."""
        with tf.name_scope(name):
            h_pool = tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                                       strides=[1, stride, stride, 1], padding=padding)
            self.print_tensor(h_pool)                       
            return h_pool

    def max_pool_2x2(self, name, x, padding='SAME'):
        """max_pool_2x2 downsamples a feature map by 2X."""
        with tf.name_scope(name):
            h_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1], padding=padding)
            self.print_tensor(h_pool)                       
            return h_pool

    def norm(self, name, x, lsize):
        """Create a local response normalization layer."""
        with tf.name_scope(name):
            return tf.nn.lrn(x, depth_radius=lsize)

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
        with tf.name_scope(name):
            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            cross_entropy = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', cross_entropy)

            with tf.name_scope('adam_optimizer'):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        
        return train_step, cross_entropy

    def validate(self, logits, labels, name='val'):
        with tf.name_scope(name):
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)
                accuracy = tf.reduce_mean(correct_prediction)
                tf.summary.scalar('accuracy', accuracy)        
        return accuracy, correct_prediction