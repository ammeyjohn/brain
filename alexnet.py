import numpy as np
import tensorflow as tf

from model import Model


class AlexNet(Model):

    def __init__(self, num_classes):
        super().__init__(num_classes)
        print('Using model Alexnet.')

    def get_shape(self):
        return [227, 227]

    def inference(self, x):

        # First convolutional layer
        # output=55*55*96
        h_conv1, _, _ = self.conv('conv1', x, [11, 11, 3, 96], stride=4, padding='VALID')        

        # First normalize layer
        h_norm1 = self.norm('norm1', h_conv1, lsize=5)

        # First pooling layer
        # output=27*27*96
        h_pool1 = self.max_pool('pool1', h_norm1, ksize=3, stride=2, padding='VALID')


        # Second convolutional layer
        # output=27*27*256
        h_conv2, _, _ = self.conv('conv2', h_pool1, [5, 5, 96, 256])

        # Second normalize layer
        h_norm2 = self.norm('norm2', h_conv2, lsize=5)

        # Second pooling layer
        # output=13*13*256
        h_pool2 = self.max_pool('pool2', h_norm2, ksize=3, stride=2, padding='VALID')

        # output=13*13*256
        h_conv3, _, _ = self.conv('conv3', h_pool2, [3, 3, 256, 384])

        # output=13*13*256
        h_conv4, _, _ = self.conv('conv4', h_conv3, [3, 3, 384, 384])

        # output=13*13*256
        h_conv5, _, _ = self.conv('conv5', h_conv4, [3, 3, 384, 256])
        h_pool5 = self.max_pool('pool5', h_conv5, ksize=3, stride=2, padding='VALID')

        # fc1
        # output=9216*4096
        w, h, c = h_pool5.get_shape().as_list()[-3:]         
        h_pool5_flat = tf.reshape(h_pool5, [-1, w*h*c])
        h_fc1, _, _ = self.fc('fc1', h_pool5_flat, [w*h*c, 4096])

        # dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = self.dropout('dropout1', h_fc1, keep_prob)

        # fc2
        # output=4096*4096
        h_fc2, _, _ = self.fc('fc2', h_fc1_drop, [4096, 4096])

        # dropout
        h_fc2_drop = self.dropout('dropout2', h_fc2, keep_prob)

        # fc3
        with tf.name_scope('fc3'):
            W_fc3 = self.weight_variable([4096, self.num_classes])
            b_fc3 = self.bias_variable([self.num_classes])

            y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

        return y_conv, keep_prob
        