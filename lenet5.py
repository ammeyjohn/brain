import numpy as np
import tensorflow as tf

from model import Model


class Lenet5(Model):

    def __init__(self, num_classes):
        super().__init__(num_classes)
        print('Using model Lenet5.')

    def get_shape(self):
        return [32, 32]

    def inference(self, x):

        # First convolutional layer
        # output=28*28*6
        h_conv1, _, _ = self.conv('conv1', x, [5, 5, 3, 6], padding='VALID')

        # First pooling layer - downsamples by 2X.
        # output=14*14*6
        h_pool1 = self.max_pool_2x2('pool1', h_conv1)

        # Second convolutional layer
        # output=10*10*16
        h_conv2, _, _ = self.conv('conv2', h_pool1, [5, 5, 6, 16], padding='VALID')

        # Second pooling layer - downsamples by 2X.
        # output=5*5*16
        h_pool2 = self.max_pool_2x2('pool2', h_conv2)

        # Reshape
        w, h, c = h_pool2.get_shape().as_list()[-3:]         
        h_pool2_flat = tf.reshape(h_pool2, [-1, w*h*c])

        # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
        # is down to 5*5*16 feature maps -- maps this to 120 features.         
        h_fc1, _, _ = self.fc('fc1', h_pool2_flat, [w*h*c, 120])

        # Fully connected layer 2
        h_fc2, _, _ = self.fc('fc2', h_fc1, [120, 84])

        # Map the 1024 features to N classes, one for each digit
        with tf.name_scope('fc3'):
            W_fc3 = self.weight_variable([84, self.num_classes])
            b_fc3 = self.bias_variable([self.num_classes])

            y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
            self.print_tensor(y_conv)

        return y_conv, tf.placeholder(tf.float32)
        