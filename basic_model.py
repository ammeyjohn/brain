import numpy as np
import tensorflow as tf

from model import Model


class BasicModel(Model):

    def __init__(self, num_classes):
        super().__init__(num_classes)

    def get_shape(self):
        return [28, 28]

    def inference(self, x):

        # First convolutional layer
        h_conv1, _, _ = self.conv('conv1', x, [5, 5, 3, 32])

        # First pooling layer - downsamples by 2X.
        h_pool1 = self.max_pool_2x2('pool1', h_conv1)

        # Second convolutional layer
        h_conv2, _, _ = self.conv('conv2', h_pool1, [5, 5, 32, 64])

        # Second pooling layer - downsamples by 2X.
        h_pool2 = self.max_pool_2x2('pool2', h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features. 
        w, h, c = h_pool2.get_shape().as_list()[-3:]         
        h_pool2_flat = tf.reshape(h_pool2, [-1, w*h*c])
        h_fc1, _, _ = self.fc('fc1', h_pool2_flat, [w*h*c, 1024])

        # # Dropout - controls the complexity of the model, prevents co-adaptation of features.
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = self.dropout('dropout1', h_fc1, keep_prob)

        # Map the 1024 features to N classes, one for each digit
        with tf.name_scope('fc2'):
            W_fc2 = self.weight_variable([1024, self.num_classes])
            b_fc2 = self.bias_variable([self.num_classes])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return y_conv, keep_prob   
        