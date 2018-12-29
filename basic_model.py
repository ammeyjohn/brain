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
        h_conv1, _, _ = super().conv('conv1', x, [5, 5, 3, 32])

        # First pooling layer - downsamples by 2X.
        h_pool1 = super().max_pool_2x2('pool1', h_conv1)

        # Second convolutional layer
        h_conv2, _, _ = super().conv('conv2', h_pool1, [5, 5, 32, 64])

        # Second pooling layer - downsamples by 2X.
        h_pool2 = super().max_pool_2x2('pool2', h_conv2)     

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features. 
        w, h, c = h_pool2.get_shape().as_list()[-3:]         
        h_pool3_flat = tf.reshape(h_pool2, [-1, w * h * c])
        h_fc1, _, _ = super().fc('fc1', h_pool3_flat, [w * h * c, 1024])

        # Dropout - controls the complexity of the model, prevents co-adaptation of features.
        h_fc1_drop, keep_prob = super().dropout('fc1', h_fc1)

        # Map the 1024 features to 5 classes, one for each digit
        y_conv, _, _ = super().fc('fc2', h_fc1_drop, [1024, self.num_classes])

        return y_conv, keep_prob   

    def train(self, logits, labels, learning_rate=1e-4, name='train'):
        with tf.name_scope(name):
            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
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
        