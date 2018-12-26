import numpy as np
import tensorflow as tf


class Model():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def get_shape(self):
        '''Return the input shape of the model.
           RETURN: Returns the shape of the model input image. [ W, H, C ]
        '''
        pass
    
    def inference(self, x):
        '''Execute model train
           RETURN: output, keep_prob
        '''
        pass


    def _print(self, tensor):
        '''Print tensor name and shape.'''
        print(tensor.op.name, ' ', tensor.get_shape().as_list())

    
    def conv2d(self, x, W):
        '''conv2d returns a 2d convolution layer with full stride.'''
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(self, x):
        '''max_pool_2x2 downsamples a feature map by 2X.'''
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')


    def weight_variable(self, shape):
        '''weight_variable generates a weight variable of a given shape.'''
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self, shape):
        '''bias_variable generates a bias variable of a given shape.'''
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d_layer(self, name, input, shape, use_pool=True):
        '''build conv2d layer.'''
        with tf.name_scope(name):
            W_conv = self.weight_variable(shape)
            b_conv = self.bias_variable([shape[2]])
            h_conv = tf.nn.relu(self.conv2d(input, W_conv) + b_conv)

            if use_pool:
                with tf.name_scope('pool'):
                    h_conv = self.max_pool_2x2(h_conv)

            self._print(h_conv)
        return h_conv 
    
    def fc_layer(self, name, input, shape):
        '''build full connection layer'''
        with tf.name_scope(name):
            W_fc = self.weight_variable(shape)
            b_fc = self.bias_variable([shape[1]])

            h_pool2_flat = tf.reshape(h_pool3, [-1, 8*8*128])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)