import tensorflow as tf

from model import Model

class BrainModel(Model):

    def __init__(self, num_classes):
        Model.__init__(self, num_classes)

    def get_shape(self):
        return [ 64, 64, 3 ]

    def inference(self, x):
        """deepnn builds the graph for a deep net for classifying digits.
        Args:
            x: an input tensor with the dimensions (N_examples, 784), where 784 is the
            number of pixels in a standard MNIST image.
        Returns:
            A tuple (y, keep_prob). y is a tensor of shape (N_examples, M_labels), with values
            equal to the logits of classifying the digit into one of 10 classes (the
            digits 0-9). keep_prob is a scalar placeholder for the probability of
            dropout.
        """
        # First convolutional layer - maps one grayscale image to 32 feature maps.
        super(Model, self).conv_layer('conv1', x, [5, 5, 3, 32], True)

        # Second convolutional layer -- maps 32 feature maps to 64.
        super(Model, self).conv_layer('conv2', x, [5, 5, 32, 64], True)

        # Third convolutional layer -- maps 32 feature maps to 128.
        super(Model, self).conv_layer('conv3', x, [5, 5, 64, 128], True)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([8 * 8 * 128, 2048])
            b_fc1 = bias_variable([2048])

            h_pool2_flat = tf.reshape(h_pool3, [-1, 8*8*128])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            _print(h_fc1_drop)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([2048, 5])
            b_fc2 = bias_variable([5])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            _print(y_conv)

        return y_conv, keep_prob