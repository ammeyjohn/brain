# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from dataset import load_data

FLAGS = None

def main(_):

  # Create the model
  x = tf.placeholder(tf.float32, [None, 64, 64, 3])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 5])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  iterator = load_data('./datasets/train-00000-of-00001.tfrecord', [64, 64])
  train_images, train_labels = iterator.get_next()

  step_count = 0
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, 1000):
      sess.run(iterator.initializer)
      while True:
        try:
          step_count += 1
          batch_images, batch_labels = \
            sess.run([train_images, train_labels])
          if step_count % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
              x: batch_images, y_: batch_labels, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (step_count, train_accuracy))
          train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})
        except tf.errors.OutOfRangeError:
          break


    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='./data/mnist',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)