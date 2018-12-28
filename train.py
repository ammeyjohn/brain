import sys, os
import time
from datetime import datetime
from math import ceil

import tensorflow as tf

from dataset import load_data
from model import deepnn


tf.app.flags.DEFINE_string('data_dir', './datasets', 
                           'Directory for storing train tfrecord files.')
tf.app.flags.DEFINE_string('validate_dir', './datasets', 
                           'Directory for storing validation tfrecord files.')                           
tf.app.flags.DEFINE_string('model_dir', './model', 
                           'Directory for save train model.')
tf.app.flags.DEFINE_string('train_dir', './tmp', 
                           'Directory for save train summary.')

tf.app.flags.DEFINE_integer('train_sample_count', 4000, 'Count of train images count.')
tf.app.flags.DEFINE_integer('num_classes', 5, 'Defines number of classes.')
tf.app.flags.DEFINE_integer('max_steps', 100000, 'Defines max train step.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Defines batch size.')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Defines learning rate')

tf.app.flags.DEFINE_integer('display_step', 100, 'Print train info every N steps.')

FLAGS = tf.app.flags.FLAGS

channels = 3
# 0 for background images
num_classes = FLAGS.num_classes + 1
image_shape = [100, 100]

def main(_):  

    # Create the model
    x = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], channels])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, num_classes])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x, num_classes)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,
                                                                logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss", cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar("accuracy", accuracy)

    summary_op = tf.summary.merge_all()

    # Load train data
    train_dataset_path = os.path.join(FLAGS.data_dir, 'train-00000-of-00001')
    train_iter = load_data(train_dataset_path, image_shape, use_one_hot=True, num_classes=num_classes)
    train_images, train_labels = train_iter.get_next()

    saver = tf.train.Saver()
    if not os.path.isdir(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)  
	
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        
        sess.run(train_iter.initializer)
        for step in range(FLAGS.max_steps):
            try:
                batch_images, batch_labels = sess.run([train_images, train_labels])
            except tf.errors.OutOfRangeError:
                sess.run(train_iter.initializer)
                batch_images, batch_labels = sess.run([train_images, train_labels])

            start_time = time.time()
            sess.run(train_step, feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})                
            duration = time.time() - start_time

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                loss = sess.run(cross_entropy, 
                    feed_dict={x: batch_images, y_: batch_labels, keep_prob: 1.0})  
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss,  examples_per_sec, duration))

            if step % 100 == 0:
                summary_str = sess.run(summary_op, 
                    feed_dict={x: batch_images, y_: batch_labels, keep_prob: 1.0})
                summary_writer.add_summary(summary_str, step)   

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.model_dir, 'meters.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)                             

        summary_writer.close()



if __name__ == '__main__': 
    tf.app.run(main=main)