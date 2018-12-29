import sys, os
import time
from datetime import datetime
from math import ceil

import tensorflow as tf

from dataset import load_data
from basic_model import BasicModel


tf.app.flags.DEFINE_string('data_dir', './datasets', 
                           'Directory for storing train tfrecord files.')
tf.app.flags.DEFINE_string('validate_dir', './datasets', 
                           'Directory for storing validation tfrecord files.')                           
tf.app.flags.DEFINE_string('model_dir', './model', 
                           'Directory for save train model.')
tf.app.flags.DEFINE_string('train_dir', './tmp', 
                           'Directory for save train summary.')

tf.app.flags.DEFINE_integer('num_classes', None, 'Defines number of classes.')
tf.app.flags.DEFINE_integer('max_steps', 100000, 'Defines max train step.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Defines batch size.')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Defines learning rate')

FLAGS = tf.app.flags.FLAGS

model_name = 'meters.ckpt'
channels = 3
# 0 for background images
num_classes = FLAGS.num_classes + 1
model = BasicModel(num_classes)
image_shape = model.get_shape()

def main(_):  

    # Create the model
    x = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], channels])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, num_classes])

    # Build the graph for the deep net
    y_conv, keep_prob = model.inference(x)

    # Create train step
    train_op, cross_entropy = model.train(y_conv, y_)

    summary_op = tf.summary.merge_all()

    # Load train data
    train_dataset_path = os.path.join(FLAGS.data_dir, 'train-00000-of-00001')
    train_iter = load_data(train_dataset_path, image_shape, FLAGS.batch_size, 
        use_one_hot=True, num_classes=num_classes)
    train_images, train_labels = train_iter.get_next()
    
    if not os.path.isdir(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    if not os.path.isdir(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)    

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step0 = 0
        # Restore model checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            step0 = int(ckpt.model_checkpoint_path.split('-')[-1])
            print('Restore model from', ckpt.model_checkpoint_path, ', Start step =', step0)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
                
        sess.run(train_iter.initializer)
        for step in range(step0, FLAGS.max_steps):
            try:
                batch_images, batch_labels = sess.run([train_images, train_labels])
            except tf.errors.OutOfRangeError:
                sess.run(train_iter.initializer)
                batch_images, batch_labels = sess.run([train_images, train_labels])

            start_time = time.time()
            _, loss = sess.run([train_op, cross_entropy], 
                feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})                
            print(loss)
            duration = time.time() - start_time

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                loss = sess.run(cross_entropy, 
                    feed_dict={x: batch_images, y_: batch_labels, keep_prob: 1.0})  
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss, examples_per_sec, duration))

            if step % 100 == 0:
                summary_str = sess.run(summary_op, 
                    feed_dict={x: batch_images, y_: batch_labels, keep_prob: 1.0})
                summary_writer.add_summary(summary_str, step)   

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.model_dir, model_name)
                saver.save(sess, checkpoint_path, global_step=step)                             

        summary_writer.close()



if __name__ == '__main__': 
    tf.app.run(main=main)