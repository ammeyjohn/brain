import os
import numpy as np
import tensorflow as tf

def _parse(example, image_shape, use_one_hot, num_classes):
    feature = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/class/text': tf.FixedLenFeature([], tf.string)
    }

    parsed_example = tf.parse_single_example(serialized=example, features=feature)

    image_raw = parsed_example['image/encoded']
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize_images(image, image_shape) / 255   
    image = tf.cast(image, tf.float32)

    label = parsed_example['image/class/label']
    if use_one_hot:
        label = tf.one_hot(label, depth=num_classes, on_value=1.0, off_value=0.0)

    return image, label

def load_data(data_files, image_shape, batch_size=32, num_epochs=1, use_one_hot=True, num_classes=None, num_threads=1):

    if use_one_hot:
        assert(num_classes is not None, 'num_classes should not be None in one hot mode.')
        assert(num_classes > 0, 'num_classes should not be large than 0.')

    dataset = tf.data.TFRecordDataset(filenames=data_files, num_parallel_reads=num_threads)
    dataset = dataset.map(lambda e: _parse(e, image_shape, use_one_hot, num_classes))
    dataset = dataset.shuffle(buffer_size=batch_size*num_threads)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    return dataset.make_initializable_iterator()
