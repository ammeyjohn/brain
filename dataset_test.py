import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import load_data


batch_size = 8
iterator = load_data('./datasets/train-00000-of-00001.tfrecord', batch_size=batch_size)

with tf.Session() as sess:
    sess.run(iterator.initializer)

    for _ in [0, 1]:
        train_images, train_labels = iterator.get_next()
        batch_images, batch_labels = \
            sess.run([train_images, train_labels])
        print(batch_images, batch_labels)

        for i in range(batch_size):
            axs = plt.subplot(5, batch_size//5+1, i+1)
            axs.set_title(str(batch_labels[i]))
            plt.imshow(batch_images[i])        
        plt.show()