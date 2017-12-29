import tensorflow as tf
import numpy as np
import os
from os.path import isfile, join
from tensorflow.contrib.data import Dataset, Iterator

def input_parser(img_path):

    # read the img from file
    img_file = tf.read_file(img_path)
    print('read ', img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    return img_decoded

def distort_input(img):

    # could be useful down the line: tf.image.total_variation
    # list of noise operations, tf.image.random_brightness, tf.image.random_contrast, tf.image.random_hue

    distorted_image = tf.image.random_brightness(img, max_delta=0.2)
    distorted_image = tf.image.random_contrast(distorted_image, lower=1.2, upper=2)
    distorted_image = tf.image.random_hue(distorted_image, max_delta=0.01)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # return a tuple of noisy and original image pairs
    return (float_image, img)

def input_pipeline(data_dir):

    BATCH_SIZE = 16

    filenames = [join(data_dir, filename) for filename in os.listdir(data_dir) if isfile(join(data_dir, filename))]

    np.random.shuffle(filenames)

    train_images = tf.constant(filenames)

    tr_data = Dataset.from_tensor_slices(train_images)
    tr_data = tr_data.map(input_parser).map(distort_input)

    tr_data = tr_data.batch(16)

    # create TensorFlow Iterator object
    iterator = Iterator.from_structure(tr_data.output_types,
                                       tr_data.output_shapes)

    training_init_op = iterator.make_initializer(tr_data)

    next_element = iterator.get_next()

    with tf.Session() as sess:

        # initialize the iterator on the training data
        sess.run(training_init_op)

        # get each element of the training dataset until the end is reached
        while True:
            try:
                elem = sess.run(next_element)
                print(elem)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
