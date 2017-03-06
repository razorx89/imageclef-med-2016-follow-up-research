# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf
import cv2
import numpy as np
from collections import Counter
import csv

from datasets import dataset_utils

_TRAIN_PATHS = [
    '/home/koitka/ImageCLEF/2013/ImageCLEF2013TrainingSet',
    '/home/koitka/ImageCLEF/2013/ImageCLEF2013TestSetGROUNDTRUTH',
    '/home/koitka/ImageCLEF/2016/SubfigureClassificationTraining2016'
]
_TEST_PATHS = ['/home/koitka/ImageCLEF/2016/SubfigureClassificationTest2016GT']

# The number of images in the validation set.
_NUM_VALIDATION = 4166

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

# Final image size to store
_IMAGE_SIZE = 360

# Generates a debug directory with images in the dataset directory
_GENERATE_DEBUG_IMAGES = True

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dirs):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    directories = []
    class_names = set()
    for dir in dirs:
        for filename in os.listdir(dir):
            path = os.path.join(dir, filename)
            if os.path.isdir(path):
                directories.append(path)
                class_names.add(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1] not in ['.jpg', '.jpeg']:
                continue
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(list(class_names))


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'imageclef_med_2016_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, meta_file):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    if _GENERATE_DEBUG_IMAGES:
        if not os.path.exists(os.path.join(dataset_dir, 'debug')):
            os.makedirs(os.path.join(dataset_dir, 'debug'))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            csv_writer = None

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        # image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                        image = cv2.imread(filenames[i], cv2.IMREAD_COLOR)
                        image, meta_info = _preprocess_image(image, squash=False)
                        meta_info['dataset'] = os.path.basename(os.path.dirname(os.path.dirname(filenames[i])))

                        if csv_writer is None:
                            csv_writer = csv.DictWriter(meta_file, sorted(meta_info.keys()))
                            if split_name == 'train':
                                csv_writer.writeheader()
                        csv_writer.writerow(meta_info)

                        # height, width = image_reader.read_image_dims(sess, image_data)
                        height, width, _ = image.shape

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        # Write image to HDD for debug purposes
                        if _GENERATE_DEBUG_IMAGES:
                            image_name = os.path.splitext(os.path.basename(filenames[i]))[0]
                            cv2.imwrite(os.path.join(dataset_dir, 'debug', image_name + '.png'), image,
                                        [cv2.IMWRITE_PNG_COMPRESSION, 6])

                        # example = dataset_utils.image_to_tfexample(image_data, 'jpg', height, width, class_id)
                        _, image = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                        example = dataset_utils.image_to_tfexample(image.tostring(), 'png', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _preprocess_image(image, squash=True, no_upscaling=True):
    if squash:
        meta_info = {
            'height': image.shape[0],
            'width': image.shape[1]
        }
        result = cv2.resize(image, (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    else:
        # Remove homogeneous areas at borders
        shape_before_crop = image.shape
        image, dominant_color = _auto_crop(image)
        shape_after_crop = image.shape

        # Resize image while keeping the aspect ratio fixed
        new_shape, padding = _compute_ara_image_shape(image.shape, _IMAGE_SIZE, no_upscaling=no_upscaling)
        image = cv2.resize(image, new_shape[0:2], interpolation=cv2.INTER_CUBIC)

        result = np.full((_IMAGE_SIZE, _IMAGE_SIZE, 3), dominant_color, dtype=image.dtype)
        #result = np.ones((_IMAGE_SIZE, _IMAGE_SIZE, 3), image.dtype)
        result[padding[1]:padding[1]+new_shape[1], padding[0]:padding[0]+new_shape[0], :] = image

        meta_info = {
            'height_before_crop': shape_before_crop[0],
            'width_before_crop': shape_before_crop[1],
            'height_after_crop': shape_after_crop[0],
            'width_after_crop': shape_after_crop[1]
        }

    return result, meta_info


def _auto_crop(image):
    x1, x2 = 0, image.shape[0]-1
    y1, y2 = 0, image.shape[1]-1
    all_cropped_border_colors = []

    while x1 < image.shape[0]:
        if not np.all(image[x1, 0, :] == image[x1, :, :]):
            break
        all_cropped_border_colors.extend([tuple(x) for x in image[x1, :, :]])
        x1 += 1

    while x2 > x1:
        if not np.all(image[x2, 0, :] == image[x2, :, :]):
            break
        all_cropped_border_colors.extend([tuple(x) for x in image[x2, :, :]])
        x2 -= 1

    if x1 >= x2:
        x1, x2 = 0, 0

    while y1 < image.shape[1]:
        if not np.all(image[0, y1, :] == image[:, y1, :]):
            break
        all_cropped_border_colors.extend([tuple(x) for x in image[:, y1, :]])
        y1 += 1

    while y2 > y1:
        if not np.all(image[0, y2, :] == image[:, y2, :]):
            break
        all_cropped_border_colors.extend([tuple(x) for x in image[:, y2, :]])
        y2 -= 1

    if y1 >= y2:
        y1, y2 = 0, 0

    tmp = Counter(all_cropped_border_colors)
    tmp = tmp.most_common(1)

    return image[x1:x2+1, y1:y2+1, :], tmp[0][0] if len(tmp) > 0 else (255, 255, 255)


def _compute_ara_image_shape(shape, new_dim, no_upscaling=False):
    aspect_ratio = float(shape[0]) / float(shape[1])
    if aspect_ratio > 1:
        # Portrait
        new_height = min(shape[0], new_dim) if no_upscaling else new_dim
        new_width = int(new_height / aspect_ratio)
        pad_top = 0
        pad_left = (new_dim - new_width) / 2
    else:
        # Landscape
        new_width = min(shape[1], new_dim) if no_upscaling else new_dim
        new_height = int(aspect_ratio * new_width)
        pad_left = 0
        pad_top = (new_dim - new_height) / 2

    return (new_width, new_height, shape[2]), (int(pad_left), int(pad_top))


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    training_filenames, class_names = _get_filenames_and_classes(_TRAIN_PATHS)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    validation_filenames, _ = _get_filenames_and_classes(_TEST_PATHS)

    random.seed(_RANDOM_SEED)
    random.shuffle(training_filenames)
    random.shuffle(validation_filenames)

    # First, convert the training and validation sets.
    with open(os.path.join(dataset_dir, 'meta.csv'), 'wb') as meta_file:
        _convert_dataset('train', training_filenames, class_names_to_ids,
                         dataset_dir, meta_file)
        _convert_dataset('validation', validation_filenames, class_names_to_ids,
                         dataset_dir, meta_file)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    print('\nFinished converting the "ImageCLEFmed 2016 Subfigure Classification" dataset!')
