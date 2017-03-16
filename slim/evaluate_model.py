import numpy as np
import os

import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python.platform import tf_logging as logging
from datasets import dataset_factory, dataset_utils
from nets import nets_factory
from preprocessing import preprocessing_factory


tf.app.flags.DEFINE_string('checkpoint_path', None, 'Path to either a checkpoint dir or checkpoint file')
tf.app.flags.DEFINE_string('dataset_name', None, 'Name of the dataset to evaluate')
tf.app.flags.DEFINE_string('dataset_dir', None, 'Path to the dataset to evaluate')
tf.app.flags.DEFINE_string('model_name', None, 'Name of the model architecture')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'Name of the preprocessing function')
tf.app.flags.DEFINE_integer('eval_image_size', None, 'Size of the images')
tf.app.flags.DEFINE_boolean('oversampling', False, 'Use 10 crops to evaluate one image')
tf.app.flags.DEFINE_string('output_dir', '.', 'Output directory of result files')
tf.app.flags.DEFINE_string('output_prefix', None, 'Prefix for each result file name')
FLAGS = tf.app.flags.FLAGS


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        ####################################
        # Select dataset and create reader #
        ####################################
        dataset = dataset_factory.get_dataset(name=FLAGS.dataset_name,
                                              split_name='validation',
                                              dataset_dir=FLAGS.dataset_dir)

        provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                  shuffle=False,
                                                                  num_readers=1,
                                                                  common_queue_capacity=2,
                                                                  common_queue_min=1)
        [image, label, name] = provider.get(['image', 'label', 'name'])

        num_images = provider.num_samples()
        logging.info('Number of images in validation set: %d', num_images)

        ###############
        # Load labels #
        ###############
        if dataset_utils.has_labels(FLAGS.dataset_dir):
            labels = dataset_utils.read_label_file(FLAGS.dataset_dir)
        else:
            labels = None

        ########################
        # Create model network #
        ########################
        network_fn = nets_factory.get_network_fn(name=FLAGS.model_name,
                                                 num_classes=dataset.num_classes,
                                                 is_training=False)

        #################################
        # Select preprocessing function #
        #################################
        preprocess_fn = preprocessing_factory.get_preprocessing(name=FLAGS.preprocessing_name,
                                                                is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
        network_input_size = network_fn.default_image_size

        #############################
        # Take crops and preprocess #
        #############################
        if FLAGS.oversampling:
            pad = eval_image_size - network_input_size
            half_pad = int(pad / 2)
            flipped = tf.image.flip_left_right(image)
            crops = [
                tf.image.crop_to_bounding_box(image, 0, 0, network_input_size, network_input_size),
                tf.image.crop_to_bounding_box(image, pad, 0, network_input_size, network_input_size),
                tf.image.crop_to_bounding_box(image, 0, pad, network_input_size, network_input_size),
                tf.image.crop_to_bounding_box(image, pad, pad, network_input_size, network_input_size),
                tf.image.crop_to_bounding_box(image, pad / 2, pad / 2, network_input_size, network_input_size),
                tf.image.crop_to_bounding_box(flipped, 0, 0, network_input_size, network_input_size),
                tf.image.crop_to_bounding_box(flipped, pad, 0, network_input_size, network_input_size),
                tf.image.crop_to_bounding_box(flipped, 0, pad, network_input_size, network_input_size),
                tf.image.crop_to_bounding_box(flipped, pad, pad, network_input_size, network_input_size),
                tf.image.crop_to_bounding_box(flipped, half_pad, half_pad, network_input_size, network_input_size),
            ]

            crops = tf.pack([preprocess_fn(x, network_input_size, network_input_size) for x in crops], axis=0)
            images = tf.reshape(crops, [10, network_input_size, network_input_size, 3])
        else:
            crops = tf.expand_dims(preprocess_fn(image, network_input_size, network_input_size), 0)
            images = tf.reshape(crops, [1, network_input_size, network_input_size, 3])

        batch_queue = slim.prefetch_queue.prefetch_queue([images, label, name], capacity=4)
        images, label, name = batch_queue.dequeue()

        #####################
        # Instantiate model #
        #####################
        logits, _ = network_fn(images)
        probabilities = tf.nn.softmax(logits)

        ####################
        # Create a session #
        ####################
        with tf.Session() as sess:
            ######################
            # Restore checkpoint #
            ######################
            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            else:
                checkpoint_path = FLAGS.checkpoint_path

            tf.logging.info('Evaluating from %s' % checkpoint_path)
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, checkpoint_path)

            ############################
            # Start asynchronous tasks #
            ############################
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)

            #######################
            # Evaluate all images #
            #######################
            output_suffix = '_oversampled' if FLAGS.oversampling else ''
            output_prefix = FLAGS.output_prefix + '_' if FLAGS.output_prefix else ''

            prob_path = os.path.join(FLAGS.output_dir, output_prefix + 'probabilities' + output_suffix + '.csv')
            pred_path = os.path.join(FLAGS.output_dir, output_prefix + 'predictions' + output_suffix + '.csv')

            with open(prob_path, 'w') as ofile_prob, \
                    open(pred_path, 'w') as ofile_pred:
                # Write header lines to files
                ofile_pred.write('name,label\n')
                ofile_prob.write('name,%s\n' % ','.join([labels[x] for x in sorted(labels.keys())]))

                num_correct = 0
                for i in range(num_images):
                    # Execute one step
                    prediction, gt_class, image_name = sess.run([probabilities, label, name])

                    # Compute statistics
                    prediction = np.mean(prediction, axis=0)
                    predicted_class = np.argmax(prediction)
                    if predicted_class == gt_class:
                        num_correct += 1

                    # Get label of image
                    if labels:
                        predicted_label = labels[predicted_class]
                    else:
                        predicted_label = '%d' % predicted_class

                    # Write to output files
                    ofile_pred.write('%s,%s\n' % (image_name, predicted_label ))
                    ofile_prob.write('%s,%s\n' % (image_name, ','.join(['%f' % x for x in prediction])))

                    # Output status info
                    if (i + 1) % 100 == 0:
                        logging.info('%d/%d images processed' % (i+1, num_images))

            if (i + 1) % 100 != 0:
                logging.info('%d/%d images processed' % (num_images, num_images))

            logging.info('Accuracy: %f', num_correct / float(num_images))

            ###########################
            # Stop asynchronous tasks #
            ###########################
            coord.request_stop()
            coord.join(stop_grace_period_secs=5)

if __name__ == '__main__':
    tf.app.run()
