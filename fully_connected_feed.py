"""Trains and Evaluates the MNIST network using a feed dictionary.

TensorFlow install instructions:
https://tensorflow.org/get_started/os_setup.html

MNIST tutorial:
https://tensorflow.org/tutorials/mnist/tf/index.html

"""
from __future__ import print_function
# pylint: disable=missing-docstring
import os.path
import time

import tensorflow.python.platform
import numpy as np
import tensorflow as tf

import input_data
import mnist


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 4096, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 4096, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 2048, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 2048, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 2048, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('batch_size', 1000, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('train_channel_index', 1, 'Directory to put the training data.')
flags.DEFINE_string('log_dir','log/fully_connected_feed','dir to put log data')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.OBSERVATION_NUM))
  labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size,3))
  keep_prob_placeholder = tf.placeholder(tf.float32)
  shadow_prob_placeholder = tf.placeholder(tf.float32)
  return images_placeholder, labels_placeholder, keep_prob_placeholder, shadow_prob_placeholder


def fill_feed_dict(data_set, observations_pl, normals_pl, keep_prob_pl, shadow_prob_pl, keep_prob_value,shadow_prob_value):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  observations_feed, normals_feed = data_set.next_batch(FLAGS.batch_size)
  feed_dict = {
      observations_pl: observations_feed,
      normals_pl: normals_feed,
      keep_prob_pl: keep_prob_value,
      shadow_prob_pl: shadow_prob_value

  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            keep_prob_placeholder,
            shadow_prob_placeholder,
            data_set,
            keep_prob_value,
            shadow_prob_value):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  error = 0  # Counts the number of correct predictions.
  steps_per_epoch = int(data_set.num_examples / FLAGS.batch_size)
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder,
                               keep_prob_placeholder,
                               shadow_prob_placeholder,
                               keep_prob_value,
                               shadow_prob_value)
    error += sess.run(eval_correct, feed_dict=feed_dict)
  avg_error = float(error) / float(num_examples)
  print('  Num examples: %d  avg_error: %0.04f' %
        (num_examples, avg_error))


def run_training(channel_index, log_folder):
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(channel_index)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder, keep_prob_placeholder, shadow_prob_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits, keep_prob_placeholder, shadow_prob_placeholder = mnist.inference(images_placeholder)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter('summary',
                                            graph=sess.graph)
    _total_experiment_time = 10000000
    shadow_prob_buffer = np.random.binomial(_total_experiment_time, 0.05, FLAGS.max_steps)
    shadow_prob_buffer = shadow_prob_buffer/_total_experiment_time
    shadow_prob_buffer = 1-shadow_prob_buffer
    # And then after everything is built, start the training loop.
    for step in range(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder,
                                 keep_prob_placeholder,
                                 shadow_prob_placeholder,
                                 0.5,
                                 shadow_prob_buffer[step])

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_file = os.path.join(log_folder, 'model.ckpt')
          saver.save(sess, checkpoint_file, global_step=step)


def main(_):
  run_training(1, 'log/shadow/channel1')
  run_training(2, 'log/shadow/channel2')
  run_training(3, 'log/shadow/channel3')


if __name__ == '__main__':
  tf.app.run()
