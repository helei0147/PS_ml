"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.

TensorFlow install instructions:
https://tensorflow.org/get_started/os_setup.html

MNIST tutorial:
https://tensorflow.org/tutorials/mnist/tf/index.html
"""
import math

import tensorflow.python.platform
import tensorflow as tf

OBSERVATION_NUM = 96


def inference(images):
    hidden1_units = 4096
    hidden2_units = 4096
    hidden3_units = 2048
    hidden4_units = 2048
    hidden5_units = 2048
    # Hidden 1
    with tf.name_scope('hidden1') as scope:
        weights = tf.Variable(
            tf.truncated_normal([OBSERVATION_NUM, hidden1_units],
                                stddev=1.0 / math.sqrt(float(OBSERVATION_NUM))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2') as scope:
        keep_prob = tf.placeholder(tf.float32);
        hidden1_drop = tf.nn.dropout(hidden1,keep_prob);
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, weights) + biases)
    # Hidden 3
    with tf.name_scope('hidden3') as scope:
        keep_prob = tf.placeholder(tf.float32);
        hidden2_drop = tf.nn.dropout(hidden2,keep_prob);
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, hidden3_units],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden3_units]),
                             name='biases')
        hidden3 = tf.nn.relu(tf.matmul(hidden2_drop, weights) + biases)
    # Hidden 4
    with tf.name_scope('hidden4') as scope:
        keep_prob = tf.placeholder(tf.float32);
        hidden3_drop = tf.nn.dropout(hidden3,keep_prob);
        weights = tf.Variable(
            tf.truncated_normal([hidden3_units, hidden4_units],
                                stddev=1.0 / math.sqrt(float(hidden3_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden4_units]),
                             name='biases')
        hidden4 = tf.nn.relu(tf.matmul(hidden3_drop, weights) + biases)
    # Hidden 5
    with tf.name_scope('hidden5') as scope:
        keep_prob = tf.placeholder(tf.float32);
        hidden4_drop = tf.nn.dropout(hidden4,keep_prob);
        weights = tf.Variable(
            tf.truncated_normal([hidden4_units, hidden5_units],
                                stddev=1.0 / math.sqrt(float(hidden4_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden5_units]),
                             name='biases')
        hidden5 = tf.nn.relu(tf.matmul(hidden4_drop, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear') as scope:
        weights = tf.Variable(
            tf.truncated_normal([hidden5_units, 3],
                                stddev=1.0 / math.sqrt(float(hidden5_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([3]),
                             name='biases')
        normals = tf.matmul(hidden5, weights) + biases
    return normals


def loss(est_normals, gts):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    pow_para = tf.zeros(tf.shape(est_normals))+2
    a = est_normals-gts
    L = tf.pow(a, pow_para)
    loss = tf.reduce_sum(L,1)
    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: observation tensor, float - [batch_size, NUM_CLASSES].
      labels: normal tensor, float - [batch_size, 3]

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
