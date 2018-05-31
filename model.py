from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

class RnnGan(object):
  """Generative Adversarial Net implemented with RNNs."""

  def __init__(self, sess, training_data, labels, sequence_lengths, flags):
    """
    Args:
      sess: tf.Session.
      training_data: np.array or array_like, input training data (X).
      labels: np.array or array_like, true labels of data (Y).
      sequence_lengths: np.array, length of each song in number of frames.
      flags: tf.flags, commandline flags passed in by the user.
    """
    self.sess = sess
    self.data = training_data
    self.labels = labels
    self.sequence_lengths = sequence_lengths
    self.flags = flags
    self.tensorboard_log_dir = flags.tensorboard_log_dir

    # Initialize tensorboard filewriter (saves summary data for visualization).
    if not self.tensorboard_log_dir:
      self.tensorboard_log_dir = os.getcwd()
    writer = tf.summary.FileWriter(self.tensorboard_log_dir)
    writer.add_graph(sess.graph)

    # Initialize generator & discriminator.
    self.d_real, self.d_logit_real = self.discriminator(self.data)
    self.g_sample = self.generator()
    self.d_fake, self.d_logit_fake = self.discriminator(self.g_sample)

    # Discriminator loss.
    self.d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            self.d_logit_real, tf.ones_like(self.d_logit_real)))
    self.d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            self.d_logit_fake, tf.zeros_like(self.d_logit_fake)))
    self.d_loss = self.d_loss_real + self.d_loss_fake

    # Generator loss.
    self.g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            self.d_logit_fake, tf.ones_like(self.d_logit_fake)))

  def train(self, config):
    d_optimizer = tf.train.AdamOptimizer(
        config.learning_rate, beta1=config.beta1).minimize(
            self.d_loss, var_list=self.theta_d)
    g_optimizer = tf.train.AdamOptimizer(
        config.learning_rate, beta1=config.beta1).minimize(
            self.g_loss, var_list=self.theta_g)

    tf.global_variables_initializer().run()

    for epoch in xrange(config.num_epoch):
      self.sess.run([d_optimizer, self.d_loss],
                    feed_dict={X: self.data, Y: self.labels})
       

  def generator(self):
    """Returns generated fake data."""
    return np.random.uniform(-1., 1., size=[100,100])

  def discriminator(self, input_data, true_labels=None):
    """Discriminator takes data and outputs K+1 probability vector.
    
    K is number of output labels (in our chord estimation task, 25).
    +1 node to indicate whether D thinks the data is real or not.
    So total output = K+1 = 26 nodes.
    """

    # Input data X of shape [m, frame, chroma vector]. Zero-padded.
    # Output should be [m, frame, prediction] where prediction is a vector
    # of size 26 (25 chord classes + extra bit for real or generated data).
    X = tf.placeholder(tf.float32, shape=(890, 15122, 25))
    Y = tf.placeholder(tf.float32, shape=(890, 15122, 26))

    # Use named scopes for better tensorboard visualization.
    with tf.name_scope("discriminator_lstm_fw"):
        # 2-layer LSTM, each cell has num_hidden_units hidden units.
        rnn_cell_fw = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.BasicLSTMCell(self.flags.num_hidden_units),
            tf.contrib.rnn.BasicLSTMCell(self.flags.num_hidden_units),
        ])

    with tf.name_scope("discriminator_lstm_bw"):
        # backwards LSTM. We want bi-directional for chord estimation.
        rnn_cell_bw = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.BasicLSTMCell(self.flags.num_hidden_units),
            tf.contrib.rnn.BasicLSTMCell(self.flags.num_hidden_units),
        ])
    
    # TODO: implement initial state of LSTM.
    # TODO: dropout, minibatch...
    
    # Input shape is (batch_size, n_time_steps, n_input), 
    # Output shape is (batch_size, n_time_steps, n_output).
    (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=rnn_cell_fw,
        cell_bw=rnn_cell_bw,
        dtype=tf.float32,
        # sequence_length indicates where to stop in a single training example.
        # Since we zero-pad inputs, we should stop early based on actual song
        # length to save computation cost.
        sequence_length=self.sequence_lengths,
        inputs=X)

    print("output_fw.shape:", output_fw.shape)
    print("output_bw.shape:", output_bw.shape)

    # Concatenate forward and backward outputs.
    outputs = tf.concat([output_fw, output_bw], axis=2)
    print("concatenated outputs.shape:", outputs.shape)

    # Add softmax classifier.
    out_size = 26
    logit = tf.contrib.layers.fully_connected(
        outputs, out_size, activation_fn=None)
    prediction = tf.nn.softmax(logit)

    return prediction

  def load(self, checkpoint_dir):
    pass
