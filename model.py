from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

class RnnGan(object):
  """Generative Adversarial Net implemented with RNNs."""


  def __init__(self, sess, chroma, chord, sequence_lengths, 
      index2chord, chord2index, flags):
    """
    Args:
      sess: tf.Session.
      chroma: np.array or array_like, input data (X).
      chord: np.array or array_like, true labels of data (Y).
      sequence_lengths: np.array, length of each song in number of frames.
      index2chord: Python dictionary mapping indices to chords in string.
      chord2index: inverse of index2chord.
      flags: tf.flags, commandline flags passed in by the user.
    """

    ######### Constants ##########
    # Discriminator predicts chord labels = 24 major/minor, "N" label for no 
    # chords, and another class for whether or not the input is real. So 
    # 24 + 1 + 1 = 26.
    self.DISCRIMINATOR_OUTPUT_NUM_CLASSES = 26
    ##############################

    self.sess = sess
    self.chroma = chroma
    self.chord = chord
    self.sequence_lengths = sequence_lengths
    self.index2chord = index2chord
    self.chord2index = chord2index
    self.flags = flags
    self.tensorboard_log_dir = flags.tensorboard_log_dir

    # Initialize tensorboard filewriter (saves summary data for visualization).
    if not self.tensorboard_log_dir:
      self.tensorboard_log_dir = os.getcwd()
    writer = tf.summary.FileWriter(self.tensorboard_log_dir)
    writer.add_graph(sess.graph)

    # Initialize generator & discriminator.
    self.d_logit_real = self.discriminator(self.chroma)
    #self.g_sample = self.generator()
    #self.d_logit_fake = self.discriminator(self.g_sample)

    # Append extra class to the end of the predicted output to indicate whether     # or not the input data was real.
    self.d_logit_real = tf.concat(
        [self.d_logit_real, tf.ones(shape=[890, 15122, 1])], axis=2)
    #self.d_logit_fake = tf.concat(
    #    [self.d_logit_fake, tf.zeros(shape=[890, 15122, 1])], axis=2)

    # Remove timeline from chord labels. np.delete returns new array.
    self.labels = np.delete(self.chord, 0, axis=2)

    # Discriminator loss.
    self.d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_logit_real,
            labels=tf.concat(
                [self.labels, tf.ones(shape=[890, 15122, 1])], axis=2)))
    #self.d_loss_fake = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(
    #        logits=self.d_logit_fake,
    #        labels=tf.concat(
    #            [self.labels, tf.zeros(shape=[890, 15122, 1])], axis=2)))
    self.d_loss = self.d_loss_real #+ self.d_loss_fake

    # TODO: Generator loss.
    #self.g_loss = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(
    #        logits=self.d_logit_fake, labels=tf.ones_like(self.d_logit_fake)))

  def train(self, config):
    d_optimizer = tf.train.AdamOptimizer(
        config.learning_rate, beta1=config.beta1).minimize(
            self.d_loss) # TODO: only train D vars, i.e. var_list=self.theta_d)
    #g_optimizer = tf.train.AdamOptimizer(
    #    config.learning_rate, beta1=config.beta1).minimize(
    #        self.g_loss) # TODO: only train G vars, i.e. var_list=self.theta_g)

    #tf.global_variables_initializer()
    self.sess.run(tf.global_variables_initializer())

    for epoch in xrange(config.num_epoch):
      # TODO: make this work with generator.
      _, loss_val = self.sess.run([d_optimizer, self.d_loss],
          feed_dict={self.X_placeholder: self.chroma})
      print("epoch %d: self.d_loss = %f" % (epoch, loss_val))
       

  def generator(self):
    """Returns generated fake data."""
    return np.random.uniform(-1., 1., size=[890,15122,26])

  def discriminator(self, input_data, true_labels=None):
    """Discriminator takes data and outputs K+1 probability vector.
    
    K is number of output labels (in our chord estimation task, 25).
    +1 node to indicate whether D thinks the data is real or not.
    So total output = K+1 = 26 nodes.
    """

    # Input data X of shape [m, frame, chroma vector]. Zero-padded.
    # Output should be [m, frame, prediction] where prediction is a vector
    # of size 26 (25 chord classes + extra bit for real or generated data).
    self.X_placeholder = tf.placeholder(
        tf.float32, shape=(890, 15122, 25), name="discriminator_X")
    #Y = tf.placeholder(
    #    tf.float32, shape=(890, 15122, self.DISCRIMINATOR_OUTPUT_NUM_CLASSES))

    with tf.variable_scope("discriminator_lstm_fw", reuse=tf.AUTO_REUSE):
        # 2-layer LSTM, each cell has num_hidden_units hidden units.
        rnn_cell_fw = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.LSTMCell(
                num_units=self.flags.num_hidden_units,
                num_proj=self.DISCRIMINATOR_OUTPUT_NUM_CLASSES),
            tf.contrib.rnn.LSTMCell(
                num_units=self.flags.num_hidden_units,
                num_proj=self.DISCRIMINATOR_OUTPUT_NUM_CLASSES),
        ])

    with tf.variable_scope("discriminator_lstm_bw", reuse=tf.AUTO_REUSE):
        # backwards LSTM. We want bi-directional for chord estimation.
        rnn_cell_bw = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.LSTMCell(
                num_units=self.flags.num_hidden_units,
                num_proj=self.DISCRIMINATOR_OUTPUT_NUM_CLASSES),
            tf.contrib.rnn.LSTMCell(
                num_units=self.flags.num_hidden_units,
                num_proj=self.DISCRIMINATOR_OUTPUT_NUM_CLASSES),
        ])
    
    # TODO: dropout, minibatch...
    
    # Input shape is (batch_size, n_time_steps, n_input), 
    # Output shape is (batch_size, n_time_steps, n_output).
    with tf.variable_scope("discriminator_bidi_lstm", reuse=tf.AUTO_REUSE):
      (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=rnn_cell_fw,
          cell_bw=rnn_cell_bw,
          dtype=tf.float32,
          # sequence_length s where to stop in a single training example.
          # Since we zero-pad inputs, we should stop early based on actual song
          # length to save computation cost.
          sequence_length=self.sequence_lengths,
          inputs=self.X_placeholder)

    #print("output_fw.shape:", output_fw.shape)
    #print("output_bw.shape:", output_bw.shape)

    # Concatenate forward and backward outputs.
    outputs = tf.concat([output_fw, output_bw], axis=2)
    #print("concatenated outputs.shape:", outputs.shape)

    # Add fully connected layer as input to softmax later.
    logits = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=None)
    print("logits.shape:", logits.shape)

    return logits

  def load(self, checkpoint_dir):
    pass
