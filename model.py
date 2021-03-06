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
    # For true labels, 24 chords + N = 25 total. For converting chords to one-
    # hot vectors.
    self.LABEL_ONE_HOT_SIZE = 25
    ##############################

    self.sess = sess
    self.chroma = chroma
    self.chord = chord
    self.sequence_lengths = sequence_lengths
    self.index2chord = index2chord
    self.chord2index = chord2index
    self.flags = flags
    self.tensorboard_log_dir = flags.tensorboard_log_dir
    self.model_save_dir = flags.model_save_dir

    # Extract some useful numbers.
    self.num_songs = self.chroma.shape[0]

    # Batching data.
    self.chroma_input_placeholder = tf.placeholder(chroma.dtype, chroma.shape)
    self.chord_input_placeholder = tf.placeholder(chord.dtype, chord.shape)
    self.sequence_lengths_input_placeholder = tf.placeholder(
        sequence_lengths.dtype, sequence_lengths.shape)
    self.dataset = tf.data.Dataset.from_tensor_slices((
        self.chroma_input_placeholder,
        self.chord_input_placeholder,
        self.sequence_lengths_input_placeholder))
    self.dataset = self.dataset.batch(self.flags.minibatch_size)
    self.iterator = self.dataset.make_initializable_iterator()

    # Initialize tensorboard filewriter (saves summary data for visualization).
    if not self.tensorboard_log_dir:
      self.tensorboard_log_dir = os.getcwd()
    writer = tf.summary.FileWriter(self.tensorboard_log_dir)
    writer.add_graph(sess.graph)

    # Initialize generator & discriminator.
    self.d_logit_real = self.discriminator(self.chroma)
    #self.g_sample = self.generator()
    #self.d_logit_fake = self.discriminator(self.g_sample)

    # Remove timeline from chord labels. np.delete returns new array.
    self.Y_placeholder = tf.placeholder(
        self.chord.dtype, shape=(None, 15122, 1), name="Y_placeholder")

    # Convert labels to one-hot vector.
    self.Y_placeholder = tf.squeeze(tf.one_hot(
        indices=self.Y_placeholder, depth=self.LABEL_ONE_HOT_SIZE))

    # Then, append last class, all one's for real data.
    self.Y_placeholder = tf.concat([
        self.Y_placeholder,
        tf.ones(shape=[tf.shape(self.Y_placeholder)[0], 15122, 1])], axis=2)
    print("self.Y_placeholder.shape:", self.Y_placeholder.shape)

    self.Y_placeholder_noprop = tf.stop_gradient(self.Y_placeholder)
    
    # Discriminator loss.
    self.d_loss_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.d_logit_real,
            labels=self.labels))
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

    self.model_saver = tf.train.Saver()
    self.sess.run(tf.global_variables_initializer())

    next_element = self.iterator.get_next()

    for epoch in xrange(1, config.num_epoch+1):
      # Initialize batches.
      self.sess.run(self.iterator.initializer,
          feed_dict={
              self.chroma_input_placeholder: self.chroma,
              self.chord_input_placeholder: self.chord,
              self.sequence_lengths_input_placeholder: self.sequence_lengths,
          }
      )
      while True:
        try:
          chroma, chord, sequence_lengths = self.sess.run(next_element)
          _, loss_val = self.sess.run(
              [d_optimizer, self.d_loss],
              feed_dict={
                  self.X_placeholder: chroma,
                  self.Y_placeholder: chord,
                  self.seq_len_placeholder: sequence_lengths,
              })
        except tf.errors.OutOfRangeError:
          break

      print("epoch %d: self.d_loss = %f" % (epoch, loss_val))

      # Save the model every once in a while.
      if epoch % config.checkpoint_frequency == 0:
        self.model_saver.save(
            sess=self.sess,
            save_path=self.model_save_dir + r'/',
            global_step=epoch,
        )
        print("Saved model to %s." % self.model_save_dir)
       

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
    # Output should be [m, frame, prediction] where prediction is size 1.
    self.X_placeholder = tf.placeholder(
        tf.float32, shape=(None, 15122, 24), name="discriminator_X")
    self.seq_len_placeholder = tf.placeholder(
        self.sequence_lengths.dtype,
        shape=(None,),
        name="seq_len_placeholder"
    )

    with tf.variable_scope("discriminator_lstm_fw", reuse=tf.AUTO_REUSE,
        initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
        # 2-layer LSTM, each cell has num_hidden_units hidden units.
        rnn_cell_fw = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.LSTMCell(
                num_units=self.flags.num_hidden_units,
                num_proj=self.DISCRIMINATOR_OUTPUT_NUM_CLASSES),
            tf.contrib.rnn.LSTMCell(
                num_units=self.flags.num_hidden_units,
                num_proj=self.DISCRIMINATOR_OUTPUT_NUM_CLASSES),
        ])

    with tf.variable_scope("discriminator_lstm_bw", reuse=tf.AUTO_REUSE,
        initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
        # backwards LSTM. We want bi-directional for chord estimation.
        rnn_cell_bw = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.LSTMCell(
                num_units=self.flags.num_hidden_units,
                num_proj=self.DISCRIMINATOR_OUTPUT_NUM_CLASSES),
            tf.contrib.rnn.LSTMCell(
                num_units=self.flags.num_hidden_units,
                num_proj=self.DISCRIMINATOR_OUTPUT_NUM_CLASSES),
        ])
    
    # TODO: dropout
    
    # Input shape is (batch_size, n_time_steps, n_input), 
    # Output shape is (batch_size, n_time_steps, n_output).
    with tf.variable_scope("discriminator_bidi_lstm", reuse=tf.AUTO_REUSE,
        initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
      (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=rnn_cell_fw,
          cell_bw=rnn_cell_bw,
          dtype=tf.float32,
          # sequence_length s where to stop in a single training example.
          # Since we zero-pad inputs, we should stop early based on actual song
          # length to save computation cost.
          sequence_length=self.seq_len_placeholder,
          inputs=self.X_placeholder)

    # Concatenate forward and backward outputs.
    outputs = tf.concat([output_fw, output_bw], axis=2)
    print("concatenated outputs.shape:", outputs.shape)

    # Add fully connected layer as input to softmax later.
    logits = tf.contrib.layers.fully_connected(
        outputs, self.DISCRIMINATOR_OUTPUT_NUM_CLASSES, activation_fn=None)
    if not self.flags.is_train:
      self.d_logits = logits
    print("logits.shape:", logits.shape)

    return logits

  def load(self, sess, model_load_dir, model_load_meta_path, output_path):
    self.model_saver = tf.train.import_meta_graph(model_load_meta_path)
    self.model_saver.restore(sess,tf.train.latest_checkpoint(model_load_dir))
    print("Loaded model from %s, with meta file %s" % (
        model_load_dir, model_load_meta_path))

    probabilities = tf.nn.softmax(self.d_logits)
    predictions = tf.argmax(probabilities, axis=2)
    self.sess.run(tf.global_variables_initializer())
    probs, preds = self.sess.run(
        [probabilities, predictions],
        feed_dict={self.X_placeholder: self.chroma})
    print("probs.shape:", probs.shape)
    print("preds.shape:", preds.shape)

    # save predictions somewhere.
    np.save(os.path.join(output_path, "probabilities.npy"), probs)
    np.save(os.path.join(output_path, "predictions.npy"), preds)
    print("Saved predictions.npy and probabilities.npy to '%s'" % output_path)
