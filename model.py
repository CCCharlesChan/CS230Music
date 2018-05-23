import os.path
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
    
    K is number of output labels (in our chord estimation task, 38?).
    +1 node to indicate whether D thinks the data is real or not.
    So total output = K+1 nodes.
    """

    # Input data X of shape [m, frame, chroma vector]. Zero-padded.
    X = tf.placeholder(tf.float32, shape=(None, 15122, 25))

    # 2-layer LSTM, each layer has num_hidden_units hidden units.
    rnn_cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.BasicLSTMCell(self.flags.num_hidden_units),
        tf.contrib.rnn.BasicLSTMCell(self.flags.num_hidden_units),
    ])

    # Define initial state.
    initial_state = rnn_cell.zero_state(batch_size=15122, dtype=tf.float32)

    outputs, state = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        dtype=tf.float32,
        # sequence_length indicates where to stop in a single training example.
        # Since we zero-pad inputs, we should stop early based on actual song
        # length to save computation cost.
        sequence_length=self.sequence_lengths,
        inputs=X)

    # Only care about output activation at last layer.
    print("outputs.shape:", outputs.shape)
    outputs = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

    # Add softmax classifier.
    out_size = 38  # Number of chords according to index2chord.
    logit = tf.contrib.layers.fully_connected(
        last, out_size, activation_fn=None)
    prediction = tf.nn.softmax(logit)

    return prediction

  def load(self, checkpoint_dir):
    pass
