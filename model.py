import os.path
import time
import numpy as np
import tensorflow as tf

class RnnGan(object):
  """Generative Adversarial Net implemented with RNNs."""

  def __init__(self, sess, training_data, labels):
    """
    Args:
      sess: tf.Session.
      training_data: np.array or array_like, input training data (X).
      labels: np.array or array_like, true labels of data (Y).
    """
    self.sess = sess
    self.data = training_data
    self.labels = labels

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
      self.sess.run([d_optimizer, d_loss],
                    feed_dict={X: self.data, Y: self.labels})
       

  def generator(self):
    """Returns generated fake data."""
    return np.random.uniform(-1., 1., size=[100,100])

  def discriminator(self, input_data, true_labels=None):
    """Discriminator takes data and outputs probability the data is real."""
    # Create placeholdrs & initialize parameters.
    X = tf.placeholder(tf.float32, shape=(None, 15122, 25), name="X")
    with tf.variable_scope("discriminator") as scope:
      d_W1 = tf.get_variable("d_W1", shape=(100, 100),
          initializer=tf.contrib.layers.xavier_initializer())
      d_b1 = tf.get_variable("d_b1", shape=(1, 100),
          initializer=tf.zeros_initializer())
      d_W2 = tf.get_variable("d_W2", shape=(100, 25),
          initializer=tf.contrib.layers.xavier_initializer())
      d_b2 = tf.get_variable("d_b2", shape=(1, 25),
          initializer=tf.zeros_initializer())

    # TODO(elizachu): Forward prop with tf.contrib.rnn.BasicLSTMCell or 
    # tf.contrib.rnn.MultiRNNCell or tf.nn.static_rnn() to compute prediction.

    prediction = 0.5
    d_theta = [d_W1, d_b1, d_W2, d_b2]
    return (prediction, d_theta)

  def load(self, checkpoint_dir):
    pass
