from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import logging
import os.path
import pprint
import time
import numpy as np
import tensorflow as tf

from model import RnnGan

flags = tf.flags
logging = tf.logging
pp = pprint.PrettyPrinter()

flags.DEFINE_integer("num_epoch", 1, "epoch or iterations to run training.")
flags.DEFINE_string("input_data_dir", None, "path to training/test data.")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing.")

FLAGS = flags.FLAGS

def main(_):
  """Read preprocessed data, init parameters, train, infer."""
  pp.pprint(flags.FLAGS.__flags)

  # Read preprocessed data from file path.
  if not FLAGS.input_data_dir:
    raise ValueError("Must set --input_data_dir")

  # TODO(elizachu): figure out better way to store/load preprocessed data.
  chroma = np.load(os.path.join(FLAGS.input_data_dir, "chroma.npy"))
  chord = np.load(os.path.join(FLAGS.input_data_dir, "chord.npy"))
  chord2index = np.load(
      os.path.join(FLAGS.input_data_dir, "chord2index.npy"))
  index2chord = np.load(
      os.path.join(FLAGS.input_data_dir, "index2chord.npy"))
  song_num = np.load(os.path.join(FLAGS.input_data_dir, "song_num.npy"))

  with tf.Session(config=tf.ConfigProto()) as sess:
    rnn_gan = RnnGan(sess, training_data=chroma, labels=chord)

  if FLAGS.is_train():
    print("============= TRAINING =============")
    rnn_gan.train(FLAGS)
    print("========== DONE TRRAINING ===========")
  else:
    if not FLAGS.checkpoint_dir:
      raise ValueError("Train model with --is_train, then run test mode with "
                       "trained output at --checkpoint_dir")
    rnn_gan.load(FLAGS.checkpoint_dir)


if __name__ == "__main__":
    tf.app.run()


