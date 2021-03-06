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

flags.DEFINE_float("beta1", 0.9, "beta1 for AdamOptimizer.")
flags.DEFINE_float("learning_rate", 0.001, "learning rate for AdamOptimizer.")
flags.DEFINE_integer("checkpoint_frequency", 1, 
    "How often to save model during training, in num epochs [Default=1].")
flags.DEFINE_integer("minibatch_size", 64, "size of one batch for training.")
flags.DEFINE_integer("num_epoch", 1, "epoch or iterations to run training.")
flags.DEFINE_integer("num_hidden_units", 10, 
                     "number of hidden units in each layer.")
flags.DEFINE_string("input_data_dir", None, "path to training/test data.")
flags.DEFINE_string("model_load_dir", None,
    "path to load a model previously trained with --is_train=True. This runs a "
    "test or validation on input_data_dir.")
flags.DEFINE_string("model_load_meta_path", None,
    "path to .meta file of the saved model.")
flags.DEFINE_string("model_save_dir", None,
    "path to save the trained model. --is_train must be True. If unspecified, "
    "creates a dir under current working dir called './model_YYYYMMDD_HHMMSS'.")
flags.DEFINE_string("output_path", None,
    "path to save predicted outputs. This is a no-op if in training mode "
    "(--is_train). Saves prediction.npy in this output_path. Default uses "
    "--model_load_dir.")
flags.DEFINE_string("tensorboard_log_dir", None, 
    "path to save tensorboard log data, defaults to current working dir.")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing.")

FLAGS = flags.FLAGS

def main(_):
  """Read preprocessed data, init parameters, train, infer."""
  pp.pprint(flags.FLAGS.flag_values_dict())

  if FLAGS.is_train:
    if not FLAGS.input_data_dir:
      raise ValueError("Must set --input_data_dir if --is_train is True.")
    if not FLAGS.model_save_dir:
      # Save model under current working dir by default.
      model_dir_name = "rnngan_%s" % (time.strftime("%Y%m%d_%H%M%S"))
      model_dir = os.path.join(os.getcwd(), model_dir_name)
      FLAGS.model_save_dir = model_dir

  chroma = np.delete(np.load(
      os.path.join(FLAGS.input_data_dir, "chroma.npy")).astype(np.float32),
      0, axis=2)
  chord = np.delete(
      np.load(os.path.join(FLAGS.input_data_dir, "chord.npy")).astype(np.int32),
      0, axis=2)
  chord2index = np.load(
      os.path.join(FLAGS.input_data_dir, "chord2index.npy")).item()
  index2chord = np.load(
      os.path.join(FLAGS.input_data_dir, "index2chord.npy")).item()
  song_num = np.load(
      os.path.join(FLAGS.input_data_dir, "song_num.npy")).tolist()
  song_lengths = np.load(
      os.path.join(FLAGS.input_data_dir, "song_lengths.npy"))

  print("++++++++++++ INPUT DATA SANITY CHECK ++++++++++++++")
  print("chroma.shape:", chroma.shape)
  print("chord.shape:", chord.shape)
  print("len(chord2index.keys()):", len(chord2index.keys()))
  print("len(index2chord.keys()):", len(index2chord.keys()))
  print("len(song_num):", len(song_num))
  print("song_lengths.shape:", song_lengths.shape)
  print("chroma.dtype:", chroma.dtype)
  print("chord.dtype:", chord.dtype)
  print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

  with tf.Session(config=tf.ConfigProto()) as sess:
    rnn_gan = RnnGan(
        sess,
        chroma=chroma,
        chord=chord,
        sequence_lengths=song_lengths,
        index2chord=index2chord,
        chord2index=chord2index,
        flags=FLAGS)

    if FLAGS.is_train:
      print("============= TRAINING =============")
      rnn_gan.train(FLAGS)
      print("========== DONE TRRAINING ===========")
    else:
      if not FLAGS.model_load_dir or not FLAGS.model_load_meta_path:
        raise ValueError("Train model with --is_train, then run test mode with "
                         "trained output at --model_load_dir and meta file at "
                         "--model_load_meta_path.")
      if not os.path.exists(FLAGS.model_load_dir):
        raise ValueError("Cannot find model_load_dir, are you sure it "
                         "exists? %s" % FLAGS.model_load_dir)
      if not os.path.exists(FLAGS.model_load_meta_path):
        raise ValueError("Cannot find model_load_meta_path, are you sure it "
                         "exists? %s" % FLAGS.model_load_meta_path)
      
      if not FLAGS.output_path:
        FLAGS.output_path = FLAGS.model_load_dir

      print("============ GENERATING VALIDATION OUTPUT ============")
      rnn_gan.load(
          sess,
          FLAGS.model_load_dir,
          FLAGS.model_load_meta_path,
          FLAGS.output_path,
      )
      print("================= DONE VALIDATION =================")


if __name__ == "__main__":
    tf.app.run()


