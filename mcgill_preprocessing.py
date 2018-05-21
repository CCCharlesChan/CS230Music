from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os.path
import pandas as pd

def mcgill_preprocess(sample_range=1301,
                      working_dir="",
                      output_list=True,
                      max_frame_num=15122,
                      print_progress=False):
  """Preprocessing for the McGill_Billboard dataset.
  
  Input:
      sample_range : Range of songs targeted for analysis (denoted by their
          numbers). The maximal song number is 1300.
      working_dir : Directory path till 'McGill_Billboard/'
      output_list : Whether or not to output in Python 'List' format. If opted
          out, chromagrams will be zero-padded to form matrices of uniform
          dimension (max_frame_num, 25). However, a typical 3 minute song has
          ~4000 frames, while the longest song has 15122. Therefore, it is
          generally not an efficient way to use array. A list with appropriate
          dimensionality can be easily converted to a numpy array by
          numpy.asarray().
      max_frame_num : Only relevant when output_list = False. The length to
          which each chromagram will be zero-padded to.
      print_progress : Boolean, if True, will print number of true samples
          that are processed for every 50 samples.
  
  Output:
      chroma : List of length m, or array of dimension (M, max_frame_num, 25),
          where M is number of valid samples. (m, f, 1:25) is the chromagram
          of song m in time frame f. (m, :, 0) is timeline.
      chord : List of length m, or array of dimension (M, max_frame_num, 2),
          where M is number of valid samples. (m, f, 1) is the chord in time
          frame f. (m, :, 0) is timeline.
      dict_chord2idx: Dictionary. Mapping from chord name (e.g., "Cmin") to
          integer.
      dict_idx2chord: Dictionary. The invert of dict_chord2idx.
      song_num : List. The actual file number of each song. 
  """
  chroma_base = "Chroma_vector"
  chroma_filename = "bothchroma.csv"
  chord_base = "MIREX_style"
  chord_filename = "majmin.lab"
  
  dict_chord2idx = dict()
  dict_idx2chord = dict()
  idx_chord = 0
  
  song_num = []
  if output_list:
    chroma = []
    chord = []
  else:
    chroma = np.zeros((0, max_frame_num, 25))
    chord = np.zeros((0, max_frame_num, 2))
  
  true_samples = 0
  for sample_num in xrange(sample_range):
    chroma_dir = os.path.join(
        working_dir, chroma_base,'{:0>4}'.format(sample_num))
    chord_dir = os.path.join(
        working_dir, chord_base,'{:0>4}'.format(sample_num))
    if not (os.path.isdir(chroma_dir) and os.path.isdir(chord_dir)):
        continue
    
    true_samples += 1
    song_num.append(sample_num)
    chroma_dat = pd.read_csv(
        os.path.join(chroma_dir, chroma_filename), header=None)
    chroma_dat = chroma_dat.drop(labels=0, axis=1)
    chord_dat = pd.read_csv(
        os.path.join(chord_dir, chord_filename),
        delimiter='\t',
        header=None)
    
    for i in range(chord_dat.shape[0]):
      if chord_dat[2][i] not in dict_chord2idx:
        dict_chord2idx.update({chord_dat[2][i]: idx_chord})
        dict_idx2chord.update({idx_chord: chord_dat[2][i]})
        idx_chord +=1
      
    chord_label = np.zeros((chroma_dat.shape[0], 2))
    for i in range(chord_dat.shape[0]):
      sel_time = np.logical_and(
          chroma_dat[1] >= chord_dat[0][i],
          chroma_dat[1] < chord_dat[1][i])
      chord_label[sel_time, 1] = dict_chord2idx[chord_dat[2][i]]
      chord_label[:,0] = chroma_dat[1]
      
    chroma_ready = chroma_dat.as_matrix()
      
    if output_list:
      chroma.append(chroma_ready)
      chord.append(chord_label)
      if print_progress and true_samples % 50 == 0:
          print("Preprocessing McGill data... processed %d songs." %
                true_samples)
    else:
      chroma_ready = np.pad(
          array=chroma_ready,
          pad_width=((0, max_frame_num-chroma_ready.shape[0]), (0,0)),
          mode= 'constant',
          constant_values = 0)
      chroma = np.concatenate(
          (chroma, np.zeros((1, max_frame_num,25))), axis = 0)
      chroma[-1, :, :] = chroma_ready
        
      chord_ready = np.pad(
          array=chord_label,
          pad_width=((0, max_frame_num-chord_label.shape[0]), (0,0)),
          mode= 'constant', constant_values = 0)
      chord = np.concatenate(
          (chord, np.zeros((1, max_frame_num,2))), axis = 0)
      chord[-1, :, :] = chord_ready
      if print_progress and true_samples % 50 == 0:
        print("Preprocessing McGill data... processed %d songs." %
              true_samples)
  
  if print_progress:
    print("Done with preprocessing! Total processed samples: %d songs." %
          true_samples)
  
  return chroma, chord, dict_chord2idx, dict_idx2chord, song_num


def preprocess_data_and_store(input_dir, output_dir, verbose=False):
  """Preprocess & store the McGill data once and for all.

  Args:
    intput_dir: full path to where McGill_Billboard dataset is.
    output_dir: where to store the output. Must already exist.
    verbose: Boolean, if True, will print progress to stdout.

  TODO(elizachu): stop hard-coding the directory paths. Use optparse.
  """

  # TODO(elizachu): open a file and write periodically in this function, instead
  # of waiting for all songs to be processed then saving them. If program is
  # interrupted, we risk saving corrupted data.
  chroma, chord, chord2index, index2chord, song_num = mcgill_preprocess(
      working_dir=working_dir,
      output_list=False,
      print_progress=verbose)

  chroma_filename = os.path.join(output_dir, "chroma")
  chord_filename = os.path.join(output_dir, "chord")
  chord2index_filename = os.path.join(output_dir, "chord2index")
  index2chord_filename = os.path.join(output_dir, "index2chord")
  song_num_filename = os.path.join(output_dir, "song_num")

  np.save(chroma_filename, chroma)
  np.save(chord_filename, chord)
  np.save(chord2index_filename, chord2index)
  np.save(index2chord_filename, index2chord)
  np.save(song_num_filename, song_num)

  if verbose:
    print("Saved all numpy outputs to: %s." % output_dir)
    print("chroma.shape:", chroma.shape)
    print("chroma[0].shape:", chroma[0].shape)
    print("chord.shape:", chord.shape)
    print("chord[0].shape:", chord[0].shape)
    print("index2chord[0]:", index2chord[0])
  

def extract_song_lengths(input_dir, output_dir):
  """Given a chroma output where output_list=True, save a song_lengths.npy.
   
  Args:
    input_dir: directory where .npy files were stored, when it was processed
      with output_list = True.
    output_dir: string, where to save the song_lengths.npy output.
      song_lengths.npy is a numpy array of each song's length in number of
      frames.
  """
  print("Extracting song lengths...")

  # chroma_output_dir is 
  chroma = np.load(os.path.join(chroma_output_dir, "chroma.npy"))

  song_lengths = [chroma[i].shape[0] for i in xrange(chroma.shape[0])]
  print(song_lengths[0:10])
  np.save(os.path.join(output_dir, "song_lengths.npy"), song_lengths)
  print("Saved to song_lengths.npy")

def main(unused_argv=None):

  # working_dir should be full path to where McGill_Billboard dataset is.
  working_dir = "/Users/elizachu/Desktop/McGill_Billboard"

  # output_dir needs to already exist; full path to directory to save .npy
  # output files.
  output_dir = "/Users/elizachu/Desktop/mcgill_zero_pad"

  # This should be output_dir of when we ran preprocess_data... with
  # output_list = True.
  list_input_dir = "/Users/elizachu/Desktop/preprocessed_mcgill"

  preprocess_data_and_store(
      input_dir=working_dir, output_dir=output_dir, verbose=True)
  extract_song_lengths(input_dir=list_input_dir, output_dir=output_dir)


if __name__ == "__main__":
  main()
