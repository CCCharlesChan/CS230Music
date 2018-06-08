from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os.path
import pandas as pd

def get_set(set_type):
  samples = ["0003", "0004", "0006", "0010", "0012", "0015", "0016", "0018", "0019", "0021", "0022", "0023", "0025", "0026", "0027", "0029", "0030", "0033", "0034", "0035", "0037", "0039", "0040", "0041", "0043", "0044", "0046", "0049", "0050", "0051", "0053", "0054", "0055", "0056", "0059", "0061", "0062", "0064", "0066", "0067", "0068", "0070", "0071", "0072", "0073", "0074", "0075", "0077", "0078", "0079", "0081", "0083", "0085", "0086", "0088", "0089", "0091", "0092", "0094", "0095", "0097", "0099", "0100", "0101", "0102", "0104", "0105", "0106", "0107", "0109", "0111", "0112", "0114", "0115", "0116", "0119", "0120", "0122", "0123", "0124", "0126", "0127", "0128", "0130", "0131", "0133", "0134", "0139", "0140", "0141", "0145", "0147", "0148", "0149", "0150", "0153", "0154", "0155", "0157", "0158", "0159", "0160", "0162", "0167", "0168", "0169", "0170", "0172", "0176", "0177", "0179", "0180", "0181", "0182", "0183", "0184", "0185", "0187", "0188", "0190", "0191", "0192", "0193", "0194", "0195", "0196", "0198", "0199", "0202", "0203", "0204", "0205", "0206", "0207", "0208", "0209", "0210", "0212", "0213", "0214", "0215", "0216", "0217", "0218", "0220", "0221", "0222", "0223", "0224", "0227", "0228", "0229", "0231", "0234", "0235", "0236", "0238", "0239", "0240", "0241", "0242", "0244", "0245", "0246", "0247", "0248", "0249", "0250", "0251", "0253", "0254", "0256", "0257", "0258", "0259", "0260", "0261", "0263", "0264", "0265", "0267", "0268", "0269", "0270", "0271", "0275", "0276", "0278", "0279", "0280", "0281", "0282", "0284", "0288", "0289", "0290", "0291", "0292", "0293", "0294", "0295", "0296", "0297", "0300", "0302", "0303", "0304", "0306", "0307", "0308", "0309", "0310", "0312", "0314", "0315", "0317", "0318", "0319", "0320", "0322", "0323", "0324", "0325", "0326", "0329", "0330", "0331", "0332", "0334", "0335", "0336", "0338", "0339", "0341", "0343", "0345", "0346", "0347", "0348", "0349", "0351", "0352", "0353", "0354", "0355", "0356", "0358", "0359", "0360", "0361", "0362", "0364", "0366", "0367", "0369", "0370", "0371", "0372", "0377", "0378", "0380", "0381", "0382", "0383", "0384", "0385", "0386", "0387", "0389", "0390", "0391", "0393", "0395", "0396", "0397", "0399", "0400", "0401", "0402", "0403", "0404", "0406", "0407", "0410", "0412", "0414", "0415", "0417", "0418", "0419", "0421", "0425", "0426", "0427", "0429", "0430", "0432", "0433", "0434", "0439", "0442", "0443", "0444", "0445", "0446", "0448", "0450", "0451", "0452", "0454", "0455", "0456", "0457", "0458", "0461", "0463", "0464", "0465", "0467", "0468", "0469", "0471", "0472", "0473", "0474", "0475", "0476", "0477", "0478", "0479", "0480", "0481", "0482", "0483", "0484", "0485", "0490", "0492", "0494", "0497", "0500", "0501", "0502", "0503", "0504", "0506", "0507", "0508", "0510", "0511", "0512", "0515", "0516", "0517", "0518", "0521", "0522", "0524", "0525", "0526", "0528", "0530", "0531", "0533", "0537", "0539", "0540", "0542", "0543", "0545", "0546", "0547", "0549", "0550", "0552", "0553", "0554", "0555", "0559", "0560", "0561", "0562", "0565", "0567", "0568", "0570", "0571", "0572", "0573", "0574", "0577", "0578", "0579", "0580", "0582", "0583", "0585", "0587", "0588", "0589", "0590", "0591", "0592", "0594", "0596", "0597", "0598", "0599", "0600", "0601", "0603", "0605", "0606", "0607", "0608", "0610", "0614", "0615", "0616", "0617", "0618", "0619", "0620", "0621", "0623", "0625", "0627", "0628", "0629", "0631", "0633", "0634", "0635", "0636", "0637", "0638", "0640", "0643", "0645", "0647", "0648", "0649", "0650", "0651", "0654", "0655", "0656", "0657", "0658", "0659", "0660", "0662", "0663", "0664", "0668", "0669", "0670", "0671", "0672", "0674", "0675", "0677", "0678", "0680", "0681", "0682", "0683", "0684", "0685", "0687", "0688", "0689", "0690", "0691", "0692", "0695", "0696", "0698", "0699", "0700", "0701", "0705", "0706", "0707", "0708", "0709", "0711", "0713", "0716", "0720", "0721", "0722", "0723", "0725", "0726", "0727", "0728", "0729", "0730", "0731", "0733", "0734", "0735", "0736", "0737", "0740", "0741", "0742", "0743", "0746", "0747", "0748", "0749", "0751", "0752", "0755", "0757", "0758", "0759", "0761", "0762", "0765", "0766", "0767", "0768", "0769", "0770", "0772", "0773", "0775", "0776", "0777", "0779", "0780", "0781", "0782", "0783", "0785", "0787", "0788", "0789", "0790", "0791", "0792", "0793", "0794", "0795", "0796", "0797", "0798", "0800", "0802", "0803", "0804", "0805", "0806", "0807", "0809", "0810", "0811", "0812", "0813", "0814", "0816", "0818", "0819", "0821", "0822", "0823", "0824", "0827", "0828", "0830", "0831", "0832", "0833", "0834", "0837", "0838", "0839", "0841", "0842", "0843", "0844", "0845", "0846", "0847", "0848", "0849", "0850", "0852", "0853", "0856", "0857", "0859", "0861", "0863", "0864", "0865", "0870", "0872", "0873", "0874", "0875", "0879", "0881", "0882", "0884", "0885", "0886", "0887", "0888", "0889", "0890", "0891", "0893", "0894", "0895", "0896", "0898", "0900", "0901", "0902", "0903", "0904", "0905", "0909", "0910", "0911", "0913", "0914", "0915", "0916", "0917", "0920", "0923", "0925", "0926", "0927", "0928", "0929", "0932", "0933", "0935", "0940", "0941", "0943", "0944", "0945", "0946", "0947", "0948", "0950", "0952", "0954", "0956", "0958", "0961", "0963", "0964", "0965", "0967", "0968", "0969", "0970", "0973", "0974", "0978", "0979", "0981", "0982", "0984", "0985", "0986", "0987", "0988", "0990", "0991", "0992", "0993", "0995", "0996", "0999", "1002", "1003", "1006", "1007", "1009", "1011", "1012", "1013", "1014", "1016", "1018", "1019", "1020", "1021", "1022", "1024", "1025", "1027", "1031", "1032", "1033", "1034", "1037", "1039", "1040", "1041", "1042", "1043", "1044", "1045", "1046", "1048", "1051", "1052", "1053", "1054", "1055", "1056", "1058", "1059", "1061", "1062", "1063", "1064", "1066", "1067", "1068", "1069", "1070", "1071", "1072", "1073", "1076", "1078", "1082", "1084", "1085", "1086", "1087", "1089", "1091", "1093", "1094", "1096", "1097", "1098", "1099", "1100", "1101", "1102", "1103", "1104", "1106", "1107", "1109", "1110", "1111", "1112", "1113", "1114", "1116", "1117", "1118", "1119", "1120", "1121", "1123", "1124", "1125", "1126", "1127", "1132", "1133", "1134", "1135", "1136", "1138", "1139", "1140", "1141", "1142", "1143", "1145", "1146", "1147", "1148", "1149", "1150", "1151", "1152", "1153", "1154", "1155", "1157", "1160", "1161", "1162", "1163", "1164", "1166", "1167", "1168", "1169", "1170", "1171", "1173", "1174", "1177", "1178", "1180", "1181", "1182", "1183", "1186", "1188", "1190", "1192", "1193", "1194", "1197", "1200", "1201", "1203", "1204", "1208", "1210", "1211", "1212", "1213", "1217", "1218", "1220", "1221", "1222", "1223", "1225", "1226", "1227", "1228", "1229", "1232", "1234", "1235", "1237", "1239", "1240", "1242", "1244", "1245", "1246", "1247", "1248", "1249", "1250", "1253", "1256", "1257", "1258", "1260", "1261", "1263", "1265", "1266", "1267", "1268", "1269", "1270", "1271", "1272", "1273", "1274", "1276", "1277", "1279", "1280", "1281", "1282", "1283", "1285", "1286", "1287", "1289", "1290", "1292", "1296", "1297", "1300"]

  np.random.seed(-2)
  np.random.shuffle(samples)

  train_pct = .9
  train_qty = int(len(samples) * train_pct)
  test_qty = len(samples) - train_qty
  train_set = samples[:train_qty]
  test_set = samples[train_qty:train_qty + test_qty]
  if set_type == "train":
    return train_set
  elif set_type == "test":
    return test_set
  else:
    raise ValueError("Unknown set type")

def mcgill_preprocess(sample_range=1301,
                      working_dir="",
                      output_list=True,
                      max_frame_num=15122,
                      data_set_type="train",
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
      data_set_type : "train" or "test", fed into get_set
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

  # The note A# is equivalent to Bb, etc. 
  #
  # The McGill dataset seems to contain quite a few of these overlaps. Without
  # removing these overlaps, we will have 38 chord labels when it should be 25 =
  # maj/min of each of the 12 notes plus one N denoting "no chord."
  # 
  # There's also label 'X', denoting chords outside of the 24 maj/min, which for 
  # the sake of simplicity we simply map to "N" = no chord.
  repeated_chords = {
      "A#:maj" : "Bb:maj",
      "Cb:maj" : "B:maj",
      "Cb:min" : "B:min",
      "C#:maj" : "Db:maj",
      "C#:min" : "Db:min",
      "D#:maj" : "Eb:maj",
      "D#:min" : "Eb:min",
      "Fb:maj" : "E:maj",
      "F#:maj" : "Gb:maj",
      "F#:min" : "Gb:min",
      "G#:maj" : "Ab:maj",
      "G#:min" : "Ab:min",
      "X" : "N",
  }
  
  song_num = []
  if output_list:
    chroma = []
    chord = []
  else:
    chroma = np.zeros((0, max_frame_num, 25))
    chord = np.zeros((0, max_frame_num, 2))
  
  true_samples = 0
  # for sample_num in range(sample_range):
  #   chroma_dir = os.path.join(
  #       working_dir, chroma_base,'{:0>4}'.format(sample_num))
  #   chord_dir = os.path.join(
  #       working_dir, chord_base,'{:0>4}'.format(sample_num))
  #   if not (os.path.isdir(chroma_dir) and os.path.isdir(chord_dir)):
  #       continue

  for sample_str in get_set(data_set_type):
    chroma_dir = os.path.join(working_dir, chroma_base, sample_str)
    chord_dir = os.path.join(working_dir, chord_base, sample_str)
    
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
      chord_label_str = chord_dat[2][i]
      # Remove repeated chords. C# = Db, G# = Ab etc.
      if chord_label_str in repeated_chords:
        chord_label_str = repeated_chords[chord_label_str]
      if chord_label_str not in dict_chord2idx:
        dict_chord2idx.update({chord_label_str: idx_chord})
        dict_idx2chord.update({idx_chord: chord_label_str})
        idx_chord +=1
      
    chord_label = np.zeros((chroma_dat.shape[0], 2))
    for i in range(chord_dat.shape[0]):
      sel_time = np.logical_and(
          chroma_dat[1] >= chord_dat[0][i],
          chroma_dat[1] < chord_dat[1][i])
      chord_label_str = chord_dat[2][i]
      # Remove repeated chords.
      if chord_label_str in repeated_chords:
        chord_label_str = repeated_chords[chord_label_str]
      chord_label[sel_time, 1] = dict_chord2idx[chord_label_str]
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


def preprocess_data_and_store(
    input_dir, output_dir, data_set_type, output_list=True, verbose=False):
  """Preprocess & store the McGill data once and for all.

  Args:
    intput_dir: full path to where McGill_Billboard dataset is.
    output_dir: where to store the output. Must already exist.
    output_list: see mcgill_preprocess(), store output as Python list or 
        numpy array.
    verbose: Boolean, if True, will print progress to stdout.

  TODO(elizachu): stop hard-coding the directory paths. Use optparse.
  """

  # TODO(elizachu): open a file and write periodically in this function, instead
  # of waiting for all songs to be processed then saving them. If program is
  # interrupted, we risk saving corrupted data.
  chroma, chord, chord2index, index2chord, song_num = mcgill_preprocess(
      working_dir=input_dir,
      output_list=output_list,
      data_set_type=data_set_type,
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
    if output_list:
      print("len(chroma):", len(chroma))
      print("len(chord):", len(chord))
    else:
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

  chroma = np.load(os.path.join(input_dir, "chroma.npy"))

  # chroma stores python List of length m, m = number of songs.
  # Each item in List is a matrix of shape (num_frames, 25) for 25 chords.
  # song_lengths[i] returns the number of frames in song i.
  song_lengths = [chroma_data_matrix.shape[0] for chroma_data_matrix in chroma]

  print("song_lengths[0:10]:", song_lengths[0:10])
  np.save(os.path.join(output_dir, "song_lengths.npy"), np.array(song_lengths))

  print("Saved to song_lengths.npy")

def main(unused_argv=None):

  ############################# MODIFY FLAGS HERE ############################

  # working_dir should be full path to where McGill_Billboard dataset is.
  working_dir = ("/Users/charleschen/Documents/Courses/CS230/Project/dataset/"
                 "McGill_Billboard")

  # output_dir needs to already exist; full path to directory to save .npy
  # output files.
  test_output_dir = ("/Users/charleschen/Documents/Courses/CS230/Project/dataset/"
                     "preprocessed_mcgill_matrix_test")
  train_output_dir = ("/Users/charleschen/Documents/Courses/CS230/Project/dataset/"
                      "preprocessed_mcgill_matrix_train")

  # See def mcgill_preprocess. Whether to store output data in Python List or
  # zero-pad with numpy arrays.
  output_list = False

  ############################### END FLAGS ##################################

  preprocess_data_and_store(
      input_dir=working_dir,
      output_dir=test_output_dir,
      output_list=output_list,
      data_set_type="test",
      verbose=True,
  )

  extract_song_lengths(input_dir=output_dir, output_dir=test_output_dir)

  preprocess_data_and_store(
      input_dir=working_dir,
      output_dir=train_output_dir,
      output_list=output_list,
      data_set_type="train",
      verbose=True,
  )

  extract_song_lengths(input_dir=output_dir, output_dir=train_output_dir)


if __name__ == "__main__":
  main()
