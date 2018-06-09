from __future__ import division
import numpy as np

from multiprocessing import Pool

# mirex format + frame -> array[index] == chord

dt = 2048 / 44100
scale = lambda i: int(round(i / dt))

def mirex_lines_from_file(filename):
    mirex_file = open(filename, 'r')
    mirex_lines = mirex_file.readlines()
    mirex_lines = (line for line in mirex_lines if line.strip() != "")
    return mirex_lines    

def parse_line(line):
    start_str, end_str, chord = line.split()
    start, end = map(float, [start_str, end_str])
    start_index, end_index = map(scale, [start, end])
    return list(range(start_index, end_index+1)), chord

def create_array(indices, chords):
    array = ['0'] * (indices[-1][-1] + 1)
    for index_list, chord in zip(indices, chords):
        for index in index_list:
            array[index] = chord

    return array

def mirex_to_array(filename):
    mirex_lines = mirex_lines_from_file(filename)
    indices, chords = zip(*map(parse_line, mirex_lines))
    return create_array(indices, chords)

def compare_mirexes(filename1, filename2):
    a1 = mirex_to_array(filename1)
    a2 = mirex_to_array(filename2)

    return overlap

def overlap_ratio(a, b, length):
    scale = 1/length
    correct = 0
    for i, j in zip(a, b)[:length]:
        if i == j:
            correct += 1
    return scale * correct

def weighted_overlap_ratio(a, b):
    # ?
    pass


def runlength_extract(a):
    current = a[0]
    val = 0
    res = []
    for el in a:
        if current == el:
            val += 1
        else:
            current = el
            val = 1
        res.append(val)
    current = res[-1]
    for el, i in zip(res[::-1], range(len(res))[::-1]):
        if current == -1:
            current = el
        res[i] = current
        if el == 1:
            current = -1
    return res

npy_loaded = {}
def npy_file(path):
    if path in npy_loaded:
        return npy_loaded[path]
    else:
        npy = np.load(path)
        npy_loaded[path] = npy
        return npy

idx_to_song = ['0811', '0064', '0891', '0253', '0107', '0973', '0913', '0824', '0964']

def idx_array_to_chord_array(idx_array, index2chord):
    out = []
    for i in idx_array:
        out.append(index2chord[i])
    return out


def preds_to_array(pred_filename, index2chord_filename, song_idx):
    preds = npy_file(pred_filename)[song_idx]
    index2chord = npy_file(index2chord_filename).item()

    chords = idx_array_to_chord_array(preds, index2chord)
    return chords

def chords_to_array(pred_filename, index2chord_filename, song_idx):
    preds = npy_file(pred_filename)[song_idx,:,1]
    index2chord = npy_file(index2chord_filename).item()

    chords = idx_array_to_chord_array(preds, index2chord)
    return chords

if __name__ == '__main__':
    import glob
    files = glob.glob('/home/c/CS230/Project/Data/McGill_Billboard/*/*/*.lab')
    # files = ['/home/c/CS230/Project/Data/McGill_Billboard/MIREX_style/0003/majmin.lab',
    #     '/home/c/CS230/Project/Data/McGill_Billboard/MIREX_style/0003/majmin7.lab',
    #     '/home/c/CS230/Project/Data/McGill_Billboard/MIREX_style/0003/majmin7inv.lab',
    #     '/home/c/CS230/Project/Data/McGill_Billboard/MIREX_style/0003/majmininv.lab',
    #     '/home/c/CS230/Project/Data/McGill_Billboard/LAB_files/0003/full.lab']
    # arrays = map(mirex_to_array, files)
    # for array in arrays:
    #     if '0' in array:
    #         print('!!!')

    pred_filename = "/home/ubuntu/CS230Music/rnngan_20180609_082130/predictions.npy"
    index2chord_validation = "/home/ubuntu/McGill_Billboard_matrix_train/index2chord.npy"
    index2chord_train = "/home/ubuntu/Data/McGill_Billboard_test/index2chord.npy"
    chord_filename = "/home/ubuntu/McGill_Billboard_matrix_train/chord.npy"
    song_lengths_filename = "/home/ubuntu/McGill_Billboard_matrix_train/song_lengths.npy"

    for i in range(len(idx_to_song)):
        file = '/home/c/CS230/Project/Data/McGill_Billboard/MIREX_style/{}/majmin.lab'.format(idx_to_song[i])
        pred_array = preds_to_array(pred_filename, index2chord_validation, i)
        chord_array = chords_to_array(chord_filename, index2chord_train, i)

        length = npy_file(song_lengths_filename)[i]


        print(overlap_ratio(chord_array, pred_array, length))
