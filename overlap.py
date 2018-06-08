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

# def compare_mirexes(filename1, filename2):
#     a1 = mirex_to_array(filename1)
#     a2 = mirex_to_array(filename2)

#     union = []
#     for i, j in zip(a1, a2):
#         if i == j:

#     intersection = []
#     for i, j in zip(a1, a2):
#         if i == j:

def overlap_ratio(a, b):
    scale = 1/len(a)
    correct = 0
    for i, j in zip(a, b):
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


if __name__ == '__main__':
    import glob
    files = glob.glob('/home/c/CS230/Project/Data/McGill_Billboard/*/*/*.lab')
    # files = ['/home/c/CS230/Project/Data/McGill_Billboard/MIREX_style/0003/majmin.lab',
    #     '/home/c/CS230/Project/Data/McGill_Billboard/MIREX_style/0003/majmin7.lab',
    #     '/home/c/CS230/Project/Data/McGill_Billboard/MIREX_style/0003/majmin7inv.lab',
    #     '/home/c/CS230/Project/Data/McGill_Billboard/MIREX_style/0003/majmininv.lab',
    #     '/home/c/CS230/Project/Data/McGill_Billboard/LAB_files/0003/full.lab']
    arrays = map(mirex_to_array, files)
    for array in arrays:
        if '0' in array:
            print('!!!')


    # compare 2 files
    # compare to csv
    # weighting
    # generator tuesday