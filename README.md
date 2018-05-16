# CS230Music
Class project of CS230 Chord recognition group

To start training, first preprocess data and store it in some existing directory:

```shell
# First, modify working_dir and output_dir in mcgill_preprocessing.py.
$ python mcgill_preprocessing.py
```

Then, run:

```shell
$ python main.py --input_data_dir=/full/path/to/mcgill_preprocessing/output
```


