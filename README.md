# CS230Music
Class project of CS230 Chord recognition group

To run training, for example:

```
 $ python main.py --input_data_dir=/path/to/training/data --is_train \
   --num_epoch=100 --checkpoint_frequency=10 --num_hidden_units=200 \
   --learning_rate=0.003 &> log_train.txt
```

To run testing, for example:

```
$ python main.py --input_data_dir=/path/to/testing/data \
  --model_load_dir=/output/from/training/looks/like/rnngan_20180607_234205 \
  --model_load_meta_path=/same/folder/find/meta/file/rnngan_20180607_234205/-100.meta \
  &> log_test.txt
```

Note, you must first preprocess the data by running `mcgill_preprocessing`.
