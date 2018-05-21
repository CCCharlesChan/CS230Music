# CS230Music
Class project of CS230 Chord recognition group

To start training, first preprocess data and store it in some existing
directory:
1. Modify working_dir and output_dir in `mcgill_preprocessing.py`, `def main()`
1. Also modify line 146, change `output_list = False` to `output_list = True`.
1. Then run:
    ```shell
    $ python mcgill_preprocessing.py
    ```

Need to rerun `mcgill_preprocessing.py` to generate file `song_legnths.npy`:
1. Modify `mcgill_preprocessing.py` line 146, change to `output_list = False`
1. Modify `def main()`, change `list_input_dir` to where you saved the output
   in the previous step (= output_dir).
1. Run:
    ```shell
    $ python mcgill_preprocessing.py
    ```

Lastly, start training:
```shell
$ python main.py --input_data_dir=/full/path/to/mcgill_preprocessing/output
```


