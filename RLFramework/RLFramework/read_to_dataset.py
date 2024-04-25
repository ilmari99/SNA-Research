import os
import random
from typing import Tuple
import warnings
import tensorflow as tf


def read_to_dataset(paths,
                    frac_test_files=0,
                    add_channel=False,
                    shuffle_files=True,
                    filter_files_fn = None) -> Tuple[tf.data.Dataset, int, int]:
    """ Create a tf dataset from a folder of files.
    If split_files_to_test_set is True, then frac_test_files of the files are used for testing.
    
    """
    assert 0 <= frac_test_files <= 1, "frac_test_files must be between 0 and 1"
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    if filter_files_fn is None:
        filter_files_fn = lambda x: True
    
    # Find all files in paths, that fit the filter_files_fn
    file_paths = [os.path.join(path, file) for path in paths for file in os.listdir(path) if filter_files_fn(file)]
    if shuffle_files:
        random.shuffle(file_paths)
        
    # Read one file to get the number of samples in a file
    with open(file_paths[0], "r") as f:
        num_samples = sum(1 for line in f)

    print("Found {} files".format(len(file_paths)))
    def txt_line_to_tensor(x):
        s = tf.strings.split(x, sep=",")
        s = tf.strings.to_number(s, out_type=tf.float32)
        return (s[:-1], s[-1])

    def ds_maker(x):
        ds = tf.data.TextLineDataset(x, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        ds = ds.map(txt_line_to_tensor,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=False)
        return ds
    
    test_files = file_paths[:int(frac_test_files*len(file_paths))]
    train_files = file_paths[int(frac_test_files*len(file_paths)):]
    
    if len(test_files) > 0:
        test_ds = tf.data.Dataset.from_tensor_slices(test_files)
        test_ds = test_ds.interleave(ds_maker,
                                cycle_length=tf.data.experimental.AUTOTUNE,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                deterministic=False)
        if add_channel:
            test_ds = test_ds.map(lambda x, y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    train_ds = train_ds.interleave(ds_maker,
                                cycle_length=tf.data.experimental.AUTOTUNE,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                deterministic=False)
    # Add a channel dimension if necessary
    if add_channel:
        train_ds = train_ds.map(lambda x, y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if len(test_files) > 0:
        return train_ds, test_ds, len(file_paths), num_samples*len(file_paths)
    return train_ds, len(file_paths), num_samples*len(file_paths)