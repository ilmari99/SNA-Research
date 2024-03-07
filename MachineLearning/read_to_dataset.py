import os
import random
from typing import Tuple
import tensorflow as tf


def read_to_dataset(paths, add_channel=False, shuffle_files=True, filter_files_fn = None) -> Tuple[tf.data.Dataset, int]:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    if filter_files_fn is None:
        filter_files_fn = lambda x: True
    
    # Find all files in paths, that fit the filter_files_fn
    file_paths = [os.path.join(path, file) for path in paths for file in os.listdir(path) if filter_files_fn(file)]
    if shuffle_files:
        random.shuffle(file_paths)

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
    
    
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.interleave(ds_maker,
                                cycle_length=tf.data.experimental.AUTOTUNE,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                deterministic=False)

    # Add a channel dimension if necessary
    if add_channel:
        dataset = dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset, len(file_paths)