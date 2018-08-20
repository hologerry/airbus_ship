"""
TensorFlow dataset
"""
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from load_data import make_image_gen
from load_data import get_balanced_train_test
from load_data import get_unique_img_ids

class Dataset():
    def __init__(self, args):
        self.batch_size = args.batch_size

        masks = pd.read_csv(os.path.join(args.dataset_dir, args.train))
        unique_img_ids = get_unique_img_ids(masks, args)
        train_df, valid_df = get_balanced_train_test(masks, unique_img_ids, args)
        train_gen = make_image_gen(train_df, args)
        valid_gen = make_image_gen(valid_df, args)

        self.train_dataset = tf.data.Dataset.from_generator(
            train_gen, output_types=(tf.float32, tf.int64),
            output_shapes=
            (tf.TensorShape([None, args.size, args.size, 3]),
                tf.TensorShape([None, args.size, args.size, 1]))).batch(self.batch_size)
        
        self.valid_dataset = tf.data.Dataset.from_generator(
            valid_gen, output_types=(tf.float32, tf.int64),
            output_shapes=
            (tf.TensorShape([None, args.size, args.size, 3]),
                tf.TensorShape([None, args.size, args.size, 1]))).batch(self.batch_size)
        
        self.iter = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                    self.train_dataset.output_shapes)
        
        self.train_init_op = self.iter.make_initializer(self.train_dataset)
        self.valid_init_op = self.iter.make_initializer(self.valid_dataset)
    
    