"""
TensorFlow dataset
"""
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from functools import partial
from load_data import make_image_gen
from load_data import get_balanced_train_test
from load_data import get_unique_img_ids

class Dataset():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.args = args
        masks = pd.read_csv(os.path.join(args.dataset_dir, args.train_masks_csv))
        unique_img_ids = get_unique_img_ids(masks, args)
        train_df, valid_df = get_balanced_train_test(masks, unique_img_ids, args)
        train_gen = partial(make_image_gen, df=train_df, args=args)
        valid_gen = partial(make_image_gen, df=valid_df, args=args)
        tf.set_random_seed(args.seed)

        self.train_dataset = tf.data.Dataset.from_generator(
            train_gen, output_types=(tf.float32, tf.int64),
            output_shapes=
                (tf.TensorShape([args.size, args.size, 3]),
                tf.TensorShape([args.size, args.size, 1])))

        self.train_dataset = self.train_dataset.map(map_func=self.data_aug_fn, num_parallel_calls=args.num_parallel_calls)
        self.train_dataset = self.train_dataset.batch(batch_size=args.batch_size)
        self.train_dataset = self.train_dataset.prefetch(1)
        
        self.valid_dataset = tf.data.Dataset.from_generator(
            valid_gen, output_types=(tf.float32, tf.int64),
            output_shapes=
                (tf.TensorShape([args.size, args.size, 3]),
                tf.TensorShape([args.size, args.size, 1])))

        # self.valid_dataset = self.valid_dataset.map(map_func=self.data_aug_fn, num_parallel_calls=args.num_parallel_calls)
        self.valid_dataset = self.valid_dataset.batch(args.valid_img_count)
        
        self.iter = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                    self.train_dataset.output_shapes)
        
        self.train_init_op = self.iter.make_initializer(self.train_dataset)
        self.valid_init_op = self.iter.make_initializer(self.valid_dataset)


    def data_aug_fn(self, image, mask_as_img):
        """
        data augment function used in in tensorflow data map function
        """
        image = tf.contrib.image.rotate(image, tf.random_uniform([1], maxval=0.25*math.pi))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, self.args.brightness)

        return image, mask_as_img


if __name__ == "__main__":
    ### test whether the dataset work as expected
    import argparse
    parser = argparse.ArgumentParser(description="parser of airbus ship competition project")
    parser.add_argument("--dataset_dir", type=str, default="/media/gerry/Data_2/kaggle_airbus_data", help="root directory of dataset")
    parser.add_argument("--train_masks_csv", type=str, default='train_ship_segmentations.csv', help="train masks csv name")
    parser.add_argument('--train_img_dir', type=str, default="train", help="train image dir")
    parser.add_argument("--size", type=int, default=768//4, help="image size down sampled")
    parser.add_argument('--debug', type=bool, default=True, help="debug?")
    parser.add_argument('--samples_per_ship_group', type=int, default=2000, help="upper bound of number of ships per group")
    parser.add_argument('--train_valid_ratio', type=float, default=0.3, help="split ratio")
    parser.add_argument("--img_scaling", type=tuple, default=(4,4), help="downsampling during preprocessing")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_parallel_calls", type=int, default=16, help="num of parallel calls when preprocessing image")
    parser.add_argument("--valid_img_count", type=int, default=600, help="number of validation images in one batch to use")
    parser.add_argument("--brightness", type=float, default=0.5, help="max delta augment of the img brightness")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()

    t_dataset = Dataset(args)
    features, labels = t_dataset.iter.get_next()

    t_epoch = 2
    with tf.Session() as sess:
        print("training ...")
        for _ in range(t_epoch):
            sess.run(t_dataset.train_init_op)
            f, l = sess.run([features, labels])
            print(f.shape)
            print(l.shape)
        print("validating ... ")
        sess.run(t_dataset.valid_init_op)
        f, l = sess.run([features, labels])
        print(f.shape)
        print(l.shape)
