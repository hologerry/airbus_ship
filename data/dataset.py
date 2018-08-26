"""
TensorFlow dataset
"""
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from functools import partial
from data.load_data import make_image_gen
from data.load_data import make_image_gen_test
from data.load_data import get_balanced_train_valid
from data.load_data import get_unique_img_ids

class Dataset():
    def __init__(self, args):
        self.args = args

        if self.args.subcommand == "train":
            self.batch_size = args.batch_size
            masks = pd.read_csv(os.path.join(args.dataset_dir, args.train_masks_csv))
            unique_img_ids = get_unique_img_ids(masks, args)
            train_df, valid_df = get_balanced_train_valid(masks, unique_img_ids, args)
            train_gen = partial(make_image_gen, df=train_df, args=args)
            valid_gen = partial(make_image_gen, df=valid_df, args=args)

            self.train_dataset = tf.data.Dataset.from_generator(
                train_gen, output_types=(tf.float32, tf.float32),
                output_shapes=
                    (tf.TensorShape([args.size, args.size, 3]),
                    tf.TensorShape([args.size, args.size, 1]))).repeat()

            ## used for data augment
            np.random.seed(args.seed)

            self.train_dataset = self.train_dataset.map(map_func=self.data_aug_fn, num_parallel_calls=args.num_parallel_calls)
            self.train_dataset = self.train_dataset.batch(batch_size=args.batch_size)
            self.train_dataset = self.train_dataset.prefetch(1)

            self.valid_dataset = tf.data.Dataset.from_generator(
                valid_gen, output_types=(tf.float32, tf.float32),
                output_shapes=
                    (tf.TensorShape([args.size, args.size, 3]),
                    tf.TensorShape([args.size, args.size, 1])))

            # self.valid_dataset = self.valid_dataset.map(map_func=self.data_aug_fn, num_parallel_calls=args.num_parallel_calls)
            self.valid_dataset = self.valid_dataset.batch(args.valid_batch_size)

            self.iter = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                        self.train_dataset.output_shapes)

            self.train_init_op = self.iter.make_initializer(self.train_dataset)
            self.valid_init_op = self.iter.make_initializer(self.valid_dataset)

        elif self.args.subcommand == "test":
            test_gen = partial(make_image_gen_test, args=args)

            self.test_dataset = tf.data.Dataset.from_generator(
                test_gen, output_types=(tf.float32, tf.string),
                output_shapes=(tf.TensorShape([args.size, args.size, 3]), tf.TensorShape([]))
            )
            self.test_dataset = self.test_dataset.batch(args.test_batch_size)

            self.test_iter = tf.data.Iterator.from_structure(self.test_dataset.output_types,
                                                            self.test_dataset.output_shapes)
            self.test_init_op = self.test_iter.make_initializer(self.test_dataset)


    def data_aug_fn(self, image, mask_as_img):
        """
        data augment function used in in tensorflow data map function
        """
        rand = np.random.random_sample()
        if rand > 0.5:
            ag = tf.random_uniform([1], maxval=0.25*math.pi)
            image = tf.contrib.image.rotate(image, ag)
            mask_as_img = tf.contrib.image.rotate(mask_as_img, ag)
        rand = np.random.random_sample()
        if rand > 0.5:
            image = tf.image.flip_left_right(image)
            mask_as_img = tf.image.flip_left_right(mask_as_img)
        rand = np.random.random_sample()
        if rand > 0.5:
            image = tf.image.flip_up_down(image)
            mask_as_img = tf.image.flip_up_down(mask_as_img)

        ## TODO: more image augments

        return image, mask_as_img


if __name__ == "__main__":
    ### test whether the dataset work as expected
    print("Testing dataset ...")
    import argparse
    import matplotlib.pyplot as plt
    import gc

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
    parser.add_argument("--valid_img_count", type=int, default=100, help="number of validation images in one batch to use")
    parser.add_argument("--brightness", type=float, default=0.5, help="max delta augment of the img brightness")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()

    t_dataset = Dataset(args)

    features, labels = t_dataset.iter.get_next()
    t_epoch = 3
    with tf.Session() as sess:
        print("training ...")
        sess.run(t_dataset.train_init_op)
        for _ in range(t_epoch):
            print("one epoch ...")
            cnt = 0
            while True:
                try:
                    f, l = sess.run([features, labels])
                    print("x", f.shape, f.dtype, f.min(), f.max())
                    print("y", l.shape, l.dtype, l.min(), l.max())
                    f = f[0]
                    l = l[0].reshape([args.size, args.size])
                    print("f", f)
                    print("l", l)
                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    # ax1.imshow(f)
                    # ax1.set_title("images")
                    # ax2.imshow(l, cmap="gray_r")
                    # ax2.set_title("ships")
                    # plt.show()

                    cnt += 1
                    if (cnt > 3):
                        break
                except:
                    print(cnt, "batchs in one epoch")
                    break

        print("validating ... ")
        sess.run(t_dataset.valid_init_op)
        f, l = sess.run([features, labels])
        print("x", f.shape, f.dtype, f.min(), f.max())
        print("y", l.shape, l.dtype, l.min(), l.max())

        gc.collect()
