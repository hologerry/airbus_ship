"""
Data loading helper functions
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def get_unique_img_ids(masks, args):
    """
    masks: all masks DataFrame
    Return: unique_img_ids DataFrame "ImageId  ships  has_ships has_ships_vec  img_file_size_kb"
    """
    masks = masks[masks.ImageId != '6384c3e78.jpg']
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ships'] = unique_img_ids['ships'].map(lambda x: 1 if x > 0 else 0)
    unique_img_ids['has_ships_vec'] = unique_img_ids['has_ships'].map(lambda x: [x])

    unique_img_ids['img_file_size_kb'] = unique_img_ids['ImageId'].map(
        lambda c_img_id: os.stat(os.path.join(args.dataset_dir, args.train_img_dir, c_img_id)).st_size/1024)

    # only keep +50kb img files
    unique_img_ids = unique_img_ids[unique_img_ids['img_file_size_kb'] > 50]
    masks.drop(['ships'], axis=1, inplace=True)

    if args.debug:
        unique_img_ids.hist()
        unique_img_ids['ships'].hist(bins=unique_img_ids['ships'].max())
        print(unique_img_ids.head(7))

    return unique_img_ids


def get_balanced_train_valid(masks, unique_img_ids, args):
    """
    masks: training segment all masks
    unique_img_ids: returned by get_unique_img_ids func
    Return:
        train_df: ImageId EncodedPixels ships has_ships_vec
        valid_df:
    """
    SAMPLES_PER_GROUP = args.samples_per_ship_group
    ratio = args.train_valid_ratio
    balanced_train_df = unique_img_ids.groupby('ships').apply(
        lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

    if args.debug:
        balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)

    train_ids, valid_ids = train_test_split(balanced_train_df, test_size=ratio, stratify=balanced_train_df['ships'])
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)

    if args.debug:
        print(len(train_ids), "training images: ")
        print(train_ids.head())
        print(len(valid_ids), "validating images: ")
        print(train_ids.head())
        print(len(train_df), "training masks: ")
        print(train_df.head())
        print(len(valid_df), "validating masks: ")
        print(valid_df.head())

    return train_df, valid_df
