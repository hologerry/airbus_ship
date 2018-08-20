"""
Data loading helper functions
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from skimage.io import imread

from rle import masks_as_image

def get_unique_img_ids(masks, args):
    """
    masks: all masks DataFrame
    Return: unique_img_ids DataFrame "ImageId  ships  has_ships has_ships_vec  img_file_size_kb"
    """
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ships'] = unique_img_ids['ships'].map(lambda x: 1 if x>0 else 0)
    unique_img_ids['has_ships_vec'] = unique_img_ids['has_ships'].map(lambda x: [x])

    unique_img_ids['img_file_size_kb'] = unique_img_ids['ImageId'].map(lambda 
        c_img_id: os.stat(os.path.join(args.train_img_dir, c_img_id)).st_size/1024)

    ## only keep +50kb img files
    unique_img_ids = unique_img_ids[unique_img_ids['img_file_size_kb'] > 50]
    masks.drop(['ships'], axis=1, inplace=True)

    if args.debug:
        unique_img_ids.hist()
        unique_img_ids['ships'].hist(bins=unique_img_ids['ships'].max())
        print(unique_img_ids.head(7))

    return unique_img_ids


def get_balanced_train_test(masks, unique_img_ids, args):
    """
    masks: training segment all masks
    unique_img_ids: returned by get_unique_img_ids func
    Return:
        train_df: ImageId EncodedPixels ships has_ships_vec
        valid_df:
    """
    SAMPLES_PER_GROUP = args.samples_per_ship_group
    ratio = args.train_valid_ratio
    balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: 
        x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

    if args.debug:
        balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)
    
    train_ids, valid_ids = train_test_split(balanced_train_df, test_size=ratio, stratify=balanced_train_df['ships'])
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)

    if args.debug:
        print(train_df.head())
        print(len(train_df), "training masks")
        print(valid_df.head())
        print(len(valid_df), "testing masks")

    return train_df, valid_df


def make_image_gen(df, args):
    """Image and mask generator for tensorflow dataset
    df: train_df or valid_df DataFrame
    Yield: single (one batch) rgb, mask batch
    """
    all_batches = list(df.groupby('ImageId'))
    ### for one batch
    # out_rgb = []
    # out_mask = []
    while True:
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(args.train_img_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            if args.img_scaling is not None:
                c_img = c_img[::args.img_scaling[0], ::args.img_scaling[1]]
                c_mask = c_mask[::args.img_scaling[0], ::args.img_scaling[1]]
            ### single sample
            yield c_img/255.0, c_mask
            
            ### one batch
            # out_rgb += [c_img]
            # out_mask += [c_mask]
            # if len(out_rgb) >= args.batch_size:
            #     yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
            #     out_rgb, out_mask=[], []


if __name__ == '__main__':
    ## test whether work
    import gc
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="parser of airbus ship competition project")
    parser.add_argument('--train_img_dir', type=str, default="/media/gerry/Data_2/kaggle_airbus_data/train")
    parser.add_argument('--debug', type=bool, default=True, help="debug?")
    parser.add_argument('--samples_per_ship_group', type=int, default=2000, help="upper bound of number of ships per group")
    parser.add_argument('--train_valid_ratio', type=float, default=0.3, help="split ratio")
    masks = pd.read_csv('/media/gerry/Data_2/kaggle_airbus_data/train_ship_segmentations.csv')
    args = parser.parse_args()
    
    unique = get_unique_img_ids(masks, args)
    train, valid = get_balanced_train_test(masks, unique, args)
    plt.show()
    
    del masks
    gc.collect()