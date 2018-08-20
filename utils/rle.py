"""
run length encoding helper functions
"""
import numpy as np
from skimage.morphology import label # Label connected region of an integer array

img_shape = (768, 768)

def rle_decode(mask_rle, shape=img_shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of arrary to return
    Return: numpy array, 1 - mask, 0 - background, order: top->bottom, left->right
    """
    s = mask_rle.split()
    starts, lengths = [np.array(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 1

    return img.reshape(shape).T


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Return: run length as string formated
    """
    pixels = img.T.flatten()    # left -> right  top -> bottom to top -> bottom left -> right
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def multi_rle_encode(img, **kwargs):
    """
    Encode connected regions as separated masks
    """
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]


def masks_as_color(in_mask_list):
    """
    in_mask_list: individual mask list in one image
    Return: a color mask array for each ship
    """
    all_masks = np.zeros(img_shape, dtype=np.float)
    ## scale the heatmap image to shift
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2)
    for i, mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks += scale(i) * rle_decode(mask)
    return all_masks


def masks_as_image(in_mask_list):
    """
    in_mask_list: individual mask list in one image
    Return: a single mask array for all ships in one image
    """
    all_masks = np.zeros(img_shape, dtype=np.float)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return all_masks


if __name__ == "__main__":
    
    # test whether the encode decode work as expected
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import gc

    masks = pd.read_csv('/media/gerry/Data_2/kaggle_airbus_data/train_ship_segmentations.csv')
    not_empty = masks.EncodedPixels.notna()
    print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')
    print((~not_empty).sum(), 'masks in', masks.ImageId.nunique(), 'total images')
    print(masks.head())
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    rle_0 = masks.query('ImageId == "00021ddc3.jpg"')['EncodedPixels']
    img_0 = masks_as_image(rle_0)
    axes[0].imshow(img_0)
    axes[0].set_title('Mask as image')
    
    rle_1 = multi_rle_encode(img_0)
    img_1 = masks_as_image(rle_1)
    axes[1].imshow(img_1)
    axes[1].set_title('Re-encode')
    
    img_c = masks_as_color(rle_0)
    axes[2].imshow(img_c)
    axes[2].set_title('Mask in color')
    
    img_c = masks_as_color(rle_1)
    axes[3].imshow(img_c)
    axes[3].set_title('Re-encode in color')

    print('Check decoding->encoding', 'RLE_0', len(rle_0), '->', 'RLE_1', len(rle_1))
    print(np.sum(img_0 - img_1), 'errors')
    plt.show()
    del masks
    gc.collect()

