"""PyTorch dataset
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from torchvision.transforms import Compose, ToTensor, Normalize

from data.rle import masks_as_image


class ShipDataset(Dataset):
    def __init__(self, in_df, args, transform=None, mode='train'):
        self.args = args
        grouplist = list(in_df.groupby('ImageId'))
        self.image_ids = [_id for _id, _ in grouplist]
        self.image_masks = [m['EncodedPixels'].values for _, m in grouplist]
        # data augment transform
        self.transform = transform
        self.mode = mode

        # basic tranform for iamge
        self.image_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_file_name = self.image_ids[index]

        if self.mode == 'train':
            rgb_path = os.path.join(self.args.dataset_dir, self.args.train_img_dir, img_file_name)
        else:
            rgb_path = os.path.join(self.args.dataset_dir, self.args.test_img_dir, img_file_name)
        img = imread(rgb_path)
        mask = masks_as_image(self.image_masks[index])

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        if self.mode == 'train':
            return self.image_transform(img), torch.from_numpy(np.moveaxis(mask, -1, 0)).float()
        else:
            return self.image_transform(img), str(img_file_name)


def make_dataloader(in_df, args, batch_size, shuffle=False, transform=None, mode='train'):
    return DataLoader(dataset=ShipDataset(in_df, args, transform=transform, mode=mode),
                      shuffle=shuffle, num_workers=0, batch_size=batch_size, pin_memory=torch.cuda.is_available())
