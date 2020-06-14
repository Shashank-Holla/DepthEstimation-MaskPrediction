from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import glob
import os
from PIL import Image
from data.albumentations_transform import albumentation_compose

class schrodingersCatDataset(Dataset):
    '''
    Class provides dictionary containing bg, bg_fg, mask and depth images.

    Input:
    data_root - Path containing Image folders of bg, bg_fg, mask and depth images.
    train - Boolean flag to indicate if the dataset is for Train or test.
    train_split - Ratio of train to test dataset.
    transform - Boolean flag to apply transformations on input images (bg and bg_fg images).
    target_transform - Boolean flag to apply target transformation on output images (mask and depth images).

    Output:
    Dictionary containing images of quartet of images- bg_images , bg_fg_images , mask_images and depth_images
    '''

    # Class variable to get train and test split, stats for normalization.
    images_mean = {"bg_images": [0.68968, 0.65092, 0.60790], "bg_fg_images" : [0.68084, 0.64135, 0.59833], "mask_images":[0.06140], "depth_images":[0.49981]}
    images_std = {"bg_images": [0.18897, 0.20892, 0.23450], "bg_fg_images" : [0.19432, 0.21262, 0.23641], "mask_images":[0.23733], "depth_images":[0.27300]}

    indices = np.arange(400000)
    random.shuffle(indices)

    def __init__(self, data_root,train=True, train_split=0.7, transform=None, target_transform=None, resize=None):
        self.bg_fg_paths = glob.glob(os.path.join(data_root,'bg_fg','*.jpg'))
        self.bg_fg_paths.sort()
        self.bg_paths = glob.glob(os.path.join(data_root,'bg','*.jpg'))
        self.bg_paths.sort()
        self.mask_paths = glob.glob(os.path.join(data_root, 'mask','*.jpg'))
        self.mask_paths.sort()
        self.depth_paths = glob.glob(os.path.join(data_root, 'depth','*.jpg'))
        self.depth_paths.sort()

        self.train = train
        self.train_split = train_split
        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize

        split_id = int(len(self.indices) * self.train_split)
        if self.train:
            self.indices = self.indices[:split_id]
        else:
            self.indices = self.indices[split_id:]

        if self.transform:
            self.bg_transform = albumentation_compose('input', means=self.images_mean["bg_images"], stddev=self.images_std["bg_images"], resize=self.resize)
            self.bg_fg_transform = albumentation_compose('input', means=self.images_mean["bg_fg_images"], stddev=self.images_std["bg_fg_images"], resize=self.resize)
        
        if self.target_transform:
            self.mask_transform = albumentation_compose('target', resize=self.resize)
            self.depth_transform = albumentation_compose('target', resize=self.resize)            

        
    def __len__(self):
        'Total number of samples'
        return len(self.indices)
        

    def __getitem__(self, index):
        'getitem method is called one index at a time, even for a batch.'
        'Since bg images are repeated for 4000 for bg_fg and mask, divide index by 4000 to get bg image'
        image_id = self.indices[index]
        bg_fg_image = Image.open(self.bg_fg_paths[image_id])
        bg_image = Image.open(self.bg_paths[image_id//4000])    
        mask_image = Image.open(self.mask_paths[image_id])
        depth_image = Image.open(self.depth_paths[image_id])

        if self.transform:
            bg_image = self.bg_transform(bg_image)
            bg_fg_image = self.bg_fg_transform(bg_fg_image)

        if self.target_transform:
            mask_image = self.mask_transform(mask_image)
            depth_image = self.depth_transform(depth_image)

        return {"bg_image": bg_image, "bg_fg_image": bg_fg_image, "mask_image": mask_image, "depth_image": depth_image}