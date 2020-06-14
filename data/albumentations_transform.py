import torch
import torchvision

from PIL import Image
import cv2
import numpy as np

import albumentations
print('Albumentations version:',albumentations.__version__)
from albumentations import Compose, Normalize, HorizontalFlip, Resize, RandomBrightnessContrast, Cutout, CoarseDropout, GaussNoise, PadIfNeeded, RandomCrop
from albumentations.pytorch import ToTensor

class albumentation_compose:
    def __init__(self, settype, means=None, stddev=None, resize=(192, 192)):
        self.settype = settype
        self.means = means
        self.stddev = stddev
        self.resize = resize
        if self.settype == 'input':
          print("input transforms applied")
          self.albumentation_transform = Compose([
                Resize(height=self.resize[0], width=self.resize[1], interpolation=1, always_apply=True, p=1),
                # PadIfNeeded(min_height=128, min_width=128, border_mode=4, p=1.0),
                #   RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.40, 0.82), contrast_limit=(-0.40, 0.82), brightness_by_max=True),
                # RandomCrop(height=64, width=64, always_apply=True, p=1.0),
                # HorizontalFlip(always_apply=True, p=1.0),
                # Cutout(always_apply=True, p=1.0, num_holes=1, max_h_size=8, max_w_size=8, fill_value=list(255 * self.means)),
                #   GaussNoise(always_apply=False, p=1.0, var_limit=(60, 100)),
                #   CoarseDropout(max_holes=2, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, fill_value=list(255 * self.means), always_apply=False, p=1.0),
                  Normalize(
                      mean = list(self.means),
                      std = list(self.stddev),
                      ),
                ToTensor()
          ])
        elif self.settype == 'target':
          print("target transforms applied")
          self.albumentation_transform = Compose([
                Resize(height=self.resize[0], width=self.resize[1], interpolation=1, always_apply=True, p=1),
                ToTensor()
          ])


    def __call__(self, img):
        img = np.array(img)
        if img.ndim == 2:
            img = img[:,:, np.newaxis]
        img = self.albumentation_transform(image=img)['image']
        return img
