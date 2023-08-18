# -*- coding: utf-8 -*-
"""
Dataet
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


join = os.path.join
import torch
from torch.utils.data import Dataset

import nibabel as nib
import random

# set seeds
torch.manual_seed(23)
torch.cuda.empty_cache()

MRI_SCAN = {
        'T1w'   : 't1',   
        'T1Gd'  : 't1ce', # or NON-ENHANCING tumor CORE - RED
        'T2'    : 't2',  # Green
        'FLAIR' : 'flair' # original 4 -> converted into 3 later, Yellow
    }


class BraTS2021Dataset(Dataset):
    def __init__(self, data_root, files, bbox_shift=20, img_size=1024, mri_scan="T2"):
        """
            Valid mri_scan strings:
                - T1w: T1-weighted pre-contrast
                - T1Gd: T1-weighted post-contrast
                - T2: T2-weighted
                - FLAIR: Fluid Attenuated Inversion Recovery
        """
        self.data_root = data_root
        
        self.img_path_files = files
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.img_path_files)}")
        
        self.img_size = img_size

    def __len__(self):
        return len(self.img_path_files)

    def __getitem__(self, index):
        file_name = self.img_path_files[index]
        scan_type = file_name.split('_')[-1].split('.')[0]
        scan = nib.load(file_name)
        raw_img = scan.get_fdata()
        raw_mask = nib.load(file_name.replace(scan_type, 'seg')).get_fdata() # TODO: Remove hardcoded z slice

        # Slice into mid-image size
        raw_mask = raw_mask[:,:,raw_mask.shape[2] // 2]
        raw_img = raw_img[:,:,[raw_img.shape[2] //2]].astype('float64')
        
        # resize image to (self.img_size, self.img_size) or 1024x1024 by default
        img = cv2.resize(raw_img, (self.img_size, self.img_size), 0, 0, interpolation=cv2.INTER_CUBIC)

        # resize mask
        mask = cv2.resize(raw_mask, (self.img_size, self.img_size), 0, 0, interpolation=cv2.INTER_NEAREST)

        # Clipped-intensity values to range between 95th and 99.5th
        img = np.clip(img, np.percentile(img,0.95), np.percentile(img,99.50))

        # Normalize values between [0,255]
        img *= (255.0/img.max())

        # Ensure no values go above desired range
        img = np.clip(img, 0, 255)

        """# TODO: Remove plotting code when finished debugging
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15, 10))
        ax1.imshow(img)
        ax2.imshow(mask)
        
        plt.savefig('test.png')
        plt.close()"""

        # Convert image and mask to 3 channels
        img3C = np.zeros((img.shape[0], img.shape[1], 3))#np.dstack([img, img, img])#
        img3C[:,:,0] = img # same value in each channel
        img3C[:,:,1] = img
        img3C[:,:,2] = img
        
        # convert the shape to (num_channels, H, W)
        img3C = np.transpose(img3C, (2, 0, 1))
        
        assert (
            np.max(img3C) <= 255.0 and np.min(img3C) >= 0.0
        ), "image should be normalized to [0, 255]"

        # TODO: Replace this with segmentation mask values
        label_ids = np.unique(mask)

        mask2D = np.uint8(
            mask == random.choice(list(label_ids))
        )  # only one label, (256, 256)

        # assert np.max(mask2D) == 1 and np.min(mask2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(mask2D > 0)
        
        if len(x_indices) == 0:
            return(
                torch.tensor(img3C).float(),
                torch.tensor(mask2D[None, :, :]).long(),
                torch.zeros(4).float(), # Size 4 for each corresponding coordinate
                scan_type
            )
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = mask2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img3C).float(),
            torch.tensor(mask2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            scan_type,
        )
