"""
Sasank Desaraju
9/14/22
"""

"""
*** This actually does NOT have any Lightning in it. (-_-) ***
It is just a standard Pytorch dataset. I still created this newly to
remake it intentionally."""

import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import cv2
import os
import pandas as pd


class LitJTMLDataset(Dataset):
    
    def __init__(self, config, evaluation_type, transform=None):
        """
        Args:
            config (config): Dictionary of vital constants about data.
            store_data_ram (boolean): Taken from config.
            evaluation_type (string): Dataset evaluation type (must be 'training', 'validation', or 'test')
            num_points (int): Taken from config.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Create local copies of the arguments
        self.config = config
        self.num_points = self.config.segmentation_net_module['NUM_KEY_POINTS']
        self.transform = self.config.transform
        
        # Check that evaluation_type is valid and then store
        if evaluation_type in ['train', 'val', 'test', 'naive']:
            self.evaluation_type = evaluation_type
        else:
            raise Exception('Incorrect evaluation type! Must be either \'train\', \'val\', \'test\', or \'naive\'.')

        # Load the data from the big_data CSV file into a pandas dataframe
        self.data = pd.read_csv(os.path.join(self.config.etl['DATA_DIR'], self.config.dataset['DATA_NAME'], self.evaluation_type + '_' + self.config.dataset['DATA_NAME'] + '.csv'))
        




    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        

        # Get the row of the dataframe
        row = self.data.iloc[idx]

        # Get the image name
        image_name = row['Image address']

        # Get the image
        image = io.imread(os.path.join(self.config.datamodule['IMAGE_DIRECTORY'], image_name))

        # Get the keypoint labels and segmentation labels
        if self.config.dataset['MODEL_TYPE'] == 'fem':
            kp_label = row['Femur 2D KP points']
            seg_label = io.imread(os.path.join(self.config.datamodule['IMAGE_DIRECTORY'], row['Fem label address']))
        elif self.config.dataset['MODEL_TYPE'] == 'tib':
            kp_label = row['Tibia 2D KP points']
            seg_label = io.imread(os.path.join(self.config.datamodule['IMAGE_DIRECTORY'], row['Tib label address']))
        else:
            raise Exception('Incorrect model type! Must be either \'fem\' or \'tib\'.')

        kp_label = kp_label[2:-2]
        kp_label = kp_label.split(']\n [')
        kp_label = [np.array([float(x) for x in list(filter(None, kp.split(' ')))]) for kp in kp_label]
        kp_label = np.array(kp_label)
        kp_label[:, 1] = 1 - kp_label[:, 1]         # ! New kp_label preprocessing
        kp_label = kp_label * 1024
        

        # * Transformations
        # Albumenations
        image_no_transform = image
        if self.transform and self.config.dataset['USE_ALBUMENTATIONS'] == True:
            transformed = self.transform(image=image, mask=seg_label, keypoints=kp_label)
            image, seg_label, kp_label = transformed['image'], transformed['mask'], transformed['keypoints']

        """
        Name: Subset Pixels
        Jiayu Huang

        Inputs:

        image: input image
        seg_label: segmentation label
        Outputs:

        image: subsetted input image
        Rationale:
        This code subsets pixels based on segmentation labels to focus on areas of interest and reduce noise. 

        Future:
        Consider making kernel size and dilation iterations range configurable or using adaptive methods based on image content.
        """
        # * Subset Pixels
        full_image = image             # Save full image (no subset_pixels) for visualization
        if self.config.dataset['SUBSET_PIXELS'] == True:
            label_dst = np.zeros_like(seg_label)
            label_normed = cv2.normalize(seg_label, label_dst, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)
            seg_label = label_normed

            random_kernel_size = (np.random.randint(10, 50), np.random.randint(10, 50))  # Generate random kernel size
            kernel = np.ones(random_kernel_size, np.uint8)

            # Generate a random number of iterations within a specific range
            random_iterations = np.random.randint(1, 11)

            label_dilated = cv2.dilate(seg_label, kernel, iterations = random_iterations)
            image_subsetted = cv2.multiply(label_dilated, image)
            image = image_subsetted

        # * Convert to tensors
        image = torch.FloatTensor(image[None, :, :]) # Store as byte (to save space) then convert when called in __getitem__. - What. What does this mean?
        full_image = torch.FloatTensor(full_image[None, :, :]) # Store as byte (to save space) then convert when called in __getitem__
        seg_label = torch.FloatTensor(seg_label[None, :, :])
        #kp_label = torch.FloatTensor(kp_label.reshape(-1))      # Reshape to 1D array so that it's 2*num_keypoints long
        kp_label = torch.FloatTensor(kp_label)          # kp_label is of shape (num_keypoints, 2)
        
        #print("kp_label.shape:")
        #print(kp_label.shape)

        # * Create a dictionary of the sample
        # sample = {'image': image,
        #             'img_name': image_name,
        #             'kp_label': kp_label,
        #             'seg_label': seg_label,
        #             'full_image': full_image,
        #             'image_no_transform': image_no_transform}
        
        sample = {'image': image, 'label': seg_label, 'img_name': image_name}

        # * Return the sample
        return sample

