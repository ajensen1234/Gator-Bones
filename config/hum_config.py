import torch
import torch.nn as nn
import albumentations as A
import numpy as np
import time
import os

class Configuration:
    def __init__(self):
        self.init = {
            'PROJECT_NAME': 'Humeral Segmentation',
            'MODEL_NAME': 'HUM_ALL_DATA_2023',
            'RUN_NAME': time.strftime('%Y-%m-%d-%H-%M-%S'),
            'WANDB_RUN_GROUP': 'Local',
            'FAST_DEV_RUN': False,  # Runs inputted batches (True->1) and disables logging and some callbacks
            'MAX_EPOCHS': 100,
            'MAX_STEPS': -1,    # -1 means it will do all steps and be limited by epochs
            'STRATEGY': 'ddp'    # This is the training strategy. Should be 'ddp' for multi-GPU (like HPG)
        }
        self.etl = {
            #'RAW_DATA_FILE': -1,    # -1 means it will create a full data csv from the image directory, using all images in the image directory
            'RAW_DATA_FILE': 'TSA_data.csv',
            'DATA_DIR': "data",
            'VAL_SIZE':  0.2,       # looks sus
            'TEST_SIZE': 0.01,      # I'm not sure these two mean what we think
            #'random_state': np.random.randint(1,50)
            # HHG2TG lol; deterministic to aid reproducibility
            'RANDOM_STATE': 42,

            'CUSTOM_TEST_SET': False,
            'TEST_SET_NAME': '/my/test/set.csv'
        }

        self.dataset = {
            'DATA_NAME': 'HUM_ALL_DATA_2023',
            'IMAGE_HEIGHT': 1024,
            'IMAGE_WIDTH': 1024,
            'MODEL_TYPE': 'hum',        # specifies that it's a femur model. how should we do this? not clear this is still best...
            'CLASS_LABELS': {0: 'bone', 1: 'background'},
            'IMG_CHANNELS': 1,      # Is this differnt from self.module['NUM_IMAGE_CHANNELS']
            'IMAGE_THRESHOLD': 0,
            'USE_ALBUMENTATIONS': True,
            'SUBSET_PIXELS': False
        }
        # segmentation_net_module needs to be below dataset because it uses dataset['IMG_CHANNELS']
        self.segmentation_net_module = {
                'NUM_KEY_POINTS': 1,
                'NUM_IMG_CHANNELS': self.dataset['IMG_CHANNELS']
        }

        self.swin_unetr_module = {
            'IMG_SIZE': (1024,1024),
            'IN_CHANNELS': 1,
            'OUT_CHANNELS': 1,
            'USE_CHECKPOINT': False,
            'SPATIAL_DIMS': 2
        }

        self.model = {
            'FEATURE_EXTRACTOR': 'swin_unetr', # See models/feature_extractors
            'HEAD': "swin_unetr", # See models/nets
            'LOSS': nn.BCEWithLogitsLoss(), # monai.losses.DiceLoss(sigmoid=True)
        }

        self.datamodule = {
            # *** CHANGE THE IMAGE DIRECTORY TO YOUR OWN ***
            #'IMAGE_DIRECTORY': '/media/sasank/LinuxStorage/Dropbox (UFL)/Canine Kinematics Data/TPLO_Ten_Dogs_grids',
            'IMAGE_DIRECTORY': '/blue/banks/ajensen123/JTML/JTML_ALL_DATA_ONE_FOLDER/',
            # *** CHANGE THE CHECKPOINT PATH TO YOUR OWN FOR TESTING ***
            #'CKPT_FILE': 'path/to/ckpt/file.ckpt',  # used when loading model from a checkpoint
            'CKPT_FILE': None,  # used when loading model from a checkpoint, such as in testing
            'BATCH_SIZE': 1,
            'SHUFFLE': True,        # Only for training, for test and val this is set in the datamodule script to False
            'NUM_WORKERS': 4,   # This number seems fine for local but on HPG, we have so many cores that a number like 4 seems better.
            'PIN_MEMORY': False,
            'SUBSET_PIXELS': False
        }


        # hyperparameters for training
        self.hparams = {
            'LOAD_FROM_CHECKPOINT': False,
            'learning_rate': 1e-3
        }

        self.transform = \
        A.Compose([
        A.RandomGamma(always_apply=False, p = 0.5,gamma_limit=(50,250)),
        A.ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(-0.02, 0.02), 
                            scale_limit=(-0.05, 0.05), rotate_limit=(-5, 5), 
                            interpolation=0, border_mode=0, value=(0, 0, 0)),
        A.Blur(always_apply=False, blur_limit=(2, 5), p=0.2),
        A.CoarseDropout(always_apply=False, p=0.25, min_holes=1, max_holes=10, 
                        min_height=25, max_height=25),
        A.MultiplicativeNoise(always_apply=False, p=0.25, multiplier=(0.9, 1.1), 
                              per_channel=True, elementwise=True)
        ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    p=0.85)
