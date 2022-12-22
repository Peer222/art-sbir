from timeit import default_timer as timer
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

#import pix2pix_utils
import pix2pix_model
import data_preparation

from datetime import datetime

def train_pix2pix(epochs:int, dataloader_train, dataloader_test):

    start_time = timer()

    train_losses = []
    test_losses = []

    for i_epoch in tqdm(range(epochs), desc='Epoch'):
        
        train_loss = 0
        test_loss = 0



if __name__ == '__main__':

    BATCH_SIZE = 1 # placeholder
    LEARNING_RATE = 2e-4 # placeholder
    BETAS = (0.5, 0.999) # default for pix2pix (beta2 fixed)

    DATASET_SIZE = 0.1

    # from base_options
    param_dict = {
        'checkpoints_dir': './results',
        'name': 'test', # name of the experiment (can be freely choosed)

        'input_nc': 3, # rgb
        'output_nc': 1, # grayscale
        'ngf': 64,
        'ndf': 64,
        'n_layers_D': 3, # default (not used if netD == basic)
        'netD': 'basic', # default for pix2pix (patchGAN)
        'netG': 'unet_256', # default for pix2pix
        'norm': 'batch', # default for pix2pix -> 'instance' may be better (CycleGAN)
        'pool_size': 0,  # default for pix2pix (train)
        'gan_mode': 'vanilla',  # default for pix2pix (train)
        'isTrain': True,
        'init_type': 'normal', # default for pix2pix + cyclegan
        'init_gain': 0.02, # default

        'preprocess': 'resize_and_crop', # default
        'no_dropout': False, # placeholder

        'direction': 'AtoB',
        #'dataset_mode': 'aligned', # default for pix2pix -> own dataset

        'lambda_L1': 100.0, # weighting for L1_Loss <-> default for pix2pix ('train)
        'lr': LEARNING_RATE,
        'beta1': BETAS[0],
        'batch_size': BATCH_SIZE,
        'gpu_ids': [] # may be change pix2pix_model.py so it is not needed

    }
    options = argparse.Namespace(**param_dict)




    model = pix2pix_model.Pix2PixModel(options)

    train_dataset, test_dataset = data_preparation.get_datasets('SketchyPix2Pix', size=DATASET_SIZE)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=min(4, os.cpu_count()), shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=min(4, os.cpu_count()), shuffle=False)

    # optimizer defined in Pix2PixModel

