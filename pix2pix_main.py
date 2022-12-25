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
import utils
import visualization


device = [0] if torch.cuda.is_available() else [] # gpu_id

def train_pix2pix(model, dataloader_train, dataloader_test, opt, data_dict):

    start_time = timer()

    # ['G_GAN', 'G_L1', 'D_real', 'D_fake']
    train_losses = {'G_GAN': [], 'G_L1': [], 'D_real': [], 'D_fake': [], 'D_total': [], 'G_total' : []}
    test_losses = {'G_GAN': [], 'G_L1': [], 'D_real': [], 'D_fake': [], 'D_total': [], 'G_total' : []}

    result_path = None
    param_dict = vars(opt)

    for epoch in tqdm(range(1, opt.n_epochs + 1)):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = timer()

        train_loss = {'G_GAN': 0, 'G_L1': 0, 'D_real': 0, 'D_fake': 0, 'D_total': 0, 'G_total' : 0}
        test_loss = {'G_GAN': 0, 'G_L1': 0, 'D_real': 0, 'D_fake': 0, 'D_total': 0, 'G_total' : 0}
        samples = []

        model.train()
        for i, data in enumerate(dataloader_train):  # inner loop within one epoch

            model.set_input(data)         # unpack data from dataset and apply preprocessing

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            train_loss = utils.process_losses(train_loss, losses, opt.batch_size, 'add', opt.lambda_L1)

        
        model.eval() # can be turned of for experiments (dropout)
        with torch.inference_mode():
            for i, data in enumerate(dataloader_test):

                model.set_input(data)
                model.calculate_loss()

                losses = model.get_current_losses()
                test_loss = utils.process_losses(test_loss, losses, opt.batch_size, 'add', opt.lambda_L1)

                if i < 15 and (epoch % opt.save_epoch_freq == 0 or epoch == 1):
                    visuals = model.get_current_visuals() # ordered dict with ['real_A', 'fake_B', 'real_B']

                    visuals = utils.convert_pix2pix_to_255(visuals)
                    samples.append([visuals['real_A'].cpu(), visuals['fake_B'].cpu(), visuals['real_B'].cpu()])


        train_losses = utils.process_losses(train_losses, train_loss, len(dataloader_test), 'append', opt.lambda_L1)
        test_losses = utils.process_losses(test_losses, test_loss, len(dataloader_test), 'append', opt.lambda_L1)

        print(f'End of epoch {epoch} / {opt.n_epochs} \t Time Taken: {timer() - epoch_start_time} sec', flush=True)
        print(f'Train losses -> G_GAN: {train_loss["G_GAN"]}, G_L1: {train_loss["G_L1"]}, D_real: {train_loss["D_real"]}, D_fake: {train_loss["D_fake"]} ', flush=True)
        print(f'Test losses -> G_GAN: {test_loss["G_GAN"]}, G_L1: {test_loss["G_L1"]}, D_real: {test_loss["D_real"]}, D_fake: {test_loss["D_fake"]} ', flush=True)

        if epoch % opt.save_epoch_freq == 0 or epoch == 1:
            print(f'saving the model at the end of epoch {epoch}')
            param_dict['epoch'] = epoch
            training_dict = {"train_losses": train_losses, "test_losses": test_losses, "training_time": timer() - start_time}
            result_path = utils.save_model(model, data_dict, training_dict, param_dict, {})
            model.netG.to(model.device), model.netD.to(model.device)

            visualization.build_all_loss_curves(train_losses, test_losses, result_path, epoch)
            visualization.show_triplets(samples, result_path / f'samples_{epoch}.png', mode='image')


    return {"train_losses": train_losses, "test_losses": test_losses, "training_time": timer() - start_time}


if __name__ == '__main__':

    EPOCHS = 30

    BATCH_SIZE = 6 # 1 - 10 used depending on experiment
    BATCH_SIZE_TEST = 1
    LEARNING_RATE = 2e-4 # default for pix2pix
    BETAS = (0.5, 0.999) # default for pix2pix (beta2 fixed)

    DATASET_SIZE = 0.01#1 #0.005

    # from base_options
    param_dict = {
        'checkpoints_dir': './results',
        'name': 'placeholder', # name of the experiment (can be freely choosed)
        'save_epoch_freq': 30,
        'n_epochs': EPOCHS,

        'input_nc': 3, # rgb
        'output_nc': 1, # grayscale = 1 but not compatible with other functions # change inside dataset too
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

        #'preprocess': 'resize_and_crop', # default
        'no_dropout': False, # default for pix2pix (if instanceNorm True (cycle gan))

        'direction': 'AtoB',
        #'dataset_mode': 'aligned', # default for pix2pix -> own dataset

        'lambda_L1': 100.0, # weighting for L1_Loss <-> default for pix2pix ('train)
        'lr': LEARNING_RATE,
        'beta1': BETAS[0],
        'batch_size': BATCH_SIZE,
        'gpu_ids': device # maybe change pix2pix_model.py so it is not needed or more convenient

    }
    options = argparse.Namespace(**param_dict)




    model = pix2pix_model.Pix2PixModel(options)

    train_dataset, test_dataset = data_preparation.get_datasets('SketchyPix2Pix', size=DATASET_SIZE)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=min(4, os.cpu_count()), shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, num_workers=min(4, os.cpu_count()), shuffle=True)

    # optimizer defined in Pix2PixModel

    training_dict = train_pix2pix(model, train_dataloader, test_dataloader, options, train_dataset.state_dict)

