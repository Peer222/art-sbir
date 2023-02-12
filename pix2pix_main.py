from timeit import default_timer as timer
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import torchvision.transforms.functional as TF

#import pix2pix_utils
import pix2pix_model
import data_preparation
import utils
import visualization


device = [0] if torch.cuda.is_available() else [] # gpu_id

def train_pix2pix(model, dataloader_train, dataloader_test, opt, data_dict):

    start_time = timer()

    # Generator already pre trained so firstly train only decoder for 1 epoch
    model.train()
    for i, data in enumerate(dataloader_train):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters(decoder_only=True)   # calculate loss functions, get gradients, update network weights

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

def inference(model, dataloader, opt, result_path:Path):
    if not result_path.is_dir():
        result_path.mkdir(parents=True, exist_ok=True)
    #samples = []
    model.eval() # can be turned of for experiments (dropout)
    with torch.inference_mode():
        for i, data in tqdm(enumerate(dataloader)):
            # convert kaggle img only format to pix2pix input
            #print(i, data['path'],f"mean: {torch.mean(data['image'])}", f"std: {torch.std(data['image'])}", f"min: {torch.min(data['image'])}", f"max: {torch.max(data['image'])}")
            #converted_data = {'A':transform_pix2pix((1 - torch.std(data['image'])) * 10)(data['image']), 'B': transform_pix2pix((1 - torch.std(data['image'])) * 10)(data['image']), 'img_paths':data['path']}
            converted_data = {'A':data['image'], 'B': data['image'], 'img_paths':data['path']}

            model.set_input(converted_data)
            model.forward()

            visuals = model.get_current_visuals() # ordered dict with ['real_A', 'fake_B', 'real_B']

            #visuals = utils.convert_pix2pix_to_255(visuals)

            save_image(visuals['fake_B'].cpu(), result_path / f"{data['name'][0]}.png")
            
            #samples.append([visuals['real_A'].cpu(), visuals['fake_B'].cpu(), visuals['real_B'].cpu()])
            #if i > 15: break

    #visualization.show_triplets(samples, result_path / f'samples_inference.png', mode='image')


class ContrastTransform:
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor
    def __call__(self, x):
        return TF.adjust_contrast(x, self.contrast_factor)

def transform_pix2pix(contrast:float=1, to_grayscale:bool=False):
        # from pix2pix
        #transformations = [transforms.ToTensor(), ContrastTransform(6), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transformations = [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), ContrastTransform(contrast), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transformations += [transforms.Grayscale(1)] if to_grayscale else []
        return transforms.Compose(transformations)

if __name__ == '__main__':

    EPOCHS = 1

    BATCH_SIZE = 6 # 1 - 10 used depending on experiment
    BATCH_SIZE_TEST = 1
    LEARNING_RATE = 1e-5#2e-4 # default for pix2pix
    BETAS = (0.5, 0.999) # default for pix2pix (beta2 fixed)

    DATASET_SIZE = 1.0 #0.01#1 #0.005

    # from base_options
    param_dict = {
        'checkpoints_dir': './results',
        'name': 'placeholder', # name of the experiment (can be freely choosed)
        'save_epoch_freq': 1,
        'n_epochs': EPOCHS,

        'input_nc': 3, # rgb
        'output_nc': 1, # grayscale = 1 but not compatible with other functions # change inside dataset too
        'ngf': 64,
        'ndf': 64,
        'n_layers_D': 3, # default (not used if netD == basic)
        'netD': 'basic', # default for pix2pix (patchGAN)
        'netG': 'resnet_9blocks', #'DrawingGenerator', #'unet_256', # default for pix2pix
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

        'lambda_L1': 10, # 100.0, # weighting for L1_Loss <-> default for pix2pix ('train)
        'lr': LEARNING_RATE,
        'beta1': BETAS[0],
        'batch_size': BATCH_SIZE,
        'gpu_ids': device # maybe change pix2pix_model.py so it is not needed or more convenient

    }
    options = argparse.Namespace(**param_dict)



    # only loads generator from state dict
    model = utils.load_model('pix2pix_models', model_type='Pix2Pix', options=options) #pix2pix_model.Pix2PixModel(options)

    #train_dataset, test_dataset = data_preparation.get_datasets('SketchyPix2Pix', size=DATASET_SIZE)
    train_dataset, test_dataset = data_preparation.get_datasets('KaggleDatasetImgOnlyV1', img_type='images', size=DATASET_SIZE, transform=transform_pix2pix(6))

    #train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=min(4, os.cpu_count()), shuffle=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TEST, num_workers=min(4, os.cpu_count()), shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, num_workers=min(4, os.cpu_count()), shuffle=False)

    # optimizer defined in Pix2PixModel

    #training_dict = train_pix2pix(model, train_dataloader, test_dataloader, options, train_dataset.state_dict)

    inference(model, test_dataloader, options, Path("./data/kaggle/photo_sketch"))
    inference(model, train_dataloader, options, Path("./data/kaggle/photo_sketch"))

