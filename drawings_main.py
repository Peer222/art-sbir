import argparse
from tqdm.auto import tqdm
from timeit import default_timer as timer
import os
from pathlib import Path
from PIL import Image
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image


import utils
import models
import data_preparation
import visualization

from drawing_utils.dataset import UnpairedDepthDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# works only with batch size 1
def inference(model, dataloader, hyperparams):
    print('Inference started', flush=True)

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            input_image = batch['image']
            name = batch['name'][0]
            image = model(input_image)
            save_image(image.data, f"{hyperparams.inference_folder}/{name}.png")
    return {}

def training(model, dataloader_train, dataloader_test, hyperparams):
    print('Training not implemented', flush=True)

    model.train()
    return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # training params
    parser.add_argument('--batch_size', type=int, default=1)# default paper: 6 # only 1 causes no shape mismatch error
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=30)

    # dataset params
    parser.add_argument('--dataset_name', type=str, default='KaggleDatasetImgOnlyV1')
    parser.add_argument('--size', type=float, default=1.0)
    parser.add_argument('--img_format', type=str, default='jpg')
    parser.add_argument('--img_type', type=str, default='photos')

    # (loaded) model params
    parser.add_argument('--model_name', type=str, default='contour') # contour / anime / opensketch
    parser.add_argument('--model_folder', type=str, default='drawing_models') # models/ folder as start point

    # result params
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--training_only', action='store_true') # samples... for training evaluation are created anyway
    parser.add_argument('--inference_folder', type=str, help="Set automatically")
    parser.add_argument('--no_final_saving', action='store_true')

    hyperparams = parser.parse_args()
    param_dict = vars(hyperparams)

    if 'Kaggle' in hyperparams.dataset_name: hyperparams.inference_folder = f"data/kaggle/{hyperparams.model_name}_drawings"
    else: NotImplemented
    folder = Path(hyperparams.inference_folder)
    if not folder.is_dir():
        folder.mkdir(parents=True, exist_ok=True)

    # transformation from papercode
    transforms_img = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()]) # resizes smallest edge + keeps ratio 

    dataset_train, dataset_test = data_preparation.get_datasets(dataset=hyperparams.dataset_name, size=hyperparams.size, img_format=hyperparams.img_format, img_type=hyperparams.img_type, 
                                transform=transforms_img)

    dataloader_train = DataLoader(dataset_train, batch_size=hyperparams.batch_size, shuffle=False, num_workers=0)#min(4, os.cpu_count()))
    dataloader_test = DataLoader(dataset_test, batch_size=hyperparams.batch_size, shuffle=False, num_workers=0)#min(4, os.cpu_count()))

    if not '.pth' in hyperparams.model_name: hyperparams.model_name += '.pth'
    model = utils.load_model(f"{hyperparams.model_folder}/{hyperparams.model_name}", hyperparams.dataset_name, options=hyperparams)
    model.to(device)

    training_dict = {}
    if not hyperparams.inference_only: training_dict = training(model, dataloader_train, dataloader_test, hyperparams)

    inference_dict = {}
    if not hyperparams.training_only: 
        inference_dict = inference(model, dataloader_test, hyperparams)
        inference_dict = inference(model, dataloader_train, hyperparams)

    if not hyperparams.no_final_saving:
        param_dict['final'] = True
        utils.save_model(model, data_dict=dataset_train.state_dict, training_dict=training_dict, param_dict=param_dict, inference_dict=inference_dict)