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

import utils
import models
import data_preparation
import visualization

from drawing_utils.model import Generator
from drawing_utils.dataset import UnpairedDepthDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(model, dataloader, hyperparams):
    pass

def training(model, dataloader_train, dataloader_test, hyperparams):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # training params
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=30)

    # dataset params
    parser.add_argument('--size', type=float, default=0.1)
    parser.add_argument('--img_format', type=str, default='jpg')
    parser.add_argument('--img_type', type=str, default='photos')

    # (loaded) model params
    parser.add_argument('--model_name', type=str, default='contour') # contour / anime / opensketch
    parser.add_argument('--model_folder', type=str, default='drawing_models') # models/ folder as start point

    # result params
    parser.add_argument('--inference_only', action='store_true', type=bool, default=False)
    parser.add_argument('--training_only', action='store_true', type=bool, default=False) # samples... for training evaluation are created anyway
    parser.add_argument('--inference_folder', type=Path)

    hyperparams = parser.parse_args()
    param_dict = vars(hyperparams)

    dataset_train, dataset_test = data_preparation.get_datasets(dataset='LineDrawingsV1', size=hyperparams.size, img_format=hyperparams.img_format, img_type=hyperparams.img_type)

    dataloader_train = DataLoader(dataset_train, batch_size=hyperparams.batch_size, shuffle=True, num_workers=min(4, os.cpu_count()))
    dataloader_test = DataLoader(dataset_test, batch_size=hyperparams.batch_size, shuffle=False, num_workers=min(4, os.cpu_count()))

    if not '.pth' in hyperparams.model_name: hyperparams.model_name += '.pth'
    model = utils.load_model(f"{hyperparams.model_folder}/{hyperparams.model_name}", 'LineDrawingsV1', options=hyperparams)
    model.to(device)

    if not hyperparams.inference_only: training(model, dataloader_train, dataloader_test, hyperparams)

    if not hyperparams.training_only: inference(model, dataloader_test, hyperparams)