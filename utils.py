from pathlib import Path
from typing import Dict
from datetime import datetime
import json
import csv
from PIL import Image
import pandas as pd
from typing import List, Tuple, Dict

import torch
from torch import nn
import torchvision.transforms as transforms

from torchinfo import summary

import models
import pix2pix_model
from drawing_utils.model import DrawingGenerator
from pix2pix_model import Pix2PixModel
from artwork_gen_utils import net

def find_image_index(image_paths:List[Path], sketch_name:str) -> int:
    compare = lambda path: path.stem == sketch_name
    index, _ = next(((idx, path) for idx, path in enumerate(image_paths) if compare(path)), (-1,None))
    return index


# loss
# https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss

class CosineLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, sketch_tensor, image_tensor):
        # cosine similarity between -1 and 1 with 1 = equal and -1 opposite -> similar if value is small (*-1) and has be positive (+1)
        return (self.cosine_similarity(sketch_tensor, image_tensor) * -1) + 1

cosine_distance = CosineLoss()#nn.CosineSimilarity(dim=1)

euclidean_distance = nn.PairwiseDistance(p=2, keepdim=False)
"""
# default distance function for triplet margin loss (eventually the dimension has to be adapted)
def euclidean_distance(t1:torch.Tensor, t2:torch.Tensor) -> float:
    print(torch.sum( torch.pow(t2 - t1, 2), dim=1))
    return torch.sqrt(torch.sum( torch.pow(t2 - t1, 2), dim=2))
"""
class TripletMarginLoss_with_classification(nn.Module):
    def __init__(self, margin, classification_weight=0.5, distance_f=euclidean_distance):
        super().__init__()
        self.classification_weight = classification_weight
        self.classification_weight2 = 0
        self.margin = margin

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(margin=self.margin, distance_function=distance_f)
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, s_logits, p_logits, n_logits, cs_logits, cp_logits, labels):
        return self.triplet_loss(s_logits, p_logits, n_logits) + self.classification_weight * (self.classification_loss(cs_logits, labels) + self.classification_loss(cp_logits, labels))

class TripletMarginLoss_with_classification2(nn.Module):
    def __init__(self, margin, classification_weight=0.25, classification_weight2=0.5, distance_f=euclidean_distance):  # styles, genres (same genre for pos and neg img)
        super().__init__()
        self.classification_weight = classification_weight
        self.classification_weight2 = classification_weight2
        self.margin = margin

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(margin=self.margin, distance_function=distance_f)
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, s_logits, p_logits, n_logits, cs_logits, cp_logits, cs_logits2, cp_logits2, labels, labels2):
        classification_loss = self.classification_loss(cs_logits, labels) + self.classification_loss(cp_logits, labels)
        classification_loss2 = self.classification_loss(cs_logits2, labels2) + self.classification_loss(cp_logits2, labels2)
        return self.triplet_loss(s_logits, p_logits, n_logits) + self.classification_weight * classification_loss + self.classification_weight2 * classification_loss2

MARGIN = 0.2 # Sketching without Worrying

#triplet_euclidean_loss = nn.TripletMarginWithDistanceLoss(margin=MARGIN, distance_function=euclidean_distance)
"""
triplet_euclidean_loss = nn.TripletMarginLoss(margin=MARGIN)

triplet_euclidean_loss_with_classification = TripletMarginLoss_with_classification(margin=MARGIN)
triplet_euclidean_loss_with_classification2 = TripletMarginLoss_with_classification2(margin=MARGIN, classification_weight=0, classification_weight2=0)

triplet_cosine_loss = nn.TripletMarginWithDistanceLoss(margin=MARGIN, distance_function=cosine_distance)
triplet_cosine_loss_with_classification = TripletMarginLoss_with_classification(margin=MARGIN, distance_f=cosine_distance)
triplet_cosine_loss_with_classification2 = TripletMarginLoss_with_classification2(margin=MARGIN, distance_f=cosine_distance)
"""


def process_losses(loss_tracker:Dict, loss:Dict, size:int, method:str, lambda_:float=100):
    for key in loss_tracker.keys():
        # pix2pix combined losses
        #if key == 'Discriminator': loss[key] = (loss['D_real'] + loss['D_fake']) / 2
        #if key == 'Generator': loss[key] = loss['G_GAN'] + loss['G_L1'] * lambda_

        if method == 'add':
            loss_tracker[key] += (loss[key] / size)
        elif method == 'append':
            loss_tracker[key].append(loss[key] / size)
    return loss_tracker

# 4 dimensional input expected [batch, channels, H, W]
def convert_pix2pix_to_255(visuals:Dict) -> Dict:
    to_rgb = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) )
    for key in visuals.keys():
        if type(visuals[key]) == str: continue
        visuals[key] = ((visuals[key] + 1) / 2 * 255).type(torch.uint8)
        if visuals[key].shape[-3] == 1: visuals[key] = to_rgb(visuals[key])
    return visuals


# semi supervised
def get_sketch_gen_transform(type:str='train'):
    transform_list = []
    if type == 'train':
        transform_list.extend([transforms.Resize(256)])
    elif type == 'test':
        transform_list.extend([transforms.Resize(256)])
    # transform_list.extend(
    #     [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transforms.Compose(transform_list)


# model saver and loader

# loads resnet50m state dicts or arbitrary models
def load_model(name:str, dataset:str=None, model_type:str=None, max_seq_len=0, options=None) -> nn.Module:
    path = Path("models/") / name
    if path.is_dir() and model_type == 'Pix2Pix':
        loaded = [torch.load(path / 'latest_net_G.pth'), torch.load(path / 'latest_net_D.pth')]
    elif path.is_dir() and model_type == 'AdaIN':
        loaded = [torch.load(path / 'vgg_normalised.pth'), torch.load(path / 'decoder.pth')]
    else:
        loaded = torch.load(path, map_location=torch.device('cpu'))
    model = None

    datasetsV1 = [ 'SketchyV1', 'SketchyDatasetV1', 'Sketchy', 'KaggleV1', 'KaggleDatasetV1', 'Kaggle', 'AugmentedKaggleV1', 'AugmentedKaggleDatasetV1',  'MixedDatasetV1',  'MixedDatasetV2', 'MixedDatasetV3', 'MixedDatasetV4'] # MixedDatasetV2 because only negative image selection is used (no labels)

    if isinstance(loaded, dict) or isinstance(loaded, list):
        print("Dictionary used to load model")
        if model_type == 'Pix2Pix':
            print('Pix2Pix model loaded')
            model = Pix2PixModel(options)
            if isinstance(model.netG, torch.nn.DataParallel):
                model.netG = model.netG.module
            model.netG.load_state_dict(loaded[0])
            #model.netD.load_state_dict(loaded[1]) # fails -> model from PhotoSketch has to be used
        elif model_type == 'AdaIN':
            decoder = net.decoder
            vgg = net.vgg
            decoder.load_state_dict(loaded[1])
            vgg.load_state_dict(loaded[0])
            vgg = nn.Sequential(*list(vgg.children())[:31])
            print(f"Model {name} loaded", flush=True)
            return {'encoder': vgg, 'decoder': decoder}

        elif model_type == 'DrawingGenerator' or dataset == 'LineDrawingsV1' or 'drawing' in name:
            print('Drawing model loaded')
            model = DrawingGenerator(input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True)
            model.load_state_dict(loaded)
        elif model_type == 'ModifiedResNet' or dataset in datasetsV1:
            model = models.ModifiedResNet(layers=(3, 4, 6, 3), output_dim=1024)
            model.load_state_dict(loaded, strict=False)
        elif model_type == 'ModifiedResNet_with_classification' and dataset in ['SketchyV2', 'SketchyDatasetV2']:
            print("Model with classification layer loaded")
            model = models.ModifiedResNet_with_classification(layers=(3, 4, 6, 3), output_dim=1024)
            model.load_state_dict(loaded, strict=False)
        elif model_type == 'Photo2Sketch' or dataset == 'VectorizedSketchyV1' or dataset == 'QuickdrawV1':
            print('Photo2Sketch model loaded')
            model = models.Photo2Sketch(options.z_size, options.dec_rnn_size, options.num_mixture, max_seq_len)
            model.load_state_dict(loaded)
        elif model_type == 'ModifiedResNet_with_classification' and (dataset in ['KaggleV2', 'KaggleDatasetV2', 'AugmentedKaggleV2', 'AugmentedKaggleDatasetV2']):
            # fails if sketchy pretrained model with classifier-125 is loaded
            try:
                model = models.ModifiedResNet_with_classification(layers=(3, 4, 6, 3), output_dim=1024, num_classes=70, num_classes2=32) # styles, genres
                model.load_state_dict(loaded, strict=False)
                print("Normally loaded from state dict")
            except:
                model = models.ModifiedResNet_with_classification(layers=(3, 4, 6, 3), output_dim=1024, num_classes=125, num_classes2=32) # styles, genres
                model.load_state_dict(loaded, strict=False)
                model.classifier = nn.Linear(1024, 70)
                print("Normal load failed - Firstly loaded classifier-125 changed to classifier-70")
        elif model_type == 'ModifiedResNet_with_classification' and dataset == 'CategorizedMixedDatasetV2':
            try:
                model = models.ModifiedResNet_with_classification(layers=(3, 4, 6, 3), output_dim=1024, num_classes=33) # genres
                model.load_state_dict(loaded, strict=False)
                print("Normally loaded from state dict")
            except:
                model = models.ModifiedResNet_with_classification(layers=(3, 4, 6, 3), output_dim=1024, num_classes=125) # genres
                model.load_state_dict(loaded, strict=False)
                model.classifier = nn.Linear(1024, 33)
                print("Normal load failed - Firstly loaded classifier-125 changed to classifier-33")
        else:
            raise Exception(f"No model found with {model_type} and {dataset}")

    else:
        print("Model completely loaded from file")
        model = loaded

    print(f"Model {name} loaded", flush=True)
    return model

# saves model and related parameters and results -> returns result folder path
# after calling save_model model is on cpu
def save_model(model:nn.Module, data_dict:Dict, training_dict:Dict={}, param_dict:Dict={}, inference_dict:Dict={}) -> Path:
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    model_name = f"{model.__class__.__name__}_{data_dict['dataset']}_{date_time}"
    # just saves model if it was trained before
    if training_dict:
        suffix = "pth"

        if isinstance(model, pix2pix_model.Pix2PixModel):
            model_path = Path("models") / model_name
            model_path.mkdir(parents=True, exist_ok=True)
            for name in model.model_names:
                if isinstance(name, str):
                    net = getattr(model, 'net' + name)
                    torch.save(net.cpu().state_dict(), model_path / f'net_{name}.{suffix}')
        else:
            model_path = Path("models") / f"{model_name}.{suffix}"
            torch.save(model.cpu().state_dict(), model_path)

        print(f"Model saved as {model_name}.{suffix}")
    else:
        print("No model saved")

    result_path = Path("results")
    if not result_path.is_dir(): result_path.mkdir(parents=True, exist_ok=True)

    result_path = result_path / model_name

    result_path.mkdir(parents=True, exist_ok=True)

    with open(result_path / "data_params.json", "w") as f:
        json.dump(data_dict, f, indent=4)

    with open(result_path / "training.json", "w") as f:
        json.dump(training_dict, f, indent=4)

    with open(result_path / "training_params.json", "w") as f:
        json.dump(param_dict, f, indent=4)

    with open(result_path / "inference.json", "w") as f:
        json.dump(inference_dict, f, indent=4)

    print(f"Data saved in {str(result_path)}", flush=True)

    return result_path



def load_image_features(folder_name:str) -> Tuple[any, torch.Tensor]:
    path = Path("data/image_features") / folder_name
    image_paths = list(pd.read_csv(path / "image_paths.csv", header=None).values)
    image_paths = [Path(img_path[0]) for img_path in image_paths]
    image_features = pd.read_csv(path / "image_features.csv", header=None).values
    return image_paths, torch.from_numpy(image_features)

def save_image_features(model_name:str, dataset_name:str, inference_dataset, image_features) -> str:
    feature_path = Path("data/image_features")
    if not feature_path.is_dir(): feature_path.mkdir(parents=True, exist_ok=True)

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    feature_path = feature_path / f"{model_name}_{dataset_name}_{date_time}"
    feature_path.mkdir(parents=True, exist_ok=True)

    str_paths = [ [str(path)] for path in inference_dataset.image_paths]
    with open(feature_path / "image_paths.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(str_paths)
        print(f"Image paths saved in {feature_path / 'image_paths.csv'}")

    with open(feature_path / "image_features.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(image_features.numpy())

    print(f"Image features saved in {feature_path / 'image_features.csv'}")
    return feature_path.name