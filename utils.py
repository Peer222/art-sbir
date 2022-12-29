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

def find_image_index(image_paths:List[Path], sketch_name:str) -> int:
    compare = lambda path: path.stem == sketch_name
    index, _ = next(((idx, path) for idx, path in enumerate(image_paths) if compare(path)), (-1,None))
    return index


# loss
# https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss

cosine_distance = nn.CosineSimilarity(dim=1) #not tested

euclidean_distance = nn.PairwiseDistance(p=2, keepdim=False)
"""
# default distance function for triplet margin loss (eventually the dimension has to be adapted)
def euclidean_distance(t1:torch.Tensor, t2:torch.Tensor) -> float:
    print(torch.sum( torch.pow(t2 - t1, 2), dim=1))
    return torch.sqrt(torch.sum( torch.pow(t2 - t1, 2), dim=2))
"""
class TripletMarginLoss_with_classification(nn.Module):
    def __init__(self, margin, classification_weight=0.5):
        super().__init__()
        self.classification_weight = classification_weight

        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, s_logits, p_logits, n_logits, cs_logits, cp_logits, labels):
        return self.triplet_loss(s_logits, p_logits, n_logits) + self.classification_weight * (self.classification_loss(cs_logits, labels) + self.classification_loss(cp_logits, labels))


MARGIN = 0.2 # Sketching without Worrying

#triplet_euclidean_loss = nn.TripletMarginWithDistanceLoss(margin=MARGIN, distance_function=euclidean_distance)
triplet_euclidean_loss = nn.TripletMarginLoss(margin=MARGIN)

triplet_euclidean_loss_with_classification = TripletMarginLoss_with_classification(margin=MARGIN)


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
def load_model(name:str, dataset:str='Sketchy', max_seq_len=0, options=None) -> nn.Module:
    path = Path("models/") / name
    loaded = torch.load(path, map_location=torch.device('cpu'))
    model = None

    if isinstance(loaded, dict):
        print("Dictionary used to load model")
        if dataset == 'LineDrawingsV1' or 'drawing' in name:
            print('Drawing model loaded')
            model = None
        elif dataset == 'Sketchy':
            model = models.ModifiedResNet(layers=(3, 4, 6, 3), output_dim=1024)
            model.load_state_dict(loaded)
        elif dataset == 'SketchyV2':
            print("Model with classification layer loaded")
            model = models.ModifiedResNet_with_classification(layers=(3, 4, 6, 3), output_dim=1024)
            model.load_state_dict(loaded, strict=False)
        elif dataset == 'VectorizedSketchyV1' or dataset == 'QuickdrawV1':
            print('Photo2Sketch model loaded')
            model = models.Photo2Sketch(options.z_size, options.dec_rnn_size, options.num_mixture, max_seq_len)
            model.load_state_dict(loaded)
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

def save_image_features(model_name:str, dataset_name:str, inference_dataset, image_features) -> None:
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