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

from torchinfo import summary

import models

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

MARGIN = 0.2 # Sketching without Worrying

#triplet_euclidean_loss = nn.TripletMarginWithDistanceLoss(margin=MARGIN, distance_function=euclidean_distance)
triplet_euclidean_loss = nn.TripletMarginLoss(margin=MARGIN)


# model saver and loader

# loads resnet50m state dicts or arbitrary models
def load_model(name:str) -> nn.Module:
    path = Path("models/") / name
    loaded = torch.load(path)
    model = None

    if isinstance(loaded, dict):
        print("Dictionary used to load model")
        model = models.ModifiedResNet(layers=(3, 4, 6, 3), output_dim=1024) # 2048 has to be divisible by heads - text encoder used 8
        #model = torch.load(Path("models/CLIP_ResNet-50.pt"))
        model.load_state_dict(loaded)
    else:
        print("Model completely loaded from file")
        model = loaded

    print(f"Model {name} loaded")
    return model

# saves model and related parameters and results
def save_model(model:nn.Module, data_dict:Dict, training_dict:Dict={}, param_dict:Dict={}, inference_dict:Dict={}) -> None:
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    model_name = f"{model.__class__.__name__}_{data_dict['dataset']}_{date_time}"
    # just saves model if it was trained before
    if training_dict:
        suffix = "pth"
        model_path = Path("models") / f"{model_name}.{suffix}"

        torch.save(model.state_dict(), model_path)
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

    print(f"Data saved in {str(result_path)}")



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