from pathlib import Path
import csv
from datetime import datetime
from tqdm.auto import tqdm
from typing import List, Tuple
import re
from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

import data_preparation
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# just one sketch per call
def get_ranking_position(sketch_path:Path or str, image_paths:List[Path], sketch_feature:torch.Tensor, image_features:torch.Tensor) -> int:
    if type(sketch_path) == str: sketch_path = Path(sketch_path)

    sketch_name = re.split('-', sketch_path.stem)[0]
    pos_img_index = utils.find_image_index(image_paths, sketch_name)

    distances = utils.euclidean_distance(sketch_feature, image_features)
    _, indices = distances.topk(len(image_paths), largest=False)

    return (indices == pos_img_index).nonzero().squeeze().item()

# just one sketch per call
def get_topk_images(k:int, image_paths:List[Path], sketch_feature:torch.Tensor, image_features:torch.Tensor) -> List[Tuple[Path, float]]:
    distances = utils.euclidean_distance(sketch_feature, image_features)
    values, indices = distances.topk(k, largest=False)

    image_paths = [image_paths[i] for i in indices]
    values = [value.item() for value in values]
    return list(zip(image_paths, values))


def compute_image_features(model, dataset) -> Tuple[Dataset, torch.Tensor]:
    feature_path = Path("data/image_features")
    if not feature_path.is_dir(): feature_path.mkdir(parents=True, exist_ok=True)

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    feature_path = feature_path / f"{model.__class__.__name__}_{dataset.state_dict['dataset']}_{date_time}"
    feature_path.mkdir(parents=True, exist_ok=True)

    inference_dataset = data_preparation.InferenceDataset(dataset.photo_paths, dataset.transform)

    str_paths = [ [str(path)] for path in inference_dataset.image_paths]
    with open(feature_path / "image_paths.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(str_paths)
        print(f"Image paths saved in {feature_path / 'image_paths.csv'}")

    dataloader = DataLoader(dataset=inference_dataset, batch_size=50, num_workers=0, shuffle=False)

    image_features = torch.Tensor().to(device)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        for images in tqdm(dataloader):
            images = images.to(device)
            image_features = torch.cat(( image_features, model(images).squeeze() ))

    image_features = image_features.cpu()

    with open(feature_path / "image_features.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(image_features.numpy())

    print(f"Image features saved in {feature_path / 'image_features.csv'}")

    return inference_dataset, image_features


def load_image_features(folder_name:str, transform=utils.ResNet50m_img_transform) -> Tuple[Dataset, torch.Tensor]:
    path = Path("data/image_features") / folder_name
    image_paths = list(pd.read_csv(path / "image_paths.csv").values)
    image_paths = [Path(img_path[0]) for img_path in image_paths]
    image_features = pd.read_csv(path / "image_features.csv").values
    return data_preparation.InferenceDataset(image_paths, transform), torch.from_numpy(image_features)

data, image_features = load_image_features("ModifiedResNet_SketchyDatasetV1_2022-11-15_17-27")
model = utils.load_model("CLIP_ResNet-50.pt")
sketch_feature = model(utils.ResNet50m_img_transform(Image.open("data/sketchy/sketches_png/airplane/n02691156_10153-1.png")).unsqueeze(0))

print(get_ranking_position("data/sketchy/sketches_png/airplane/n02691156_10153-1.png", data.image_paths, sketch_feature, image_features))
#print(get_topk_images(5, data.image_paths, sketch_feature, image_features))