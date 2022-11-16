from pathlib import Path
import csv
from datetime import datetime
from tqdm.auto import tqdm
from timeit import default_timer as timer
from typing import List, Tuple, Dict
import re
import random
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

# just one sketch per call - starts at 0 ?
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

# dataset: test data, folder_name: if specified image_features will be loaded from file instead of computed
def run_inference(model, dataset, folder_name:str=None) -> Dict:
    start_time = timer()

    inference_dataset, image_features = None, None
    if folder_name:
        inference_dataset, image_features = load_image_features(folder_name)
    else:
        inference_dataset, image_features = compute_image_features(model, dataset)

    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)

    avg_rank = 0
    mean_reciprocal_rank = 0
    k = 10
    topk_acc = np.zeros(k)

    retrieval_samples = None
    random_indices = [random.randrange(0, len(dataset)) for _ in range(5)]
    model.to(device)
    model.eval()
    with torch.inference_mode():
        # because shuffle=False and batch_size = 1 i is the index of the sketch path in dataset
        for i, sketch in enumerate(tqdm(dataloader, desc="Inference")):
            sketch_feature = model(sketch.to(device))

            rank = get_ranking_position(dataset.sketch_paths[i], inference_dataset.image_paths, sketch_feature, image_features)

            avg_rank += rank
            mean_reciprocal_rank += 1/rank
            if rank < 10: topk_acc[rank:] += 1

            if random_indices.count(i) > 0:
                retrieval_samples.append({dataset.sketch_paths[i]: get_topk_images(10, inference_dataset.image_paths, sketch_feature, image_features)})

    avg_rank /= len(inference_dataset)
    mean_reciprocal_rank /= len(inference_dataset)
    topk_acc /= len(inference_dataset)

    time = timer() - start_time

    return {"avg_rank": avg_rank, "mean_reciprocal_rank": mean_reciprocal_rank, "topk_acc": topk_acc, "retrieval_samples": retrieval_samples, "size": len(inference_dataset), "inference_time": time}

"""
data, image_features = load_image_features("ModifiedResNet_SketchyDatasetV1_2022-11-15_17-27")
model = utils.load_model("CLIP_ResNet-50.pt")
sketch_feature = model(utils.ResNet50m_img_transform(Image.open("data/sketchy/sketches_png/airplane/n02691156_10153-1.png")).unsqueeze(0))

print(get_ranking_position("data/sketchy/sketches_png/airplane/n02691156_10153-1.png", data.image_paths, sketch_feature, image_features))
#print(get_topk_images(5, data.image_paths, sketch_feature, image_features))
"""

if __name__ == "__main__":
    # command line tool to run inference only
    pass