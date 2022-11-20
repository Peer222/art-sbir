from pathlib import Path
import argparse
from datetime import datetime
from tqdm.auto import tqdm
from timeit import default_timer as timer
from typing import List, Tuple, Dict
import re
import random
from PIL import Image

import numpy as np

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
    if pos_img_index < 0:
        print(f"No image found: {sketch_path} | {sketch_name}")
        return len(image_paths)

    distances = utils.euclidean_distance(sketch_feature, image_features)
    _, indices = distances.topk(len(image_features), largest=False)

    try:
        ranking = (indices == pos_img_index).nonzero().squeeze().item()
    except:
        # in case multiple images exist for a sketch
        ranking = (indices == pos_img_index).nonzero().squeeze()[0].item()
        print(f"Multiple images found: {sketch_path}")
    return ranking

# just one sketch per call
def get_topk_images(k:int, image_paths:List[Path], sketch_feature:torch.Tensor, image_features:torch.Tensor) -> List[Tuple[str, float]]:
    distances = utils.euclidean_distance(sketch_feature, image_features)
    values, indices = distances.topk(k, largest=False)

    image_paths = [str(image_paths[i]) for i in indices]
    values = [value.item() for value in values]
    return list(zip(image_paths, values))


def compute_image_features(model, dataset) -> Tuple[Dataset, torch.Tensor]:

    inference_dataset = data_preparation.InferenceDataset(dataset.photo_paths, dataset.transform)

    dataloader = DataLoader(dataset=inference_dataset, batch_size=50, num_workers=0, shuffle=False)

    image_features = torch.Tensor().to(device)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        for images in tqdm(dataloader):
            images = images.to(device)
            image_features = torch.cat(( image_features, model(images).squeeze() ))

    image_features = image_features.cpu()

    utils.save_image_features(model.__class__.__name__, dataset.state_dict['dataset'], inference_dataset, image_features)

    return inference_dataset, image_features


# dataset: test data, folder_name: if specified image_features will be loaded from file instead of computed
def run_inference(model, dataset, folder_name:str=None) -> Dict:
    start_time = timer()

    inference_dataset, image_features = None, None
    if folder_name:
        inference_dataset, image_features = utils.load_image_features(folder_name, model.transform)
        print("Image features loaded from file")
    else:
        inference_dataset, image_features = compute_image_features(model, dataset)

    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)

    avg_rank = 0
    mean_reciprocal_rank = 0
    k = 10
    topk_acc = np.zeros(k)

    retrieval_samples = []
    random_indices = [random.randrange(0, len(dataset)) for _ in range(5)]

    image_features = image_features.to(device)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        # because shuffle=False and batch_size = 1 i is the index of the sketch path in dataset
        for i, (sketch, _, _) in enumerate(tqdm(dataloader, desc="Inference")):
            sketch_feature = model(sketch.to(device))

            rank = get_ranking_position(dataset.sketch_paths[i], inference_dataset.image_paths, sketch_feature, image_features)

            # rank starts at 0
            avg_rank += rank + 1
            mean_reciprocal_rank += 1/(rank + 1)
            if rank < 10: topk_acc[rank:] += 1

            if random_indices.count(i) > 0:
                retrieval_samples.append({str(dataset.sketch_paths[i]): get_topk_images(k, inference_dataset.image_paths, sketch_feature, image_features)})

    avg_rank /= len(inference_dataset)
    mean_reciprocal_rank /= len(inference_dataset)
    topk_acc /= len(inference_dataset)

    time = timer() - start_time

    return {"avg_rank": avg_rank, "mean_reciprocal_rank": mean_reciprocal_rank, "topk_acc": list(topk_acc), "retrieval_samples": retrieval_samples, "size": len(inference_dataset), "inference_time": time}


if __name__ == "__main__":
    # command line tool to run inference only
    parser = argparse.ArgumentParser(description="TODO")

    parser.add_argument("-m", "--model", type=str, required=True, help="file name of model in models/")
    parser.add_argument("-f", "--folder_name", type=str, required=True, help="corresponding folder in data/image_features/")
    parser.add_argument("-d", "--dataset", type=str, default="SketchyDatasetV1", help="corresponding folder in data/image_features/")
    parser.add_argument("-s", "--dsize", type=float, default=0.01, help="fraction of elements used from dataset")
    
    args = parser.parse_args()

    model = utils.load_model(args.model)

    dataset_name = re.findall("\w+_(\w+)_\w+", args.folder_name)[0]

    dataset = None
    if dataset_name == args.dataset: _, dataset = data_preparation.get_datasets(size=args.dsize) #test dataset

    if not dataset: raise ValueError("no dataset found")

    inference_dict = run_inference(model, dataset, args.folder_name)

    utils.save_model(model=model, data_dict=dataset.state_dict, inference_dict=inference_dict)