from pathlib import Path
import argparse
from datetime import datetime
from tqdm.auto import tqdm
from timeit import default_timer as timer
from typing import List, Tuple, Dict
import re
import random
from PIL import Image

import os
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

import visualization
import data_preparation
import utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# just one sketch per call - starts at 0 ?
def get_ranking_position(sketch_path:Path or str, image_paths:List[Path], sketch_feature:torch.Tensor, image_features:torch.Tensor) -> int:
    if type(sketch_path) == str: sketch_path = Path(sketch_path)

    sketch_name = re.split('-', sketch_path.stem)
    if len(sketch_name) <= 2: sketch_name = sketch_name[0] # sketchy sketch names have format id-number.png | kaggle sketch names id.png
    elif len(sketch_name) == 3: sketch_name = sketch_name[1] # sketchit sketch names have format index-id-random_number.png
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


def compute_image_features(model, dataset, with_classification:bool) -> Tuple[Dataset, torch.Tensor]:

    inference_dataset = data_preparation.InferenceDataset(dataset.photo_paths, dataset.transform)

    dataloader = DataLoader(dataset=inference_dataset, batch_size=50, num_workers=0, shuffle=False)

    image_features = torch.Tensor().to(device)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        for images in tqdm(dataloader, desc="Computing Image Features"):
            images = images.to(device)
            if with_classification: image_features = torch.cat(( image_features, model(images)[0].squeeze() ))
            else: image_features = torch.cat(( image_features, model(images).squeeze() ))

    image_features = image_features.cpu()

    feature_path = utils.save_image_features(model.__class__.__name__, dataset.state_dict['dataset'], inference_dataset, image_features)

    return inference_dataset, image_features, feature_path

def process_inference(model, dataset, inference_dataset, dataloader, image_features, start_time, with_classification):
    ranks = []
    mean_reciprocal_rank = 0
    k = 10
    topk_acc = np.zeros(k)

    retrieval_samples = []
    random.seed(10)
    random_indices = [random.randrange(0, len(dataset)) for _ in range(10)]

    image_features = image_features.to(device)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        # because shuffle=False and batch_size = 1 i is the index of the sketch path in dataset
        for i, tuple in enumerate(tqdm(dataloader, desc="Inference")):
            if with_classification: sketch_feature = model(tuple[0].to(device))[0] # tuple[0] = sketch
            else: sketch_feature = model(tuple[0].to(device)) # tuple[0] = sketch

            rank = get_ranking_position(dataset.sketch_paths[i], inference_dataset.image_paths, sketch_feature, image_features)

            # rank starts at 0
            ranks.append(rank + 1)
            mean_reciprocal_rank += 1/(rank + 1)
            if rank < 10: topk_acc[rank:] += 1

            if random_indices.count(i) > 0:
                retrieval_samples.append({str(dataset.sketch_paths[i]): get_topk_images(k, inference_dataset.image_paths, sketch_feature, image_features)})

    rankings = pd.DataFrame(ranks, columns=['rank'])
    mean_reciprocal_rank /= len(dataset)
    topk_acc /= len(dataset)

    time = timer() - start_time

    stats = {"mean_reciprocal_rank": mean_reciprocal_rank, "size": len(inference_dataset), "inference_time": time}
    pandas_stats = rankings.describe().to_dict()['rank']
    for key in pandas_stats.keys():
        stats[key] = pandas_stats[key]
    stats["topk_acc"] = list(topk_acc)
    stats["retrieval_samples"] = retrieval_samples

    return stats


# dataset: test data, folder_name: if specified image_features will be loaded from file instead of computed
def run_inference(model, dataset, folder_name:str=None) -> Dict:
    start_time = timer()

    with_classification = 'with_classification' in type(model).__name__

    if folder_name:
        feature_folder = folder_name
        image_paths, image_features = utils.load_image_features(folder_name)
        inference_dataset = data_preparation.InferenceDataset(image_paths, model.transform)
        print("Image features loaded from file")
    else:
        inference_dataset, image_features, feature_folder = compute_image_features(model, dataset, with_classification)

    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)

    inference_dict = process_inference(model, dataset, inference_dataset, dataloader, image_features, start_time, with_classification)
    inference_dict2 = {}
    if 'Kaggle' in dataset.state_dict['dataset'] or 'Mixed' in dataset.state_dict['dataset']:
        _, dataset2 = data_preparation.get_datasets('KaggleInferenceV1', sketch_type='sketches', transform=dataset.transform)
        dataloader2 = DataLoader(dataset=dataset2, batch_size=1, num_workers=0, shuffle=False)
        inference_dict2 = process_inference(model, dataset2, inference_dataset, dataloader2, image_features, inference_dict['inference_time'], with_classification)
    else: 
        inference_dict['image_features'] = feature_folder
        return inference_dict

    return {'image_features': feature_folder, 'drawing_stats': inference_dict, 'sketch_stats': inference_dict2}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='recomputes Inference for given folder')

    parser.add_argument('--folder', default=None, help="Folder on which rerunning inference")
    parser.add_argument('-a', '--all', action="store_true", help="Rerun inference for all Modified_ResNet* models where results folder exist")

    args = parser.parse_args()

    FOLDERS = [] if not args.folder else [args.folder]
    if args.all:
        #FOLDERS = next(os.walk('./data/image_features/'))[1]
        FOLDERS = Path("./models").glob("ModifiedResNet*.pth")
        FOLDERS = [path.stem for path in FOLDERS]

        #FOLDERS = [folder for folder in FOLDERS if "Kaggle" in folder or "Mixed" in folder]
    print(FOLDERS, flush=True)

    for FOLDER in FOLDERS:

        MODEL = FOLDER + '.pth'
        MODEL_TYPE = FOLDER.split('_')[0] if len(FOLDER.split('_')) == 4 else "ModifiedResNet_with_classification"

        if not Path(f"models/{MODEL}").is_file():
            print(f"Model {MODEL} is not available", flush=True)
            continue

        if not Path(f"results/{FOLDER}").is_dir():
            print(f"Results {FOLDER} are not available", flush=True)
            continue

        with open(Path("results") / FOLDER / "data_params.json", 'r') as f:
            data_dict = json.load(f)

        with open(Path("results") / FOLDER / "training.json", 'r') as f:
            training_dict = json.load(f)

        inference_file = "inference_updated.json" if (Path("results") / FOLDER / "inference_updated.json").is_file() else "inference.json"
        print(inference_file)
        with open(Path("results") / FOLDER / inference_file, 'r') as f:
            inference_dict_ = json.load(f)

        DATASET = data_dict['dataset'] if not 'Mixed' in data_dict['dataset'] else data_dict['dataset'] + data_dict['version']

        print(DATASET, MODEL, MODEL_TYPE)

        model = utils.load_model(MODEL, dataset=DATASET, model_type=MODEL_TYPE)
        model.to(device)


        # options have to be added
        if 'Mixed' in DATASET:
            _, test_dataset = data_preparation.get_datasets(dataset=DATASET, size=data_dict['size'])
        elif 'Kaggle' in DATASET:
            _, test_dataset = data_preparation.get_datasets(dataset=DATASET, size=data_dict['size'], sketch_type=data_dict['sketch_type'], img_type=data_dict['img_type'], img_format=data_dict['img_format'], transform=model.transform)
        elif 'Sketchy' in DATASET:
            _, test_dataset = data_preparation.get_datasets(dataset=DATASET, size=data_dict['size'], img_type=data_dict['img_type'], img_format=data_dict['img_format'], transform=model.transform)

        #print(test_dataset.state_dict)

        feature_folder = inference_dict_['image_features'] if 'image_features' in inference_dict_.keys() else None

        inference_dict = run_inference(model, test_dataset, feature_folder)

        with open(Path("results") / FOLDER / "inference_updated.json", "w") as f:
            json.dump(inference_dict, f, indent=4)

        # saves visualizations in result folder
        visualization.visualize(Path("results") / FOLDER, training_dict, inference_dict)

        print(f"RUN INFERENCE AND VISUALIZATION FOR {FOLDER}", flush=True)