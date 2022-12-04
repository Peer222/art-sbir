import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn
import pandas as pd
import numpy as np

import torch
from PIL import Image

import json
from pathlib import Path
from typing import List, Tuple, Dict

class Color:
    BLACK = (0,0,0)
    BLUE = (0, 88/256, 154/256)
    RED = (161/256, 34/256, 0)
    GREEN = (141/256, 201/256, 20/256) #(0, 124/256, 6/256)
    YELLOW = (227/256, 193/256, 0)
    LIGHT_GREY = (240/256, 240/256, 240/256)
    GREY = (200/256, 200/256, 200/256)


# creates a directory to store visualizations
visual_path = Path("visual")
if not visual_path.is_dir():
    visual_path.mkdir(parents=True, exist_ok=True)

#plots figure or saves figure if filepath is specified
def plot(plt, file: Path=None) -> None:
    if not file:
        plt.show()
    else:
        plt.savefig(fname=file, dpi=300)


# dataset.__getitem__() or dataset.load_image_sketch_tuple() -- expects torch.Tensor or PIL.Image
def show_triplets(triplets, filename:Path=None, mode='sketch') -> None:
    fig = plt.figure(figsize=(9, 3 * len(triplets)))

    rows, cols = len(triplets), 3

    if mode == 'sketch': titles = ["Sketch", "Matching image", "Non-matching image"]
    else: titles = ['Image', 'Sketch', 'Original sketch']

    for i, tuple in enumerate(triplets):
        triplet = list(tuple)
        for j in range(3):
            fig.add_subplot(rows, cols, i * 3 + j + 1)
            
            plt.axis(False)
            if not i: plt.title(titles[j])

            if type(triplet[j]) == torch.Tensor: triplet[j] = triplet[j].permute(1, 2, 0)

            plt.imshow(triplet[j])

            if 'sketch' in titles[j].lower(): # adds a frame
                add_frame(plt)

    plot(plt, filename)


def show_loss_curves(train_losses:List[float], test_losses:List[float], filename:Path=None, title="Loss curves") -> None:
    epochs = np.arange(1, len(train_losses) + 1, 1)
    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(epochs, train_losses, c=Color.BLUE, label="Train loss")
    ax.plot(epochs, test_losses, c=Color.YELLOW, label="Test loss")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    ax.grid(True, color=Color.LIGHT_GREY)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    seaborn.despine(left=True, bottom=True, right=True, top=True)
    #plt.grid(visible=True, color=Color.LIGHT_GREY)

    plot(plt, filename)


def show_topk_accuracy(topk_acc:List[float], filename:Path=None) -> None:
    labels = [f"top_{k}" for k in range(1, 11)]

    topk_acc = [x * 100 for x in topk_acc]

    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(labels, topk_acc, color=Color.BLUE)

    plt.title("Top_k accuracy")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Top_k positions")

    ax.grid(True, color=Color.LIGHT_GREY)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    seaborn.despine(left=True, bottom=True, right=True, top=True)

    plot(plt, filename)

# show original has to be implemented
def show_retrieval_samples(samples:List[Tuple[Path, List[Path]]], show_original:bool=False, filename:Path=None) -> None:
    rows, cols = len(samples), 11

    fig = plt.figure(figsize=(cols - 1, rows))

    for i in range(rows):
        sketch_path = list(samples[i].keys())[0]
        image_paths = samples[i][sketch_path]
        sketch_path = Path(sketch_path)

        sketch = Image.open(sketch_path)
        fig.add_subplot(rows, cols, i * cols + 1)
        plt.axis(False)
        
        plt.imshow(sketch)

        add_frame(plt)

        if not i:
            plt.title('Query', fontsize=8, y=1.1)

        for j in range(len(image_paths)):
            image_path = image_paths[j][0]

            if show_original:
                if 'photos' in image_path: return
                image_path = image_path.replace('anime_drawings', 'photos') # works only for sketchy !!!
                image_path = image_path.replace('contour_drawings', 'photos')
                image_path = image_path.replace('png', 'jpg') # original photos must have jpg format eventually incompatible with other datasets !!!

            image_path = Path(image_path)

            image = Image.open(image_path)

            if image_path.stem in sketch_path.stem:
                fig.add_subplot(rows, cols, i * cols + j + 2, facecolor=Color.GREEN)
            else:
                fig.add_subplot(rows, cols, i * cols + j + 2)
            plt.axis(False)

            plt.imshow(image)
            
            if image_path.stem in sketch_path.stem:
                add_frame(plt, space=30, linewidth=1.2, color=Color.GREEN)

            if not i:
                plt.title(str(j + 1), fontsize=8, y=1.1)
    
    #y pos may has to be adjusted
    plt.suptitle("Retrieval samples", y=1 - rows * 0.008)
    plot(plt, filename)


def load_file(file_path:Path):
    with open(file_path, 'r') as f:
        if file_path.suffix == ".json":
            return json.load(f)
        else:
            print(file_path.suffix)

# adds frame around current plot (has to be called after plot is added)
def add_frame(plt, space=0, linewidth=0.4, color=Color.BLACK):
    ax = plt.gca()
    autoAxis = plt.axis()
    rec = patches.Rectangle( (autoAxis[0]- space/2, autoAxis[2] + space/2), (autoAxis[1] - autoAxis[0]) + space, (autoAxis[3]-autoAxis[2]) - space, 
                            fill=False, lw=linewidth, color=color)
    rec = ax.add_patch(rec)
    rec.set_clip_on(False)


def visualize(folder_path:Path, training_dict:Dict=None, inference_dict:Dict=None):
    show_loss_curves(training_dict["train_losses"], training_dict['test_losses'], filename=folder_path / "loss_curves")
    show_topk_accuracy(inference_dict['topk_acc'], filename=folder_path / 'topk_accuracy')
    show_retrieval_samples(inference_dict['retrieval_samples'], show_original=False, filename=folder_path / 'retrieval_samples')
    show_retrieval_samples(inference_dict['retrieval_samples'], show_original=True, filename=folder_path / 'retrieval_samples_original') # works only with sketchy and anime_drawings


#inference_dict = load_file(Path('results/ModifiedResNet_with_classification_SketchyDatasetV2_2022-11-23_21-01/inference.json'))
#show_retrieval_samples(inference_dict['retrieval_samples'], show_original=True, filename=Path('results/ModifiedResNet_with_classification_SketchyDatasetV2_2022-11-23_21-01') / 'retrieval_samples_original')

"""
def plot_transformed_images(image_paths, transform, n=3, seed=42):
    Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list, 
                        transform=data_transform, 
                        n=3)
"""