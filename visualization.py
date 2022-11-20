import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np

import torch
from PIL import Image

from pathlib import Path

from typing import List, Tuple, Dict

class Color:
    BLACK = (0,0,0)
    BLUE = (0, 88/256, 154/256)
    RED = (161/256, 34/256, 0)
    GREEN = (0, 124/256, 6/256)
    YELLOW = (227/256, 193/256, 0)
    LIGHT_GREY = (240/256, 240/256, 240/256)


# creates a directory to store visualizations
visual_path = Path("visual")
if not visual_path.is_dir():
    visual_path.mkdir(parents=True, exist_ok=True)

#plots figure or saves figure if filepath is specified
def plot(plt, file: Path=None) -> None:
    if not file:
        plt.show()
    else:
        plt.savefig(fname=file, dpi=200)


# dataset.__getitem__() or dataset.load_image_sketch_tuple() -- expects torch.Tensor or PIL.Image
def show_triplets(triplets, filename:Path=None) -> None:
    fig = plt.figure(figsize=(9, 3 * len(triplets)))

    rows, cols = len(triplets), 3

    titles = ["Sketch", "Matching image", "Non-matching image"]

    for i, tuple in enumerate(triplets):
        triplet = list(tuple)
        for j in range(3):
            fig.add_subplot(rows, cols, i * 3 + j + 1)
            
            plt.axis(False)
            if not i: plt.title(titles[j])

            if type(triplet[j]) == torch.Tensor: triplet[j] = triplet[j].permute(1, 2, 0)

            if not j: # adds a frame
                triplet[j] = np.pad(triplet[j], [(1,1), (1,1), (0,0)])

            plt.imshow(triplet[j])

    plot(plt, filename)


def plot_loss_curves(train_losses:List[float], test_losses:List[float], filename:Path=None, title="Loss curves") -> None:
    epochs = np.arange(1, len(train_losses) + 1, 1)
    plt.figure(figsize=(7,5))

    plt.plot(epochs, train_losses, c=Color.BLUE, label="Train loss")
    plt.plot(epochs, test_losses, c=Color.YELLOW, label="Test loss")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    seaborn.despine(left=True, bottom=True, right=True, top=True)
    plt.grid(visible=True, color=Color.LIGHT_GREY)

    plot(plt, filename)


plot_loss_curves([0.5, 0.31, 0.22, 0.17, 0.14, 0.12, 0.11, 0.09], [0.4, 0.3, 0.25, 0.2, 0.15, 0.18, 0.12, 0.16])

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