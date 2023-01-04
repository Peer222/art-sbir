import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn
import pandas as pd
import numpy as np

import torch
from PIL import Image
from PIL import ImageOps
import torchvision.transforms as transforms

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

class Color:
    BLACK = (0,0,0)
    BLUE = (55/256, 88/256, 136/256)
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
    plt.tight_layout()
    if not file:
        plt.show()
    else:
        plt.savefig(fname=file, dpi=300)
        plt.close()


# dataset.__getitem__() or dataset.load_image_sketch_tuple() -- expects torch.Tensor or PIL.Image in rgb
def show_triplets(triplets, filename:Path=None, mode='sketch') -> None:
    fig = plt.figure(figsize=(9, 2.7 * len(triplets)))

    rows, cols = len(triplets), 3

    inverted = [0, 0, 0]

    if mode == 'sketch': titles = ["Sketch", "Matching image", "Non-matching image"]
    elif mode == 'image': 
        titles = ['Image', 'Artificial sketch', 'Original sketch']
        #inverted = [0, 1, 0] #activate for semi supervised
    else: titles = ['','','']

    for i, tuple in enumerate(triplets):
        triplet = list(tuple)
        for j in range(3):
            fig.add_subplot(rows, cols, i * 3 + j + 1)
            
            plt.axis(False)
            if not i: plt.title(titles[j])

            if inverted[j]: triplet[j] = transforms.RandomInvert(p=1)(triplet[j])
            if type(triplet[j]) == torch.Tensor: 
                triplet[j] = triplet[j].squeeze().permute(1, 2, 0)# transforms.ToPILImage()(triplet[j]) # triplet[j].permute(1, 2, 0)

            plt.imshow(triplet[j])

            if 'sketch' in titles[j].lower(): # adds a frame
                add_frame(plt)

    plot(plt, filename)


def show_loss_curves(train_losses:List[float], test_losses:List[float], filename:Path=None, title="Loss curves") -> None:
    epochs = np.arange(1, len(train_losses) + 1, 1)
    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(epochs, train_losses, c=Color.YELLOW, label="Train loss")
    ax.plot(epochs, test_losses, c=Color.BLUE, label="Test loss")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    ax.grid(True, color=Color.LIGHT_GREY)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    ax.set_xlim(xmin=0)
    seaborn.despine(left=True, bottom=True, right=True, top=True)
    #plt.grid(visible=True, color=Color.LIGHT_GREY)

    plot(plt, filename)

# train and test losses with same keys | key is used as plot title
def build_all_loss_curves(train_losses:Dict, test_losses:Dict, result_path:Path, epoch:int=None, titles:List[str]=None) -> None:
    loss_path = result_path / f'loss_curves_{epoch}' if epoch else result_path / 'loss_curves'
    if not loss_path.is_dir(): loss_path.mkdir(parents=True, exist_ok=True)

    i = 0
    for key in train_losses.keys():
        loss = 'loss ' if not 'loss' in key else ''
        title = titles[i] if titles else f"{key.replace('_', ' ')} {loss}curves"
        show_loss_curves(train_losses[key], test_losses[key], loss_path / f'{key}.png', title)
        i += 1



def show_topk_accuracy(topk_acc:List[float], filename:Path=None, title:str=None) -> None:
    labels = [f"top_{k}" for k in range(1, 11)]

    topk_acc = [x * 100 for x in topk_acc]

    bar_labels = [f"{round(acc, 1):.1f} %" for acc in topk_acc]

    fig, ax = plt.subplots(figsize=(9,5))
    bars = ax.bar(labels, topk_acc, color=Color.BLUE)
    ax.bar_label(bars, bar_labels)

    if title: plt.title(title)
    else: plt.title("Top_k accuracy")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Top_k positions")

    plt.ylim([0, 100])

    ax.grid(True, color=Color.LIGHT_GREY)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    seaborn.despine(left=True, bottom=True, right=True, top=True)

    plot(plt, filename)

# show original has to be implemented
def show_retrieval_samples(samples:List[Tuple[Path, List[Path]]], show_original:bool=False, filename:Path=None, title:str=None) -> None:
    rows, cols = len(samples), 11
    
    heights = [1 for i in range(rows + 1)]
    heights[0] = 0
    fig, axes = plt.subplots(nrows=rows + 1, ncols=cols, figsize=(cols, rows), gridspec_kw={'height_ratios': heights})

    for i, axes_rows in enumerate(axes):
        if not i: continue
        sketch_path = list(samples[i - 1].keys())[0]
        image_paths = samples[i - 1][sketch_path]
        sketch_path = Path(sketch_path)

        for j, ax in enumerate(axes_rows):
            if not j:
                sketch = Image.open(sketch_path)
                ax.axis(False)
                ax.imshow(sketch)
                add_frame(ax)
            else:
                image_path = Path(image_paths[j - 1][0])
                image = Image.open(image_path)

                ax.axis(False)
                ax.imshow(image)
            
                if image_path.stem in sketch_path.stem:
                    add_frame(ax, space=60, linewidth=2.0, color=Color.GREEN)


    col_titles = ['Query', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    for ax, col_title in zip(axes[0], col_titles):
        ax.axis(False)
        ax.set_title(col_title, fontdict={'fontsize':10})

    if title: plt.suptitle(title)
    else: plt.suptitle("Retrieval samples")

    plot(plt, filename)


def load_file(file_path:Path):
    with open(file_path, 'r') as f:
        if file_path.suffix == ".json":
            return json.load(f)
        else:
            print(file_path.suffix)

# adds frame around current plot or axes (has to be called after plot is added)
def add_frame(ax, space=0, linewidth=0.4, color=Color.BLACK):
    if not 'AxesSubplot' in str(type(ax)): ax = ax.gca()
    autoAxis = ax.axis()

    rec = patches.Rectangle( (autoAxis[0]- space/2, autoAxis[2] + space/2), (autoAxis[1] - autoAxis[0]) + space, (autoAxis[3]-autoAxis[2]) - space, 
                            fill=False, lw=linewidth, color=color)
    rec = ax.add_patch(rec)
    rec.set_clip_on(False)


def visualize(folder_path:Path, training_dict:Dict=None, inference_dict:Dict=None):
    if training_dict: show_loss_curves(training_dict["train_losses"], training_dict['test_losses'], filename=folder_path / "loss_curves")
    if len(inference_dict.keys()) > 2: # kaggle inference on drawings and sketches
        show_topk_accuracy(inference_dict['topk_acc'], filename=folder_path / 'topk_accuracy')
        show_retrieval_samples(inference_dict['retrieval_samples'], show_original=False, filename=folder_path / 'retrieval_samples')
        show_retrieval_samples(inference_dict['retrieval_samples'], show_original=True, filename=folder_path / 'retrieval_samples_original') # works only with sketchy and anime_drawings
    elif len(inference_dict.keys()) == 2:
        show_topk_accuracy(inference_dict['drawing_stats']['topk_acc'], filename=folder_path / 'topk_accuracy_drawings', title="Top_k accuracy (Drawings)")
        show_topk_accuracy(inference_dict['sketch_stats']['topk_acc'], filename=folder_path / 'topk_accuracy_sketches', title="Top_k accuracy (Sketches)")
        show_retrieval_samples(inference_dict['drawing_stats']['retrieval_samples'], show_original=False, filename=folder_path / 'retrieval_samples_drawings', title="Retrieval samples (Drawings)")
        show_retrieval_samples(inference_dict['sketch_stats']['retrieval_samples'], show_original=False, filename=folder_path / 'retrieval_samples_sketches', title="Retrieval samples (Sketches)")


if __name__ == '__main__':
    #show_loss_curves([0, 1, 2], [1, 2, 3], 'test.png')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path)
    parser.add_argument('-m', '--method', default='visualize', choices=['visualize'])
    args = parser.parse_args()

    if args.method == 'visualize':
        with open(args.path / 'training.json', 'r') as f:
            training_dict = json.load(f)
        try:
            with open(args.path / 'inference_updated.json') as f:
                inference_dict = json.load(f)
        except:
            with open(args.path / 'inference.json') as f:
                inference_dict = json.load(f)

        visualize(args.path, training_dict, inference_dict)