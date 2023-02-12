import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from PIL import ImageOps
import torchvision.transforms as transforms

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import semiSupervised_utils
import utils
import data_preparation

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
    #plt.tight_layout()
    if not file:
        plt.show()
    else:
        if not file.parent.is_dir():
            file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fname=file, dpi=300, bbox_inches='tight')
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


def show_loss_curves(train_losses:List[float], test_losses:List[float], filename:Path=None, title=None, x_label='Epoch') -> None:
    if x_label == 'Iteration':
        epochs = np.arange(10000, (len(train_losses) + 1) * 10000, 10000)
    else:
        epochs = np.arange(1, len(train_losses) + 1, 1)

    
    fig, ax = plt.subplots(figsize=(7,3.5))

    ax.plot(epochs, train_losses, c=Color.YELLOW, label="Train loss")
    ax.plot(epochs, test_losses, c=Color.BLUE, label="Test loss")

    if title: plt.title(title)
    plt.xlabel("Epoch")# x_label)
    plt.ylabel("Loss")
    plt.legend()

    ax.grid(True, color=Color.LIGHT_GREY)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    #ax.set_xlim(xmin=0)
    if x_label != 'Iteration':
        ax.set_xticks(np.arange(0, len(epochs) + 1, 1))
    else:
        ax.set_xticks(np.array([0.0, 0.33, 0.66, 1.0]) * epochs[-1])
        ax.set_xticklabels(np.arange(0, 3 + 1, 1))
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
    labels = [f"top-{k}" for k in range(1, 11)]

    topk_acc = [x * 100 for x in topk_acc]

    bar_labels = [f"{round(acc, 1):.1f}" for acc in topk_acc]

    plt.rcParams.update({'font.size': 16})
    label_font = {'size':'13'}

    fig, ax = plt.subplots(figsize=(10,4))
    bars = ax.bar(labels, topk_acc, color=Color.BLUE, label="Sketches")
    ax.bar_label(bars, bar_labels, padding=2, **label_font)

    if title: plt.title(title)
    #else: plt.title("Top-k accuracy")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Top-k positions")
    plt.legend()

    plt.ylim([0, 100])

    ax.grid(True, color=Color.LIGHT_GREY)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    seaborn.despine(left=True, bottom=True, right=True, top=True)

    plot(plt, filename)

def show_compared_topk_accuracy(topk_acc:List[float], topk_acc2:List[float], filename:Path=None, title:str=None) -> None:
    labels = [f"top-{k}" for k in range(1, 11)]

    topk_acc = [x * 100 for x in topk_acc]
    topk_acc2 = [x * 100 for x in topk_acc2]

    bar_labels = [f"{round(acc, 1):.1f}" for acc in topk_acc]
    bar_labels2 = [f"{round(acc, 1):.1f}" for acc in topk_acc2]

    x = np.arange(len(bar_labels)) * 3
    width = 1.3
    margin = 0.03

    plt.rcParams.update({'font.size': 18})
    label_font = {'size':'14'}

    fig, ax = plt.subplots(figsize=(16,4))
    bars1 = ax.bar(x-width/2 - margin, topk_acc, width=width, label="Sketches", color=Color.BLUE)
    bars2 = ax.bar(x+width/2 + margin, topk_acc2, width=width, label="Drawings", color=Color.GREY)
    ax.bar_label(bars1, bar_labels, padding=2, **label_font)
    ax.bar_label(bars2, bar_labels2, padding=2, **label_font)

    if title: plt.title(title)
    #else: plt.title("Top-k accuracy")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Top-k positions")
    plt.legend()

    plt.ylim([0, 100])

    ax.grid(True, color=Color.LIGHT_GREY)
    ax.set_xticks(x, labels)
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
                # drawings only available on server
                if not torch.cuda.is_available() and  "drawings" in str(sketch_path):
                    print("drawings skipped")
                    return
                sketch = Image.open(sketch_path)
                ax.axis(False)
                ax.imshow(sketch)
                add_frame(ax)
            else:
                image_path = Path(image_paths[j - 1][0])
                if not torch.cuda.is_available() and "kaggle" in str(image_path):
                    image_path = Path("../sketchit/public/paintings/") / image_path.name
                image = Image.open(image_path)

                ax.axis(False)
                ax.imshow(image)

                sketch_name = sketch_path.stem.split("-")[1] if len(sketch_path.stem.split("-")) == 3 else sketch_path.stem.split("-")[0]
                if image_path.stem.split('-')[0] == sketch_name:
                    add_frame(ax, space=20, linewidth=2.0, color=Color.GREEN)


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
    if training_dict and training_dict['iteration_loss_frequency'] > 0: show_loss_curves(training_dict["itrain_losses"], training_dict["itest_losses"], filename=folder_path / "loss_curves_iter", x_label="Iteration")
    if len(inference_dict.keys()) > 3: # kaggle inference on drawings and sketches
        show_retrieval_samples(inference_dict['retrieval_samples'], show_original=False, filename=folder_path / 'retrieval_samples')
        show_retrieval_samples(inference_dict['retrieval_samples'], show_original=True, filename=folder_path / 'retrieval_samples_original') # works only with sketchy and anime_drawings
        show_topk_accuracy(inference_dict['topk_acc'], filename=folder_path / 'topk_accuracy')    
    elif len(inference_dict.keys()) == 3:
        show_retrieval_samples(inference_dict['drawing_stats']['retrieval_samples'], show_original=False, filename=folder_path / 'retrieval_samples_drawings', title="Retrieval samples (Drawings)")
        show_retrieval_samples(inference_dict['sketch_stats']['retrieval_samples'], show_original=False, filename=folder_path / 'retrieval_samples_sketches', title="Retrieval samples (Sketches)")
        show_topk_accuracy(inference_dict['drawing_stats']['topk_acc'], filename=folder_path / 'topk_accuracy_drawings', title="Top_k accuracy (Drawings)")
        show_topk_accuracy(inference_dict['sketch_stats']['topk_acc'], filename=folder_path / 'topk_accuracy_sketches', title="Top_k accuracy (Sketches)")

### paper visuals

def image_comparison(cols, images1, images2=None, images3=None, images4=None, filepath=Path("test.png"), frame=[False,False,False, False]):
    rows = 0
    for images in [images1, images2, images3, images4]:
        if images: rows += 1
    #heights = [1 for i in range(rows + 1)]
    #heights[0] = 0
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 2, rows * 2))#, gridspec_kw={'height_ratios': heights})
    if not isinstance(axes[0], np.ndarray): axes = [axes]
    for i, axes_rows in enumerate(axes):
        for j, ax in enumerate(axes_rows):
            ax.axis(False)
            if not i:
                ax.imshow(images1[j - 1])
            elif i == 1:
                ax.imshow(images2[j - 1])
            elif i == 2:
                ax.imshow(images3[j - 1])
            else:
                ax.imshow(images4[j - 1])
            if frame[i]: add_frame(ax)
    plot(plt, filepath)

def get_vector_sketch(path:Path):
    if type(path) == str: path = Path(path)
    sketch = semiSupervised_utils.load_tuple_representation(path)
    return 255 - semiSupervised_utils.batch_rasterize_relative(torch.Tensor(sketch).unsqueeze(0)).squeeze().permute(1, 2, 0)

def vector_sketches():
    # vector_sketches folder from figures/experiments
    image_paths = ['vector_sketches/image', 'vector_sketches/sketch', 'vector_sketches/photo']
    images = [[], [], []]
    for i, path in enumerate(image_paths):
        paths = sorted(Path(path).glob("*.png"))
        print(paths)
        for p in paths:
            images[i].append(Image.open(p))
    image_comparison(5,images[0], images[1], images[2], filepath=Path("vector-sketches.png"))

def parsed_sketches():
    sketch_paths = ['airplane/n02691156_7989-8.png', 'apple/n07739125_8773-5.png', 'rhinoceros/n02391994_3673-5.png', 'windmill/n04587559_8803-6.png', 'teddy_bear/n04399382_6231-5.png']
    images1 = []
    images2 = []
    for i in range(5):
        images1.append(Image.open(Path("data/sketchy/sketches_png") / sketch_paths[i]))
    for i in range(5):
        sketch = semiSupervised_utils.load_tuple_representation(Path("data/sketchy/example_sketches") / f"{sketch_paths[i].split('.')[0].split('/')[1]}.json")['image']
        rasterized_sketch = 255 - semiSupervised_utils.batch_rasterize_relative(torch.Tensor(sketch).unsqueeze(0)).squeeze().permute(1, 2, 0)
        images2.append(rasterized_sketch)
    
    image_comparison(5, images1, images2, filepath=Path("parsed-sketches.png"), frame=[0, 1, 0, 0])

def sketch_samples():
    # sketch_samples folder from figures/data
    sketch_paths = sorted(Path("./sketch_samples/sketches").glob("*.png"))
    image_paths = sorted(Path("./sketch_samples/images").glob("*.jpg"))

    images1 = [Image.open(path) for path in image_paths]
    images2 = [Image.open(path) for path in sketch_paths]
    image_comparison(5, images1, images2, filepath=Path("sketch-samples.png"), frame=[0, 1, 0])

def synthetic_sketches():
    image_paths = sorted(Path("./sketch_samples/images").glob("*.jpg"))
    sketch_paths = sorted(Path("./sketch_samples/contour").glob("*.png"))
    sketch_paths2 = sorted(Path("./sketch_samples/opensketch").glob("*.png"))
    sketch_paths3 = sorted(Path("./sketch_samples/dilated").glob("*.png"))
    images1 = [Image.open(path) for path in image_paths]
    images2 = [Image.open(path) for path in sketch_paths]
    images3 = [Image.open(path) for path in sketch_paths2]
    images4 = [Image.open(path) for path in sketch_paths3]

    images4 = [image.convert("RGB") for image in images4]
    
    image_comparison(5, images1, images2, images3, images4, Path("synthetic-sketches.png"), frame=[0, 1, 1, 1])

def transformed_sketches():
    # transformations folder from figures/methology
    sketch_paths = sorted(Path("./transformations/").glob("transformed_*.png"))
    sketch_paths.append(Path("./transformations/original.png"))
    images1 = [Image.open(path) for path in sketch_paths]

    image_comparison(5, images1, filepath=Path("transformed-sketches.png"), frame=[1])

def synthetic_artworks():
    # generated_artworks folder from figures/experiments
    p = Path("./generated_artworks")
    image_paths = sorted(path for path in p.glob("*.jpg") if '-' not in str(path))

    images1 =[Image.open(path) for path in p.glob(f"{image_paths[0].stem}-*.jpg")]
    images1.append(Image.open(image_paths[0]))
    images2 =[Image.open(path) for path in p.glob(f"{image_paths[1].stem}-*.jpg")]
    images2.append(Image.open(image_paths[1]))
    images3 =[Image.open(path) for path in p.glob(f"{image_paths[2].stem}-*.jpg")]
    images3.append(Image.open(image_paths[2]))

    image_comparison(5, images1, images2, images3, filepath=Path("artwork-samples.png"))

def quickdraw_sketches():
    #folder1, filepath = "./results/Photo2Sketch_QuickDrawDatasetV1_2022-12-15_00-32/tuples_0/", Path("quickdraw-sketches.png")
    parser = argparse.ArgumentParser(description='Photo2Sketch')

    parser.add_argument('--setup', type=str, default='Sketchy')
    parser.add_argument('--batchsize', type=int, default=64) # previous 1 / paper used 64
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument("-m")

    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--eval_freq_iter', type=int, default=1000)

    parser.add_argument('--enc_rnn_size', type=int, default=256)
    parser.add_argument('--dec_rnn_size', type=int, default=512)
    parser.add_argument('--z_size', type=int, default=128)
    parser.add_argument('--num_mixture', type=int, default=20)

    parser.add_argument('--input_dropout_prob', type=float, default=0.9)
    parser.add_argument('--output_dropout_prob', type=float, default=0.9)
    parser.add_argument('--batch_size_sketch_rnn', type=int, default=100)

    parser.add_argument('--kl_weight_start', type=float, default=0.01)
    parser.add_argument('--kl_decay_rate', type=float, default=0.99995)
    parser.add_argument('--kl_tolerance', type=float, default=0.2)
    parser.add_argument('--kl_weight', type=float, default=1.0)

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--decay_rate', type=float, default=0.9999)
    parser.add_argument('--min_learning_rate', type=float, default=0.00001)
    parser.add_argument('--grad_clip', type=float, default=1.)

    parser.add_argument('--save_rate', type=int, default=30)

    hp = parser.parse_args()

    model = utils.load_model("Photo2Sketch_QuickDrawDatasetV1_2022-12-20_10-13.pth", "Quickdraw", 'Photo2Sketch', max_seq_len=100, options=hp)
    dataset = data_preparation.get_datasets("QuickdrawV1", size=1.0)[0]
    print(len(dataset))
    images1 = []
    images2 = []
    for i in range(5):
        item = dataset.__getitem__(i * len(dataset)//5)
        rgb_image = item['photo']
        sketch_vector = item['sketch_vector'].unsqueeze(0).permute(1, 0, 2).float()
        length_sketch = item['length'] - 1

        backbone_feature, rgb_encoded_dist = model.Image_Encoder(rgb_image.unsqueeze(0))
        rgb_encoded_dist_z_vector = rgb_encoded_dist.rsample()

        photo2sketch_output, attention_plot = model.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, sketch_vector, length_sketch + 1, isTrain=False)

        rasterized_sketch = 255 - semiSupervised_utils.batch_rasterize_relative(photo2sketch_output).squeeze().permute(1, 2, 0)

        images1.append(rgb_image.permute(1, 2, 0))
        images2.append(rasterized_sketch)

    print(len(images1), len(images2))

    #images2 = get_vector_sketches(folder2)
    image_comparison(5, images1, images2, filepath=Path("quickdraw-sketches.png"))

# activation functions
def plot_function(x_values, y_values, name, color=Color.BLUE, labels:Dict={'x':'x', 'y':'y'}, step_sizes:Dict={'x':1, 'y':1}, padding:Dict={'x':[0, 0], 'y':[1, 0]}):
    plt.figure(figsize=(4, 2))

    plt.plot(x_values, y_values, c=color)
    ax = plt.gca()
    ax.grid(True, color=Color.LIGHT_GREY, zorder=0)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    seaborn.despine(left=True, bottom=True, right=True, top=True)

    if labels:
        plt.xlabel(labels['x'], labelpad=7, fontsize=10)
        plt.ylabel(labels['y'], rotation=0, labelpad=7, fontsize=10)

    plt.xticks(np.arange(min(x_values) - padding['x'][0], max(x_values)+1 + padding['x'][1], step_sizes['x']), fontsize=8)
    plt.yticks(np.arange(round(min(y_values)) - padding['y'][0], round(max(y_values))+1 + padding['y'][1], step_sizes['y']), fontsize=8)

    plt.axhline(0, color=Color.BLACK, linewidth=0.6, zorder=1)
    plt.axvline(0, color=Color.BLACK, linewidth=0.6, zorder=1)

    plt.tight_layout(pad=-5)
    plot(plt, name)

def sigmoid():
    x_values = np.arange(-5, 5 + 0.1, 0.1)
    y_values = [1 / (1 + np.e ** (-x)) for x in x_values]
    plot_function(x_values, y_values, Path("../thesis/figures/foundations/sigmoid"), padding={'x':[0,0], 'y': [1, 1]})

def relu():
    x_values = [-5, 0, 5]
    y_values = [0, 0, 5]
    plot_function(x_values, y_values, Path("../thesis/figures/foundations/ReLU"))

def gelu():
    x_values = np.arange(-5, 5 + 0.1, 0.1)
    y_values = nn.GELU()(torch.Tensor(x_values)).tolist()
    plot_function(x_values, y_values, Path("../thesis/figures/foundations/GELU"))

def topk_kaggle(inference_dict):
    show_compared_topk_accuracy(inference_dict['sketch_stats']['topk_acc'], inference_dict['drawing_stats']['topk_acc'], Path("topk_acc.png"))

if __name__ == '__main__':
    #show_loss_curves([0, 1, 2], [1, 2, 3], 'test.png')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, default=None)
    parser.add_argument('-m', '--method', default='visualize', choices=['visualize', 'sigmoid', 'relu', 'gelu', 'parsed_sketches', 'quickdraw_sketches', 'vector_sketches', "sketch_samples", 'synthetic_sketches', 'topk_kaggle', 'transformed_sketches', 'synthetic_artworks'])
    args = parser.parse_args()

    if args.path:
        args.path = Path('results/') / args.path
        with open(args.path / 'training.json', 'r') as f:
            training_dict = json.load(f)
        try:
            with open(args.path / 'inference_sketchy.json') as f:
                inference_dict = json.load(f)
        except:
            with open(args.path / 'inference.json') as f:
                inference_dict = json.load(f)

    if args.method == 'visualize':
        visualize(args.path, training_dict, inference_dict)
    elif args.method == "topk_kaggle":
        topk_kaggle(inference_dict)
    else:
        eval(args.method)()