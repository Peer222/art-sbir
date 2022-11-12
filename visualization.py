import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from PIL import Image

from pathlib import Path

from typing import List, Tuple, Dict

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
    print(rows)
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