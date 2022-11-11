import os
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple, List

import random
import re

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# sketchy data prep

sketchy_path = Path("data/sketchy")

# dataset containing all sketch/photo pairs of sketchy
class SketchyDataset(Dataset):
    # no random transformation allowed because sketch/ image pairs
    # sketch type: png or svg, img type: photos or drawings?, mode: test or train, split_ratio: [0,1], random seed does only alter selection of negative photo
    def __init__(self, target_dir="data/sketchy", sketch_type='png', img_type="photos", transform=transforms.ToTensor(), 
                mode="train", split_ratio=0.2, size=1.0, seed=42) -> None:
        super().__init__()
        random.seed(seed)

        self.path = Path(target_dir)
        self.seed, self.mode, self.split_ratio, self.size = seed, mode, split_ratio, size
        self.sketch_type, self.img_type = sketch_type, img_type
        self.transform = transform

        self.classes, self.classes_to_idx = self.__sketchy_classes()

        self.sketch_paths = list()
        self.photo_paths = list()
        for cls in self.classes:
            self.sketch_paths += list( (self.path / ("sketches_" + self.sketch_type)).glob(f"{cls}/*.{self.sketch_type}") )
            self.photo_paths += list( (self.path / self.img_type).glob(f"{cls}/*.jpg") )

    # negative photo of same class? done in original paper but there was an additional classification loss other papers probably not)
    # (sketch, pos photo, neg photo)
    def load_image_sketch_tuple(self, idx: int) -> Tuple[Image.Image, Image.Image, Image.Image]:

        file_name = re.search("n\d+_\d+", self.sketch_paths[idx].name).group() + '.jpg'
        if not file_name: raise ValueError(f"regex failed! path name: {self.sketch_paths[idx]}")

        pos_photo_path = self.path / self.img_type / self.sketch_paths[idx].parent.name / file_name
        neg_photo_path = random.choice(self.photo_paths)
        return Image.open(self.sketch_paths[idx]), Image.open(pos_photo_path), Image.open(neg_photo_path)

    def load_image(self, idx:int) -> Image.Image:
        return self.photo_paths[idx]

    def load_sketch(self, idx:int) -> Image.Image:
        return self.sketch_paths[idx]

    def __len__(self) -> int:
        # sketch paths should be longer than photo paths because there ar multiple sketches per photo
        return len(self.sketch_paths)

    # (sketch, pos photo, neg photo) as [transformed] Tensor
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        sketch, pos_photo, neg_photo = self.load_image_sketch_tuple(idx)

        return self.transform(sketch), self.transform(pos_photo), self.transform(neg_photo)

    # retrieves classes and selects first n classes depending on size parameter
    def __sketchy_classes(self) -> Tuple[List[str], Dict[str, int], int]:
        
        classes = sorted(entry.name for entry in os.scandir(sketchy_path / self.img_type) if entry.is_dir() )
        if not classes:
            raise FileNotFoundError(f"No classes found in {sketchy_path / self.img_type}")

        num = round(self.size * len(classes))
        classes = classes[:num]

        classes_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, classes_idx


    @property
    def len_photos(self) -> int:
        return len(self.photo_paths)

    @property
    def len_sketches(self) -> int:
        return len(self.sketch_paths)

    @property
    def state_dict(self) -> Dict:
        return {"dataset": self.__class__.__name__, "size": self.size, "img_type": self.img_type, "sketch_type": self.sketch_type, 
                "seed": self.seed, "split_ratio": self.split_ratio, "mode": self.mode, "transform": self.transform}

set = SketchyDataset(size=0.8)
print(set.len_sketches)
print(set.state_dict)
print(set.__getitem__(5))