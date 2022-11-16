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

import visualization
import utils

# provides interface for loading duplicate free image paths with corresponding images (used in inference.compute_image_features)
class InferenceDataset(Dataset):

    def __init__(self, image_paths: List[Path], transform=transforms.ToTensor()):
        super().__init__()

        self.transform = transform
        self.image_paths = list( dict.fromkeys(image_paths) )
        self.image_paths.sort()

    def load_image(self, idx:int) -> Image.Image:
        return Image.open(self.image_paths[idx])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx:int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx])
        return self.transform(img)


# abstract base class for specific datasets
class RetrievalDataset(Dataset):

    # no random transformation allowed because sketch/ image pairs
    # sketch/img format: png/svg/jpg, img type: photos or drawings?, mode: test or train, split_ratio: [0,1], seed not for test train split
    def __init__(self, sketch_format='png', img_format='jpg', img_type="photos", transform=transforms.ToTensor(), 
                mode="train", split_ratio=0.2, size=0.1, seed=42) -> None:
        super().__init__()
        random.seed(seed)

        self.path = Path("") # has to be specified in subclasses

        self.seed, self.mode, self.split_ratio, self.size = seed, mode, split_ratio, size
        self.sketch_format, self.img_format, self.img_type = sketch_format, img_format, img_type
        self.transform = transform

        self.sketch_paths = list()
        self.photo_paths = list()

        # in subclasses _load_paths() and _sample() have to be called

    # negative photo of same class? done in original paper but there was an additional classification loss (other papers probably not)
    # (sketch, pos photo, neg photo)
    def load_image_sketch_tuple(self, idx: int) -> Tuple[Image.Image, Image.Image, Image.Image]:
        neg_photo_path = random.choice(self.photo_paths)
        return Image.open(self.sketch_paths[idx]), Image.open(self.photo_paths[idx]), Image.open(neg_photo_path)

    def load_image(self, idx:int) -> Image.Image:
        return Image.open(self.photo_paths[idx])

    def load_sketch(self, idx:int) -> Image.Image:
        return Image.open(self.sketch_paths[idx])

    def __len__(self) -> int:
        return len(self.sketch_paths)

    # (sketch, pos photo, neg photo) as [transformed] Tensor
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sketch, pos_photo, neg_photo = self.load_image_sketch_tuple(idx)
        return self.transform(sketch), self.transform(pos_photo), self.transform(neg_photo)

    # has to be implemented in subclasses
    def _load_paths(self) -> None:
        pass

    # samples sketch/image pairs depending on mode [train/test]
    def _sample(self) -> None:
        if self.mode == 'train':
            self.sketch_paths, _, self.photo_paths, _ = train_test_split(self.sketch_paths, self.photo_paths, 
                                                        test_size=self.split_ratio, random_state=42, shuffle=True)
        elif self.mode == 'test':
            _, self.sketch_paths, _, self.photo_paths = train_test_split(self.sketch_paths, self.photo_paths, 
                                                        test_size=self.split_ratio, random_state=42, shuffle=True)
        else:
            raise ValueError("invalid mode: [train, test]")

    def plot_samples(self, num:int, raw:bool=True) -> None:
        samples = []
        for i in range(num):
            pos = round(random.randrange(len(self)))
            if raw: samples.append(self.load_image_sketch_tuple(pos))
            else: samples.append(self.__getitem__(pos))
        visualization.show_triplets(samples)

    @property
    def state_dict(self) -> Dict:
        return {"dataset": f"{self.__class__.__name__}", "size": self.size, "img_number": len(self), "img_type": self.img_type, "img_format": self.img_format, "sketch_format": self.sketch_format, 
                "seed": self.seed, "split_ratio": self.split_ratio, "mode": self.mode, "transform": self.transform}

# sketchy data prep

# dataset containing all sketch/photo pairs of sketchy
class SketchyDatasetV1(RetrievalDataset):

    # no random transformation allowed because sketch/ image pairs
    # sketch/img format: png/svg/jpg, img type: photos or drawings?, mode: test or train, split_ratio: [0,1]
    def __init__(self, sketch_format='png', img_format='jpg', img_type="photos", transform=transforms.ToTensor(), 
                mode="train", split_ratio=0.2, size=1.0, seed=42) -> None:

        super().__init__(sketch_format, img_format, img_type, transform, mode, split_ratio, size, seed)

        self.path = Path("data/sketchy")

        self.classes, self.classes_to_idx = self.__sketchy_classes()

        self._load_paths()
        self._sample()

    # retrieves classes and selects first n classes depending on size parameter
    def __sketchy_classes(self) -> Tuple[List[str], Dict[str, int], int]:
        
        classes = sorted(entry.name for entry in os.scandir(self.path / self.img_type) if entry.is_dir() )
        if not classes:
            raise FileNotFoundError(f"No classes found in {self.path / self.img_type}")

        num = round(self.size * len(classes))
        classes = classes[:num]

        classes_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, classes_idx

    # photo paths will be duplicated so that there are a equal number of sketch/photo pairs
    def _load_paths(self) -> None:
        for cls in self.classes:
            self.sketch_paths += list( (self.path / ("sketches_" + self.sketch_format)).glob(f"{cls}/*.{self.sketch_format}") )
            """
            # works but then train_test_split is not possible because of different number of sketches and images
            self.photo_paths += list( (self.path / self.img_type).glob(f"{cls}/*.{self.img_format}") )
            """
        for path in self.sketch_paths:
            filename = re.search('n\d+_\d+', path.name).group() + '.' + self.img_format
            photo_path = self.path / self.img_type / path.parent.name / filename 
            self.photo_paths.append(Path(photo_path))


# returns train and test dataset
def get_datasets(dataset:str="Sketchy", size:float=1.0, sketch_format:str='png', img_format:str='jpg', img_type:str='photos', split_ratio:float=0.2, seed:int=42):
    train_dataset = None
    test_dataset = None
    if dataset == "Sketchy":
        train_dataset = SketchyDatasetV1(sketch_format, img_format, img_type, utils.ResNet50m_img_transform, 'train', split_ratio, size, seed)
        test_dataset = SketchyDatasetV1(sketch_format, img_format, img_type, utils.ResNet50m_img_transform, 'test', split_ratio, size, seed)

    return train_dataset, test_dataset