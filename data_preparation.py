import os
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple, List
from collections import defaultdict

import random
import re

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import visualization
import utils
import semiSupervised_utils


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
                mode="train", split_ratio=0.1, size=0.1, seed=42) -> None:
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
                "seed": self.seed, "split_ratio": self.split_ratio, "mode": self.mode, "transform": str(self.transform)}


# sketchy data prep

# dataset containing all sketch/photo pairs of sketchy
class SketchyDatasetV1(RetrievalDataset):

    # no random transformation allowed because sketch/ image pairs
    # sketch/img format: png/svg/jpg, img type: photos or drawings?, mode: test or train, split_ratio: [0,1]
    def __init__(self, sketch_format='png', img_format='jpg', img_type="photos", transform=transforms.ToTensor(), 
                mode="train", split_ratio=0.1, size=1.0, seed=42) -> None:

        super().__init__(sketch_format, img_format, img_type, transform, mode, split_ratio, size, seed)

        self.path = Path("data/sketchy")

        self.classes, self.classes_to_idx = self._sketchy_classes()

        self._load_paths()
        self._sample()

    # retrieves classes and selects first n classes depending on size parameter
    def _sketchy_classes(self) -> Tuple[List[str], Dict[str, int], int]:
        
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


class SketchyDatasetV2(SketchyDatasetV1):
    def __init__(self, sketch_format='png', img_format='jpg', img_type="photos", transform=transforms.ToTensor(), mode="train", split_ratio=0.1, size=0.1, seed=42) -> None:
        super().__init__(sketch_format, img_format, img_type, transform, mode, split_ratio, size, seed)

        self.categorized_images = self._provide_categorized_images()

    def _provide_categorized_images(self) -> None:
        self.categorized_images = defaultdict(list)

        for image_path in self.photo_paths:
            img_class = image_path.parent.stem
            self.categorized_images[img_class].append(image_path)
        return self.categorized_images

    def load_image_sketch_tuple(self, idx: int) -> Tuple[Image.Image, Image.Image, Image.Image, int]:
        img_class = self.photo_paths[idx].parent.stem
        label = self.classes_to_idx[img_class]
        neg_img_path = self.photo_paths[idx]
        n = 0 # in case a category has only 1 picture
        while neg_img_path == self.photo_paths[idx] or n < 10:
            neg_img_path = random.choice(self.categorized_images[img_class])
            n += 1
        return Image.open(self.sketch_paths[idx]), Image.open(self.photo_paths[idx]), Image.open(neg_img_path), label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        sketch, pos_img, neg_img, label = self.load_image_sketch_tuple(idx)
        return self.transform(sketch), self.transform(pos_img), self.transform(neg_img), label


class VectorizedSketchyDatasetV1(SketchyDatasetV1):
    
    def __init__(self, sketch_format='svg', img_format='jpg', img_type='photos', transform=transforms.ToTensor(), 
                mode='train', split_ratio=0.1, size=1.0, seed=42, include_erased:bool=True) -> None:

        super().__init__(sketch_format, img_format, img_type, transform, mode, split_ratio, size, seed)

        # inspired by Photo2SKetch_Dataset, semi-supervised fg-sbir
        # maybe max seq len has to be added

        self.max_seq_len = 0
        self.min_seq_len = 10e10
        self.avg_seq_len = 0

        self.reduce_factor = 2
        self.maximum_length = 100 # if 0 or reduce_factor = 1 itbwill not be applied
        self.include_erased = include_erased

        # if folder doesn't exist sketch tuples are loaded otherwise loaded and created
        self.vector_path = self.path / f'sketch_vectors_{mode}_{self.maximum_length}_{self.reduce_factor}'
        self.vectorized_sketches = []

        if not self.vector_path.is_dir():
            for path in self.sketch_paths:
                (self.vector_path / path.parent.name).mkdir(parents=True, exist_ok=True)
                sketch = semiSupervised_utils.parse_svg(path, self.vector_path / path.parent.name, reduce_factor=self.reduce_factor, max_length=self.maximum_length)
                self.vectorized_sketches.append(sketch)
                self.max_seq_len = max(self.max_seq_len, len(sketch['image']))
                self.min_seq_len = min(self.min_seq_len, len(sketch['image']))
                self.avg_seq_len += len(sketch['image'])
        else:
            for path in self.sketch_paths:
                vector_path = self.vector_path / path.parent.name / (path.stem + '.json')
                sketch = semiSupervised_utils.load_tuple_representation(vector_path)
                self.vectorized_sketches.append(sketch)
                self.max_seq_len = max(self.max_seq_len, len(sketch['image']))
                self.min_seq_len = min(self.min_seq_len, len(sketch['image']))
                self.avg_seq_len += len(sketch['image'])

        self.avg_seq_len /= len(self.vectorized_sketches)

        print(f"max_seq_len: {self.max_seq_len}, min_seq_len: {self.min_seq_len}, avg_seq_len: {self.avg_seq_len:.3f}")

        # scales coordinates by standard deviation
                
        data = []
        for vec_sketch in self.vectorized_sketches:
            data.extend(np.array(vec_sketch['image'])[:, 0])
            data.extend(np.array(vec_sketch['image'])[:, 1])
        data = np.array(data)
        scale_factor = np.std(data)

        for vec_sketch in self.vectorized_sketches:
            for line in vec_sketch['image']:
                line[:2] /= scale_factor

    def __getitem__(self, idx: int):
        # fill all sketches so they have same number of strokes
        sketch = self.vectorized_sketches[idx]['image']
        sketch_vector = np.zeros((self.max_seq_len, 5))
        sketch_vector[:len(sketch), :] = semiSupervised_utils.reshape_vectorSketch(self.vectorized_sketches[idx])['image']
        # !!! added 
        sketch_vector[len(sketch):, 4] = 1
        return { 'length': len(sketch), 'sketch_vector': torch.from_numpy(sketch_vector),
                'photo': self.transform(Image.open(self.photo_paths[idx]).convert('RGB')) }

    @property
    def state_dict(self) -> Dict:
        state_dict = super().state_dict
        state_dict['sequence_stats'] = {'max_seq_len': self.max_seq_len, 'min_seq_len': self.min_seq_len, 'avg_seq_len': self.avg_seq_len}

        state_dict['include_erased'] = self.include_erased
        state_dict['reduce_factor'] = self.reduce_factor
        state_dict['maximum_length'] = self.maximum_length
        return state_dict

class SketchyDatasetPix2Pix(SketchyDatasetV1):
    def __init__(self, sketch_format='png', img_format='jpg', img_type="photos", transform=transforms.ToTensor(), mode="train", split_ratio=0.1, size=1, seed=42) -> None:
        super().__init__(sketch_format, img_format, img_type, transform, mode, split_ratio, size, seed)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        image, sketch = Image.open(self.photo_paths[idx]), Image.open(self.sketch_paths[idx])
        if self.mode == 'train' and random.random() > 0.5:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
            sketch = sketch.transpose(method=Image.FLIP_LEFT_RIGHT)
        return {'image': self.transform(image), 'sketch': self.transform(sketch)}

    @property
    def state_dict(self) -> Dict:
        state_dict = super().state_dict
        state_dict['augmentation'] = 'train_random_hflip'
        return state_dict


# kaggle data prep

class KaggleDatasetImgOnlyV1(Dataset):
    def __init__(self, img_format='jpg', img_type="images", transform=transforms.ToTensor(), 
                mode="train", size=0.1, seed=42) -> None:
        super().__init__()

        self.img_format, self.img_type, self.transform, self.mode, self.size, self.seed = img_format, img_type, transform, mode, size, seed

        self.image_path = Path('../sketchit/public/paintings')#Path(f'data/kaggle/{self.img_type}/test')
        if mode == 'train': self.image_path = Path('/nfs/data/iart/kaggle/img')

        self.image_data = self._load_img_data() # sequential

        self.styles = self._get_classes('style')
        self.genres = self._get_classes('genre')

        #print(self.styles.loc['Abstract Expressionism']['index'])
        #print(self.styles.iloc[1].name)

    def _load_img_data(self) -> pd.DataFrame:
        self.csv_path = Path(f'data/kaggle/kaggle_art_dataset_{self.mode}.csv')
        data = pd.read_csv(self.csv_path)
        data['filename'] = self.image_path / data['filename']
        return data.head( int(data.shape[0] * self.size) )

    def _get_classes(self, category) -> pd.DataFrame:
        categories = pd.DataFrame(self.image_data[category].drop_duplicates(), columns=[category]).sort_values(by=category).reset_index(drop=True)
        categories['index'] = categories.index
        categories.set_index(category, inplace=True)
        return categories

    def __len__(self) -> int:
        return len(self.image_data)

    def load_image_tuple(self, idx:int) -> Tuple[Image.Image, Image.Image]:#, int, int]: # pos_image, neg_image, style, genre
        pos_img = self.image_data.iloc[idx]
        random_idx = random.randint(0, len(self.image_data) - 1)
        neg_img = self.image_data.iloc[random_idx]

        #style_label = self.styles.loc[pos_img['style']]['index']
        #genre_label = self.genres.loc[pos_img['genre']]['index']
        #if self.mode == 'test' and pos_img['genre'] > 'miniature': genre_label += 1 # genre miniature is missing in test dataset

        return Image.open(pos_img['filename']), Image.open(neg_img['filename'])#, style_label, genre_label

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:#, int, int]:
        pos_img, neg_img = self.load_image_tuple(idx)#, style, genre = self.load_image_tuple(idx)
        pos_img = self.transform(pos_img)
        neg_img = self.transform(neg_img)
        return pos_img, neg_img#, style, genre

    @property
    def state_dict(self) -> Dict:
        return {"dataset": f"{self.__class__.__name__}", "size": self.size, "img_number": len(self), "img_type": self.img_type, "img_format": self.img_format, 
                "seed": self.seed, "mode": self.mode, "transform": str(self.transform)}


# !!! only works properly with size=1 due to eventually missing genres in one of the datasets (train|test) !!!
class KaggleDatasetImgOnlyV2(KaggleDatasetImgOnlyV1):
    def __init__(self, img_format='jpg', img_type="images", transform=transforms.ToTensor(), 
                mode="train", size=0.1, seed=42) -> None:
        super().__init__(img_format, img_type, transform, mode, size, seed)

        self.categorized_images = self._get_categorized_images('genre')

    def _get_categorized_images(self, category) -> List:
        categorized = self.image_data.groupby(category)['filename'].apply(list)
        return categorized

    def load_image_tuple(self, idx:int) -> Tuple[Image.Image, Image.Image, int, int]: # pos_image, neg_image, style, genre
        pos_img = self.image_data.iloc[idx]
        neg_img = random.choice(self.categorized_images[pos_img['genre']])

        style_label = self.styles.loc[pos_img['style']]['index']
        genre_label = self.genres.loc[pos_img['genre']]['index']
        if self.mode == 'test' and pos_img['genre'] > 'miniature': genre_label += 1 # genre miniature is missing in test dataset

        return Image.open(pos_img['filename']), Image.open(neg_img), style_label, genre_label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]: # pos_image, neg_image, style, genre
        pos_img, neg_img, style, genre = self.load_image_tuple(idx)
        return self.transform(pos_img), self.transform(neg_img), style, genre

# not tested
class KaggleDatasetV2(KaggleDatasetImgOnlyV2):
    def __init__(self, sketch_format='png', img_format='jpg', sketch_type='placeholder', img_type="images", transform=transforms.ToTensor(), mode="train", size=0.1, seed=42) -> None:
        super().__init__(img_format, img_type, transform, mode, size, seed)

        self.sketch_format, self.sketch_type = sketch_format, sketch_type

        self.sketch_path = Path(f"data/kaggle/{self.sketch_type}/{self.mode}")

        self._load_sketch_paths() # adds sketchname entry to self.image_data with sketch path

    def _load_sketch_paths(self) -> None:
        for entry in self.image_data:
            entry['sketchname'] = self.sketch_path / f"{entry['filename'].stem}.{self.sketch_format}"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]: # sketch, pos_image, neg_image, style, genre
        pos_tensor, neg_tensor, style, genre = super().__getitem__(idx)
        sketch = Image.open(self.image_data[idx]['sketchname'])
        return self.transform(sketch), pos_tensor, neg_tensor, style, genre

# returns train and test dataset
def get_datasets(dataset:str="Sketchy", size:float=0.1, sketch_format:str='png', img_format:str='jpg', sketch_type:str='placeholder', img_type:str='photos', split_ratio:float=0.1, seed:int=42, transform=transforms.ToTensor()):

    if dataset == "Sketchy":
        train_dataset = SketchyDatasetV1(sketch_format, img_format, img_type, transform, 'train', split_ratio, size, seed)
        test_dataset = SketchyDatasetV1(sketch_format, img_format, img_type, transform, 'test', split_ratio, size, seed)
    elif dataset == 'SketchyV2':
        train_dataset = SketchyDatasetV2(sketch_format, img_format, img_type, transform, 'train', split_ratio, size, seed)
        test_dataset = SketchyDatasetV2(sketch_format, img_format, img_type, transform, 'test', split_ratio, size, seed)
    elif dataset == 'VectorizedSketchyV1':
        train_dataset = VectorizedSketchyDatasetV1('svg', img_format, img_type, transform, 'train', split_ratio, size, seed, include_erased=True)
        test_dataset = VectorizedSketchyDatasetV1('svg', img_format, img_type, transform, 'test', split_ratio, size, seed, include_erased=True)
    
    elif dataset == 'KaggleDatasetImgOnlyV1':
        train_dataset = KaggleDatasetImgOnlyV1(img_format, img_type, transform, 'train', size, seed)
        test_dataset = KaggleDatasetImgOnlyV1(img_format, img_type, transform, 'train', size, seed)
    elif dataset == 'KaggleDatasetImgOnlyV2':
        train_dataset = KaggleDatasetImgOnlyV2(img_format, img_type, transform, 'train', size, seed)
        test_dataset = KaggleDatasetImgOnlyV2(img_format, img_type, transform, 'train', size, seed)
    elif dataset == 'KaggleDatasetV2':
        train_dataset = KaggleDatasetV2(sketch_format, img_format, sketch_type, img_type, transform, 'train', size, seed)
        test_dataset = KaggleDatasetV2(sketch_format, img_format, sketch_type, img_type, transform, 'test', size, seed)

    return train_dataset, test_dataset


if __name__ == '__main__':
    #dataset = VectorizedSketchyDatasetV1(size=0.01, transform=utils.get_sketch_gen_transform())
    dataset = KaggleDatasetImgOnlyV1(size=1, mode='test')
    print(dataset.__getitem__(1))
    #print(len(dataset.categorized_images.index))
    dataset2 = KaggleDatasetImgOnlyV2(size=1, mode='train')
    print( list(dataset2.categorized_images.index).index('miniature'))
