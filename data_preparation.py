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
import transformations


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
    def _sample(self, lists) -> None:
        splitted_lists = train_test_split(*lists, test_size=self.split_ratio, random_state=42, shuffle=True)

        if self.mode == 'train':
            if len(lists) == 2: self.sketch_paths, _, self.photo_paths, _ = splitted_lists
            elif len(lists) == 3: self.sketch_paths, _, self.photo_paths, _, self.vectorized_sketches, _ = splitted_lists
        elif self.mode == 'test':
            if len(lists) == 2: _, self.sketch_paths, _, self.photo_paths = splitted_lists
            elif len(lists) == 3: _, self.sketch_paths, _, self.photo_paths, _, self.vectorized_sketches = splitted_lists
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
                mode="train", split_ratio=0.1, size=1.0, seed=42, max_erase_count=99999, only_valid=True, _sample=True) -> None:

        super().__init__(sketch_format, img_format, img_type, transform, mode, split_ratio, size, seed)
        self.only_valid = only_valid
        self.max_erase_count = max_erase_count

        self.path = Path("data/sketchy")

        self.classes, self.classes_to_idx = self._sketchy_classes()

        self._load_paths()
        if _sample: self._sample([self.sketch_paths, self.photo_paths])

        print(len(self.sketch_paths), len(self.photo_paths))

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
        """ # contains fewer sketches (older experiments ran on latter version)
        data = pd.read_csv(self.path / 'info' / 'stats.csv')
        if self.sketch_format == 'svg' and not 'svg_available' in data.columns: data = self.__mark_missing_svgs(data) # not all sketches are available as svgs
        for i, row in data.iterrows():
            if row['CategoryID'] >= len(self.classes): break
            if self.sketch_format == 'svg' and row['svg_available'] == 0: continue

            if row['Eraser_Count'] <= self.max_erase_count and (row['Error?']+row['Context?']+row['Ambiguous?']+row['WrongPose?'] == 0 or not self.only_valid):
                category = row['Category'].replace(' ', '_')
                # if sketch itself is used as image filename is different
                img_name = f"{row['ImageNetID']}.{self.img_format}" if not 'sketch' in self.img_type else f"{row['ImageNetID']}-{row['SketchID']}.{self.img_format}"

                self.photo_paths.append(self.path / self.img_type / category / img_name)
                self.sketch_paths.append(self.path / f"sketches_{self.sketch_format}" / category / f"{row['ImageNetID']}-{row['SketchID']}.{self.sketch_format}")
        """
        for cls in self.classes:
            self.sketch_paths += list( (self.path / ("sketches_" + self.sketch_format)).glob(f"{cls}/*.{self.sketch_format}") )

        for path in self.sketch_paths:
            if self.img_type == "artworks":
                filename = path.stem + '.' + self.img_format
            else:
                filename = re.search('n\d+_\d+', path.name).group() + '.' + self.img_format
            photo_path = self.path / self.img_type / path.parent.name / filename 
            self.photo_paths.append(Path(photo_path))

    # adds svg_available column and updates info/stats.csv
    def __mark_missing_svgs(self, data):
        data['svg_available'] = 0
         
        for i, row in data.iterrows():
            category = row['Category'].replace(' ', '_')
            svg_path = self.path / f"sketches_{self.sketch_format}" / category / f"{row['ImageNetID']}-{row['SketchID']}.{self.sketch_format}"
            if svg_path.exists(): data.loc[i, 'svg_available'] = 1

        data.to_csv(self.path / 'info' / 'stats.csv')
        return data

    @property
    def state_dict(self) -> Dict:
        state_dict = super().state_dict
        state_dict['valid_only'] = self.only_valid
        state_dict['max_erase_count'] = self.max_erase_count
        return state_dict


class SketchyDatasetV2(SketchyDatasetV1):
    def __init__(self, sketch_format='png', img_format='jpg', img_type="photos", transform=transforms.ToTensor(), mode="train", split_ratio=0.1, size=0.1, seed=42, max_erase_count=99999, only_valid=True) -> None:
        super().__init__(sketch_format, img_format, img_type, transform, mode, split_ratio, size, seed, max_erase_count, only_valid)

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
                mode='train', split_ratio=0.1, size=1.0, seed=42, max_erase_count=99999, only_valid=True) -> None:

        super().__init__(sketch_format, img_format, img_type, transform, mode, split_ratio, size, seed, max_erase_count, only_valid, _sample=False)

        # inspired by Photo2SKetch_Dataset, semi-supervised fg-sbir
        # maybe max seq len has to be added

        self.max_seq_len = 0
        self.min_seq_len = 10e10
        self.avg_seq_len = 0

        self.reduce_factor = 2
        self.maximum_length = 100 # if 0 or reduce_factor = 1 itbwill not be applied

        # if folder doesn't exist sketch tuples are loaded otherwise loaded and created
        self.vector_path = self.path / f'sketch_vectors_{self.maximum_length}_{self.reduce_factor}_V2' #_{int(self.size*100)}_V2
        self.vectorized_sketches = []

        if not self.vector_path.is_dir():
            if self.only_valid: print('Warning: Only svgs of valid sketches will be created - eventually NotFound errors with different settings', flush=True)
            for path in self.sketch_paths:
                (self.vector_path / path.parent.name).mkdir(parents=True, exist_ok=True)
                sketch = semiSupervised_utils.parse_svg(path, self.vector_path / path.parent.name, reduce_factor=self.reduce_factor, max_length=self.maximum_length)
                self.vectorized_sketches.append(sketch)
        else:
            for path in self.sketch_paths:
                vector_path = self.vector_path / path.parent.name / (path.stem + '.json')
                sketch = semiSupervised_utils.load_tuple_representation(vector_path)
                self.vectorized_sketches.append(sketch)

        self._sample([self.sketch_paths, self.photo_paths, self.vectorized_sketches])


        seq_lengths = [len(sketch['image']) for sketch in self.vectorized_sketches]
        self.avg_seq_len = np.round(np.mean(seq_lengths) + np.std(seq_lengths)) # np.mean(seq_lengths) -> from original paper std * 0.5 in shoeV2 dataset
        self.max_seq_len = np.max(seq_lengths)
        self.min_seq_len = np.min(seq_lengths)

        print(f"max_seq_len: {self.max_seq_len}, min_seq_len: {self.min_seq_len}, avg_seq_len: {self.avg_seq_len:.3f}")

        # scales coordinates by standard deviation
                
        self.vectorized_sketches = self.purify(self.vectorized_sketches)
        self.vectorized_sketches = self.normalize(self.vectorized_sketches)

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def __getitem__(self, idx: int) -> Dict:
        # fill all sketches so they have same number of strokes
        sketch = self.vectorized_sketches[idx]['image']
        sketch_vector = np.zeros((self.maximum_length, 5))
        sketch_vector[:len(sketch), :] = semiSupervised_utils.reshape_vectorSketch(self.vectorized_sketches[idx])['image']
        # !!! added 
        sketch_vector[len(sketch):, 4] = 1
        sketch_vector = sketch_vector[1:]
        sketch_vector = np.concatenate([sketch_vector, [[0, 0, 0, 0, 1]]])

        if not self.img_format == 'svg': image = transforms.ToTensor()( Image.open(self.photo_paths[idx]).convert('RGB') ).unsqueeze(0)
        else: image = 1. - semiSupervised_utils.batch_rasterize_relative(torch.from_numpy(sketch_vector).to(torch.float32).unsqueeze(0)) / 255.
        image = image.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None]).squeeze() # instead of self.transform()
        return { 'length': len(sketch), 'sketch_vector': torch.from_numpy(sketch_vector).to(torch.float32),
                'photo': image }

    @property
    def state_dict(self) -> Dict:
        state_dict = super().state_dict
        state_dict['sequence_stats'] = {'max_seq_len': int(self.max_seq_len), 'min_seq_len': int(self.min_seq_len), 'avg_seq_len': int(self.avg_seq_len)}

        state_dict['reduce_factor'] = self.reduce_factor
        state_dict['maximum_length'] = self.maximum_length
        state_dict['V2'] = 'V2' in str(self.vector_path)
        return state_dict

    def purify(self, vectorized_sketches):
        """removes to small or too long sequences + removes large gaps"""
        for i in range(len(vectorized_sketches) - 1, -1, -1):
            seq = np.array(vectorized_sketches[i]['image'], dtype=np.float32)
            if seq.shape[0] <= self.max_seq_len and seq.shape[0] > 10:
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                #seq = np.array(seq, dtype=np.float32)
                vectorized_sketches[i]['image'] = seq
            else:
                self.sketch_paths.pop(i)
                self.photo_paths.pop(i)
        return vectorized_sketches

    def calculate_normalizing_scale_factor(self, vectorized_sketches):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(vectorized_sketches)):
            strokes = vectorized_sketches[i]['image']
            for j in range(len(strokes)):
                data.append(strokes[j, 0])
                data.append(strokes[j, 1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, vectorized_sketches):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        scale_factor = self.calculate_normalizing_scale_factor(vectorized_sketches)
        for i in range(len(vectorized_sketches)):
            vectorized_sketches[i]['image'][:, 0:2] /= scale_factor
        return vectorized_sketches


class SketchyDatasetPix2Pix(SketchyDatasetV1):
    def __init__(self, sketch_format='png', img_format='jpg', img_type="photos", transform=None, mode="train", split_ratio=0.1, size=1, seed=42, max_erase_count=99999, only_valid=True) -> None:
        super().__init__(sketch_format, img_format, img_type, transform, mode, split_ratio, size, seed, max_erase_count, only_valid)

        self.grayscale_sketch = True
        self.transform_img = self.transform_pix2pix(to_grayscale=False)
        self.transform_sketch = self.transform_pix2pix(to_grayscale=self.grayscale_sketch)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        image, sketch = Image.open(self.photo_paths[idx]), Image.open(self.sketch_paths[idx])
        #sketch = sketch.convert('L')
        if self.mode == 'train' and random.random() > 0.5:
            image = image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            sketch = sketch.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        return {'A': self.transform_img(image), 'B': self.transform_sketch(sketch), 'img_paths': str(self.photo_paths[idx])}

    def transform_pix2pix(self, to_grayscale:bool):
        # from drawings
        transforms_img = [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()] # resizes smallest edge + keeps ratio 
        transforms_img += [transforms.Grayscale(1)] if to_grayscale else []
        return transforms.Compose(transforms_img)
        # from pix2pix
        #transformations = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # normalize to a range of [-1, 1] (tanh is used as final activation)
        #transformations += [transforms.Grayscale(1)] if to_grayscale else []
        #return transforms.Compose(transformations)

    @property
    def state_dict(self) -> Dict:
        state_dict = super().state_dict
        state_dict['augmentation'] = 'train_random_hflip'
        state_dict['sketch_type'] = 'grayscale' if self.grayscale_sketch else 'rgb'
        state_dict['transform'] = {'sketch': str(self.transform_sketch), 'image': str(self.transform_img)}
        return state_dict


# version untested
class QuickDrawDatasetV1(RetrievalDataset):
    def __init__(self, sketch_format='npz', img_format='npz', img_type="sketch", transform=0, mode="train", split_ratio=0, size=0.1, seed=0, max_length=100) -> None: # 0 values are unused
        super().__init__(sketch_format, img_format, img_type, transform, mode, split_ratio, size, seed)

        self.path = Path('data/quick_draw')
        self.maximum_length = max_length 

        self.categories = ['baseball bat', 'banana', 'apple', 'ant', 'alarm clock', 'airplane']

        self.sketches = self._load_sketches()

        seq_lengths = [len(seq) for seq in self.sketches]
        self.avg_seq_len = int(np.round(np.mean(seq_lengths) + np.std(seq_lengths))) # np.mean(seq_lengths) -> from original paper
        self.max_seq_len = int(np.max(seq_lengths))
        self.min_seq_len = int(np.min(seq_lengths))

        self.sketches = self.purify(self.sketches)
        self.sketches = self.normalize(self.sketches)

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])


    def _load_sketches(self) -> np.ndarray:
        sketches = np.array([])
        mode = 'train' if self.mode == 'train' else 'valid'
        for category in self.categories:
            dataset = np.load(self.path / f"{category}.npz", encoding='latin1', allow_pickle=True)
            sketches = np.concatenate([sketches, dataset[mode]])

        max_items = int(self.size * sketches.shape[0])
        return sketches[:max_items]

    def __len__(self) -> int:
        return len(self.sketches)

    def __getitem__(self, idx: int) -> Dict:
        sketch = self.sketches[idx]
        len_seq = len(sketch[:, 0])
        new_seq = np.zeros((self.maximum_length, 5)) # original max_seq_len
        new_seq[0:len_seq, :2] = sketch[:, :2]
        new_seq[0:len_seq, 3] = sketch[:, 2]
        new_seq[0:len_seq, 2] = 1 - new_seq[0:len_seq, 3]
        new_seq[(len_seq - 1):, 4] = 1
        new_seq[(len_seq - 1):, 2:4] = 0

        sketch = torch.from_numpy(new_seq)
        image = 1. - semiSupervised_utils.batch_rasterize_relative(sketch.unsqueeze(0)) / 255.
        image = image.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None]).squeeze()

        return {'length': len(self.sketches[idx]), 'sketch_vector': sketch, 'photo': image }

    @property
    def state_dict(self) -> Dict:
        state_dict = super().state_dict
        state_dict['sequence_stats'] = {'max_seq_len': self.max_seq_len, 'min_seq_len': self.min_seq_len, 'avg_seq_len': self.avg_seq_len}
        state_dict['maximum_length'] = self.maximum_length
        return state_dict

    def purify(self, strokes):
        """removes to small or too long sequences + removes large gaps"""
        data = []
        for seq in strokes:
            if seq.shape[0] <= self.max_seq_len and seq.shape[0] > 10:
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)
        return data

    def calculate_normalizing_scale_factor(self, strokes):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(strokes)):
            for j in range(len(strokes[i])):
                data.append(strokes[i][j, 0])
                data.append(strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, strokes):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(strokes)
        for seq in strokes:
            seq[:, 0:2] /= scale_factor
            data.append(seq)
        return data




# kaggle data prep

class KaggleDatasetImgOnlyV1(Dataset):
    def __init__(self, img_format='jpg', img_type="images", transform=transforms.ToTensor(), 
                mode="train", size=0.1, seed=42) -> None:
        super().__init__()

        self.img_format, self.img_type, self.transform, self.mode, self.size, self.seed = img_format, img_type, transform, mode, size, seed
        # server has cuda / pc not
        if torch.cuda.is_available():
            self.image_path = Path(f'data/kaggle/{self.img_type}')
        else:
            self.image_path = Path('../sketchit/public/paintings')
        if mode == 'train': self.image_path = Path('/nfs/data/iart/kaggle/img')

        self.image_data = self._load_img_data() # sequential

        self.photo_paths = list(self.image_data['filename']) # needed for compute_image_features in inference.py

        self.styles = self._get_classes('style')
        self.genres = self._get_classes('genre')

        Image.MAX_IMAGE_PIXELS = 283327980 # 98873.jpg
        # truncated test 5322->3917.jpg (fixed)
        # truncated/to large train 1313->79499.jpg, 8739->91033.jpg, 14997->82594.jpg, 24354->101947.jpg, 25280->33557.jpg, 32647->41945.jpg, 34485->50420.jpg, 35464->72255.jpg, 42310->81823.jpg, 51640->95347.jpg, 51683->95010.jpg, 68020->92899.jpg (removed, reason: output/slurm-2738.out)

        #print(self.styles.loc['Abstract Expressionism']['index'])
        #print(self.styles.iloc[1].name)
        print(len(self.image_data))

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

    def load_image_tuple(self, idx:int): # unused
        pass

    def __getitem__(self, idx:int) -> Dict:
        img_data = self.image_data.iloc[idx]
        try: 
            image = Image.open(img_data['filename']).convert('RGB')
            name = img_data['filename'].stem
        except Exception as e: 
            print(f"error at {idx} - Image name: {img_data['filename']}")
            print(e)
            image = Image.open(self.image_data.iloc[0]['filename']).convert('RGB')
            name = 'dummy'

        return {'image': self.transform(image), 'name': name, 'path': str(img_data['filename'])}

    @property
    def state_dict(self) -> Dict:
        return {"dataset": f"{self.__class__.__name__}", "size": self.size, "img_number": len(self), "img_type": self.img_type, "img_format": self.img_format, 
                "seed": self.seed, "mode": self.mode, "transform": str(self.transform), "num_styles": len(self.styles), "num_genres": len(self.genres)}


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

        return Image.open(pos_img['filename']).convert('RGB'), Image.open(neg_img).convert('RGB'), style_label, genre_label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]: # pos_image, neg_image, style, genre
        pos_img, neg_img, style, genre = self.load_image_tuple(idx)
        return self.transform(pos_img), self.transform(neg_img), style, genre


class KaggleDatasetV1(KaggleDatasetImgOnlyV1):
    def __init__(self, sketch_format='png', img_format='jpg', sketch_type='contour_drawings', img_type="images", transform=transforms.ToTensor(), mode="train", size=0.1, seed=42) -> None:
        super().__init__(img_format, img_type, transform, mode, size, seed)

        self.sketch_format, self.sketch_type = sketch_format, sketch_type

        self.sketch_path = Path(f"data/kaggle/{self.sketch_type}") if not isinstance(self.sketch_type, list) else Path(f"data/kaggle/{self.sketch_type[0]}")

        self._load_sketch_paths() # adds sketchname entry to self.image_data with sketch path

        self.sketch_paths = list(self.image_data['sketchname'])

    def _load_sketch_paths(self) -> None:
        for i in range(len(self.image_data)):
            self.image_data.loc[i, 'sketchname'] = self.sketch_path / f"{self.image_data.loc[i, 'filename'].stem}.{self.sketch_format}"

    def load_image_tuple(self, idx:int) -> Tuple[Image.Image, Image.Image, Image.Image]: # sketch, pos_image, neg_image
        pos_img = self.image_data.iloc[idx]
        neg_img = random.choice(self.image_data['filename'])

        sketch = pos_img['sketchname']
        if isinstance(self.sketch_type, list):
            sketch = self.sketch_path.parent / random.choice(self.sketch_type) / sketch.name

        return Image.open(sketch).convert('RGB'), Image.open(pos_img['filename']).convert('RGB'), Image.open(neg_img).convert('RGB')

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # sketch pos_image, neg_image
        sketch, pos_img, neg_img = self.load_image_tuple(idx)
        return self.transform(sketch), self.transform(pos_img), self.transform(neg_img)

    @property
    def state_dict(self) -> Dict:
        state_dict = super().state_dict
        state_dict['sketch_type'] = self.sketch_type
        state_dict['sketch_format']= self.sketch_format
        return state_dict
    


class KaggleDatasetV2(KaggleDatasetImgOnlyV2):
    def __init__(self, sketch_format='png', img_format='jpg', sketch_type='contour_drawings', img_type="images", transform=transforms.ToTensor(), mode="train", size=0.1, seed=42) -> None:
        super().__init__(img_format, img_type, transform, mode, size, seed)

        self.sketch_format, self.sketch_type = sketch_format, sketch_type

        self.sketch_path = Path(f"data/kaggle/{self.sketch_type}") if not isinstance(self.sketch_type, list) else Path(f"data/kaggle/{self.sketch_type[0]}")

        self._load_sketch_paths() # adds sketchname entry to self.image_data with sketch path

        self.sketch_paths = list(self.image_data['sketchname'])

    def _load_sketch_paths(self) -> None:
        for i in range(len(self.image_data)):
            self.image_data.loc[i, 'sketchname'] = self.sketch_path / f"{self.image_data.loc[i, 'filename'].stem}.{self.sketch_format}"

    def load_image_tuple(self, idx: int) -> Tuple[Image.Image, Image.Image, Image.Image, int, int]:
        pos_img, neg_img, style_label, genre_label = super().load_image_tuple(idx)

        sketch = self.image_data.loc[idx, 'sketchname']
        if isinstance(self.sketch_type, list):
            sketch = self.sketch_path.parent / random.choice(self.sketch_type) / sketch.name

        return [Image.open(sketch).convert('RGB'), pos_img, neg_img, style_label, genre_label]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]: # sketch, pos_image, neg_image, style, genre
        sketch, pos_img, neg_img, style, genre = self.load_image_tuple(idx)
        return self.transform(sketch), self.transform(pos_img), self.transform(neg_img), style, genre

    @property
    def state_dict(self) -> Dict:
        state_dict = super().state_dict
        state_dict['sketch_type'] = self.sketch_type
        state_dict['sketch_format']= self.sketch_format
        return state_dict


class AugmentedKaggleDatasetV1(KaggleDatasetV1):
    def __init__(self, sketch_format='png', img_format='jpg', sketch_type='contour_drawings', img_type="images", transform=transforms.ToTensor(), mode="train", size=0.1, seed=42) -> None:
        super().__init__(sketch_format, img_format, sketch_type, img_type, transform, mode, size, seed)

        self.transform, _ = transformations.get_transformation()
        self.sketch_transform, self.t_name = transformations.get_transformation("sketch_transform", "V1")

    def load_image_tuple(self, idx: int) -> Tuple[Image.Image, Image.Image, Image.Image, int, int]:
        item = list(super().load_image_tuple(idx))
        if self.mode == 'train' and random.random() > 0.5:
            item[0] = item[0].transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            item[1] = item[1].transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            item[2] = transforms.RandomHorizontalFlip(p=0.5)(item[2])
        return item

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # sketch, pos_image, neg_image
        sketch, pos_img, neg_img = self.load_image_tuple(idx)
        pos_img, neg_img = self.transform(pos_img), self.transform(neg_img)
        if self.mode == 'train': sketch = self.sketch_transform(sketch)
        else: sketch = self.transform(sketch)
        return sketch, pos_img, neg_img

    @property
    def state_dict(self) -> Dict:
        state_dict = super().state_dict
        state_dict['sketch_transform_name'] = self.t_name
        state_dict['sketch_transform'] = str(self.sketch_transform) + " + paired random horizontal flip"
        return state_dict

class AugmentedKaggleDatasetV2(KaggleDatasetV2):
    def __init__(self, sketch_format='png', img_format='jpg', sketch_type='contour_drawings', img_type="images", transform=transforms.ToTensor(), mode="train", size=0.1, seed=42) -> None:
        super().__init__(sketch_format, img_format, sketch_type, img_type, transform, mode, size, seed)

        self.transform, _ = transformations.get_transformation()
        self.sketch_transform, self.t_name = transformations.get_transformation("sketch_transform", "V1")

    def load_image_tuple(self, idx: int) -> Tuple[Image.Image, Image.Image, Image.Image, int, int]:
        item = list(super().load_image_tuple(idx))
        if self.mode == 'train' and random.random() > 0.5:
            item[0] = item[0].transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            item[1] = item[1].transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            item[2] = transforms.RandomHorizontalFlip(p=0.5)(item[2])
        return item

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]: # sketch, pos_image, neg_image, style, genre
        sketch, pos_img, neg_img, style, genre = self.load_image_tuple(idx)
        pos_img, neg_img = self.transform(pos_img), self.transform(neg_img)
        if self.mode == 'train': sketch = self.sketch_transform(sketch)
        else: sketch = self.transform(sketch)
        return sketch, pos_img, neg_img, style, genre

    @property
    def state_dict(self) -> Dict:
        state_dict = super().state_dict
        state_dict['sketch_transform_name'] = self.t_name
        state_dict['sketch_transform'] = str(self.sketch_transform) + " + paired random horizontal flip"
        return state_dict


class KaggleInferenceDatasetV1(Dataset):
    # provides sketches from sketchit for inference
    def __init__(self, sketch_type='sketches', sketch_format='png', transform=transforms.ToTensor()) -> None:
        super().__init__()

        self.path = Path('data/kaggle')

        self.sketch_type, self.sketch_format, self.transform = sketch_type, sketch_format, transform

        self.sketch_paths = self.__load_sketch_paths()

    def __load_sketch_paths(self):
        data = pd.read_csv(self.path / "categorized_sketches.csv")
        data = data[data['valid'] == 1]
        data['sketch_paths'] = self.path / self.sketch_type / data['sketch']
        return list(data['sketch_paths'])

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, idx):
        return [ self.transform(Image.open(self.sketch_paths[idx])) ]

    @property
    def state_dict(self):
        return {"dataset": f"{self.__class__.__name__}", "img_number": len(self), "sketch_type": self.sketch_type, "sketch_format": self.sketch_format,
                "transform": str(self.transform), "date": "31.12.2022"}

class MixedDataset(Dataset):
    def __init__(self, mode='train', sketch_type="contour_drawings", sketchy_img_type="photos", size=1.0, transform=transformations.get_transformation()[0], version='V1', sketch_format='png'):
        super().__init__()
        self.mode, self.size, self.transform, self.version = mode, size, transform, version
        self.sketch_type = sketch_type
        self.sketchy_img_type = sketchy_img_type

        self.kaggle = eval(f"AugmentedKaggleDataset{self.version}")(mode=self.mode, size=self.size, sketch_type=self.sketch_type, sketch_format=sketch_format)
        self.sketchy = eval(f"SketchyDataset{self.version}")(mode=self.mode, size=self.size, img_type=sketchy_img_type, transform=self.transform)

        # only needed for inference
        self.photo_paths = self.kaggle.photo_paths
        self.sketch_paths = self.kaggle.sketch_paths

    def __len__(self) -> int:
        return 2 * max(len(self.sketchy), len(self.kaggle)) if self.mode == 'train' else len(self.sketch_paths)

    def __getitem__(self, idx:int):
        if self.mode == 'test': return self.kaggle.__getitem__(idx)[:3]
        if idx % 2 == 0:
            return self.kaggle.__getitem__( (idx // 2) % len(self.kaggle) )[:3]
        else:
            return self.sketchy.__getitem__( ((idx - 1) // 2) % len(self.sketchy) )[:3]

    @property
    def state_dict(self):
        return {"dataset": f"{self.__class__.__name__}", "version": self.version, "img_number": len(self), "size": self.size, "mode": self.mode, "sketch_type": self.sketch_type, "sketchy_img_type": self.sketchy_img_type, "transform":str(self.transform), "kaggle": self.kaggle.state_dict, "sketchy": self.sketchy.state_dict}

# returns train and test dataset
def get_datasets(dataset:str="Sketchy", size:float=0.1, sketch_format:str='png', img_format:str='jpg', sketch_type:str='placeholder', img_type:str='photos', split_ratio:float=0.1, seed:int=42, transform=transforms.ToTensor(), max_erase_count=99999, only_valid=True):

    if dataset in ['SketchyV1', "Sketchy", "SketchyDatasetV1"]:
        train_dataset = SketchyDatasetV1(sketch_format, img_format, img_type, transform, 'train', split_ratio, size, seed, max_erase_count, only_valid)
        test_dataset = SketchyDatasetV1(sketch_format, img_format, img_type, transform, 'test', split_ratio, size, seed, max_erase_count, only_valid)
    elif dataset in ['SketchyV2', "SketchyDatasetV2"]:
        train_dataset = SketchyDatasetV2(sketch_format, img_format, img_type, transform, 'train', split_ratio, size, seed, max_erase_count, only_valid)
        test_dataset = SketchyDatasetV2(sketch_format, img_format, img_type, transform, 'test', split_ratio, size, seed, max_erase_count, only_valid)
    elif dataset in ['VectorizedSketchyV1', "VectorizedSketchyDatasetV1"]:
        train_dataset = VectorizedSketchyDatasetV1('svg', img_format, img_type, transform, 'train', split_ratio, size, seed, max_erase_count, only_valid)
        test_dataset = VectorizedSketchyDatasetV1('svg', img_format, img_type, transform, 'test', split_ratio, size, seed, max_erase_count, only_valid)
    elif dataset in ['SketchyPix2Pix', "SketchyDatasetPix2Pix"]:
        train_dataset = SketchyDatasetPix2Pix(sketch_format, img_format, img_type, transform, 'train', split_ratio, size, seed)
        test_dataset = SketchyDatasetPix2Pix(sketch_format, img_format, img_type, transform, 'test', split_ratio, size, seed)
    
    elif dataset == 'KaggleDatasetImgOnlyV1':
        train_dataset = KaggleDatasetImgOnlyV1(img_format, img_type, transform, 'train', size, seed)
        test_dataset = KaggleDatasetImgOnlyV1(img_format, img_type, transform, 'test', size, seed)
    elif dataset == 'KaggleDatasetImgOnlyV2':
        train_dataset = KaggleDatasetImgOnlyV2(img_format, img_type, transform, 'train', size, seed)
        test_dataset = KaggleDatasetImgOnlyV2(img_format, img_type, transform, 'test', size, seed)
    elif dataset in ['KaggleV1', 'Kaggle', 'KaggleDatasetV1']:
        train_dataset = KaggleDatasetV1(sketch_format, img_format, sketch_type, img_type, transform, 'train', size, seed)
        test_dataset = KaggleDatasetV1(sketch_format, img_format, sketch_type, img_type, transform, 'test', size, seed)
    elif dataset in ['KaggleV2', 'KaggleDatasetV2']:
        train_dataset = KaggleDatasetV2(sketch_format, img_format, sketch_type, img_type, transform, 'train', size, seed)
        test_dataset = KaggleDatasetV2(sketch_format, img_format, sketch_type, img_type, transform, 'test', size, seed)
    elif dataset in ['AugmentedKaggleV1', 'AugmentedKaggleDatasetV1']:
        train_dataset = AugmentedKaggleDatasetV1(sketch_format, img_format, sketch_type, img_type, transform, 'train', size, seed)
        test_dataset = AugmentedKaggleDatasetV1(sketch_format, img_format, sketch_type, img_type, transform, 'test', size, seed)
    elif dataset in ['AugmentedKaggleV2', 'AugmentedKaggleDatasetV2']:
        train_dataset = AugmentedKaggleDatasetV2(sketch_format, img_format, sketch_type, img_type, transform, 'train', size, seed)
        test_dataset = AugmentedKaggleDatasetV2(sketch_format, img_format, sketch_type, img_type, transform, 'test', size, seed)
    elif dataset in ['KaggleInferenceV1', 'KaggleInferencedatasetV1']:
        train_dataset = None
        test_dataset = KaggleInferenceDatasetV1(sketch_type, sketch_format, transform)
        
    elif dataset == 'MixedDatasetV1':
        train_dataset = MixedDataset(mode='train', size=size, sketch_type=sketch_type, sketchy_img_type=img_type, version='V1', sketch_format=sketch_format)
        test_dataset = MixedDataset(mode='test',size=size, sketch_type=sketch_type, sketchy_img_type=img_type, version='V1', sketch_format=sketch_format)
    elif dataset == 'MixedDatasetV2':
        train_dataset = MixedDataset(mode='train',size=size, sketch_type=sketch_type, sketchy_img_type=img_type, version='V2', sketch_format=sketch_format)
        test_dataset = MixedDataset(mode='test',size=size, sketch_type=sketch_type, sketchy_img_type=img_type, version='V2', sketch_format=sketch_format)
    elif dataset == 'QuickdrawV1':
        train_dataset = QuickDrawDatasetV1(mode='train', size=size)
        test_dataset = QuickDrawDatasetV1(mode='test', size=size)
    else:
        raise Exception(f"{dataset} is not available")

    return train_dataset, test_dataset


if __name__ == '__main__':
    #print('Start generating vectors')
    #dataset = VectorizedSketchyDatasetV1(size=0.01, img_type='sketches_svg', img_format='svg', transform=utils.get_sketch_gen_transform(), only_valid=False) # locally 0.01 size
    #dataset2 = VectorizedSketchyDatasetV1(size=0.01, transform=utils.get_sketch_gen_transform(), max_erase_count=0, only_valid=False)
    #dataset3 = VectorizedSketchyDatasetV1(size=0.01, transform=utils.get_sketch_gen_transform(), max_erase_count=3, only_valid=True)
    #dataset = KaggleDatasetImgOnlyV1(size=1, mode='test')
    #print(dataset.__getitem__(1))
    #print(len(dataset.categorized_images.index))
    #dataset2 = KaggleDatasetImgOnlyV2(size=1, mode='train')
    #print( list(dataset2.categorized_images.index).index('miniature'))
    #dataset = SketchyDatasetPix2Pix(size=0.01)
    #print(len(dataset))
    #item = dataset.__getitem__(0)
    #print(item['A'].shape)
    #visualization.show_triplets([[item['A'], item['A'], item['B']]], './test.png', mode='image')
    #item = utils.convert_pix2pix_to_255(item)
    #visualization.show_triplets([[item['A'], item['A'], item['B']]], './test2.png', mode='image')

    #dataset = KaggleDatasetImgOnlyV1(mode='test', size=1.0)

    #dataset = VectorizedSketchyDatasetV1(size=0.01, transform=utils.get_sketch_gen_transform())

    #print(dataset.state_dict)
    #print(len(dataset), len(dataset.sketch_paths), len(dataset.photo_paths), len(dataset.vectorized_sketches))
    #print(len(dataset2), len(dataset2.sketch_paths), len(dataset2.photo_paths), len(dataset2.vectorized_sketches))
    #print(len(dataset3), len(dataset3.sketch_paths), len(dataset3.photo_paths), len(dataset3.vectorized_sketches))
    #dataset = KaggleInferenceDatasetV1()
    #print(dataset.__getitem__(0))
    #print(len(dataset))

    #dataset = KaggleDatasetV2()
    #print(dataset.sketch_paths[0])
    

    dataset = SketchyDatasetV1(size=1.0, mode='test')
    #dataset2 = InferenceDataset(dataset.photo_paths)
    #print(len(dataset), len(dataset.sketch_paths), len(dataset.photo_paths), len(dataset2))
