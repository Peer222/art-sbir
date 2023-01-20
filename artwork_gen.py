from pathlib import Path
import os
from tqdm.auto import tqdm
import random

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import utils
import data_preparation
from artwork_gen_utils.function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = utils.load_model("adain_models", model_type='AdaIN')

decoder = models['decoder']
vgg = models['encoder']

decoder.eval()
vgg.eval()

vgg.to(device)
decoder.to(device)

content_tf = test_transform(256, False)
style_tf = test_transform(256, False)

content_dataset = data_preparation.SketchyDatasetV1(size=1.0, transform=transforms.Lambda(lambda x: x), _sample=False)

style_dataset = data_preparation.get_datasets(dataset='KaggleDatasetImgOnlyV1', size=1.0, transform=style_tf)[0] # !!!! change !!!!


path = Path("data/sketchy/artworks")
if not path.is_dir():
    path.mkdir(parents=True, exist_ok=True)

with torch.inference_mode():
    for i in tqdm(range(len(content_dataset))):
        img_path = content_dataset.sketch_paths[i]
        folder = path / img_path.parent.name
        if not folder.is_dir():
            folder.mkdir(parents=True, exist_ok=True)

        content_img = content_tf(Image.open(content_dataset.photo_paths[i]).convert('RGB'))
        content_img = content_img.to(device).unsqueeze(0)

        index = random.randint(0, len(style_dataset) - 1)
        style_img = style_dataset.__getitem__(index)['image']
        style_img = style_img.to(device).unsqueeze(0)

        output_img = style_transfer(vgg, decoder, content_img, style_img, 1.0).cpu()

        save_image(output_img, str(folder / f"{img_path.stem}.jpg"))