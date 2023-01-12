from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

import argparse
from pathlib import Path

image_transformV1 = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            #transforms.CenterCrop(size=(224, 224)),
            transforms.Lambda(lambda img : img.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

# 1
sketch_transformV1 = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.Lambda(lambda img : img.convert('RGB')),

            transforms.RandomApply(p=0.5, transforms=[
                transforms.RandomPerspective(distortion_scale=0.3, p=1.0, fill=255),
                transforms.RandomAffine(degrees=0, scale=(1.05, 1.3), fill=255),
            ]),
            transforms.RandomApply(p=0.5, transforms=[
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-7, 7, -7, 7), fill=255)
            ]),

            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.05, 0.2), value=1.0), # https://arxiv.org/abs/1708.04896
            #transforms.ToPILImage()
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

# 2
sketch_transformV2 = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.Lambda(lambda img : img.convert('RGB')),

            transforms.RandomApply(p=0.5, transforms=[
                transforms.RandomPerspective(distortion_scale=0.35, p=1.0, fill=255),
                transforms.RandomAffine(degrees=0, scale=(1.05, 1.3), fill=255),
            ]),
            transforms.RandomApply(p=0.7, transforms=[
                transforms.RandomAffine(degrees=15, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-10, 10, -10, 10), fill=255)
            ]),

            transforms.ToTensor(),
            transforms.RandomErasing(p=0.7, scale=(0.05, 0.1), value=1.0),
            transforms.RandomErasing(p=0.7, scale=(0.05, 0.1), ratio=(0.2, 2.0), value=1.0),
            transforms.RandomErasing(p=0.7, scale=(0.05, 0.1), ratio=(0.4, 4.0), value=1.0),
            #transforms.ToPILImage()
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

def get_transformation(type='image_transform', version='V1'):
    t = f"{type}{version}"
    return eval(t), t


def test_transform(options):
    image = Image.open('../transformations/test.png')
    for i in range(10):
        augmented_img = sketch_transformV2(image)
        augmented_img.save(f'../transformations/transformed7_img_{i}.png')

def dilate(options):
    kernel = np.ones((4, 4), np.uint8)
    print("hi")

    dir = Path("data/kaggle")
    img_dir = dir / options[0]
    print(img_dir)
    image_paths = list(img_dir.glob("*.png"))

    new_dir = dir / f"dilated_{options[0]}"
    if not new_dir.is_dir():
        new_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:

        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.dilate(img, kernel)

        img = np.asarray(img)
        np.place(img, img>250, [255])
        np.place(img, img<250, [0])

        cv2.imwrite(str(new_dir / path.name), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--method', required=True, choices=['test_transform', 'dilate'])
    parser.add_argument('-o', '--options', default=[])

    args = parser.parse_args()
    if not isinstance(args.options, list):
        args.options = [args.options]

    eval(args.method)(args.options)