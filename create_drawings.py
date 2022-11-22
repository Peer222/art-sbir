#!/usr/bin/python3

# modified test.py from informative-drawings

import argparse
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from drawing_utils.model import Generator
from drawing_utils.dataset import UnpairedDepthDataset
from PIL import Image

from pathlib import Path

# retrieves classes and selects first n classes depending on size parameter
def get_sketchy_classes(data_root, size:float=1):

    classes = sorted(entry.name for entry in os.scandir(data_root) if entry.is_dir() )
    if not classes:
        raise FileNotFoundError(f"No classes found in {data_root}")

    num = round(size * len(classes))
    return classes[:num]

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, type=str, choices=['contour', 'anime', 'opensketch'], help='Model name without suffix') # contour_style
parser.add_argument('--model_dir', type=str, default='models/drawing_models', help='Where the model checkpoints are saved') # changed
parser.add_argument('--results_dir', type=str, default='data/sketchy/', choices=['data/sketchy/', 'data/kaggle/'], help='where to save result images') # changed
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/sketchy/photos', choices=['data/sketchy/photos', 'TODO kaggle'], help='root directory of the dataset')

parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')

parser.add_argument('--n_blocks', type=int, default=3, help='number of resnet blocks for generator')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')

# needed within dataset
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--no_flip', type=bool, default=True, help='always true') # changed


opt = parser.parse_args()
print(opt)

data_dir = Path(opt.dataroot)
result_dir = Path(opt.results_dir) / f"{opt.name}_drawings"

if not result_dir.is_dir():
    result_dir.mkdir(exist_ok=True, parents=True)
is_sketchy = 'sketchy' in opt.dataroot

device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.no_grad():

    net_G = Generator(opt.input_nc, opt.output_nc, opt.n_blocks)
    # Load state dicts
    net_G.load_state_dict(torch.load(os.path.join(opt.model_dir, opt.name + '.pth'), map_location=device))
    print('loaded', os.path.join(opt.model_dir, opt.name + '.pth'))
    # Set model's test mode
    net_G.eval()

    transforms_r = [transforms.Resize(int(opt.size), Image.BICUBIC), transforms.ToTensor()]

    if is_sketchy:
        classes = get_sketchy_classes(data_root=data_dir, size=1)

        for img_cls in classes:
            data = UnpairedDepthDataset(data_dir / img_cls, '', opt, transforms_r=transforms_r, mode='test')
            dataloader = DataLoader(data, batch_size=opt.batchSize, shuffle=False)

            ###### Testing######   
            if not (result_dir / img_cls).is_dir():
                (result_dir / img_cls).mkdir(parents=True, exist_ok=True)

            for i, batch in enumerate(dataloader):

                img_r  = Variable(batch['r'])#.cuda()
                img_depth  = Variable(batch['depth'])#.cuda()
                real_A = img_r

                name = batch['name'][0]
        
                input_image = real_A
                image = net_G(input_image)
                save_image(image.data, result_dir / img_cls / f"{name}.png")

    else:
        # kaggle contains more photos than this dataset will contain -> adapt in drawing_utils/dataset.py (make_dataset)
        data = UnpairedDepthDataset(data_dir, '', opt, transforms_r=transforms_r, mode='test')
        dataloader = DataLoader(data, batch_size=opt.batchSize, shuffle=False)

        ###### Testing######   
        if not result_dir.is_dir():
            result_dir.mkdir(parents=True, exist_ok=True)

        for i, batch in enumerate(dataloader):

            img_r  = Variable(batch['r'])#.cuda()
            img_depth  = Variable(batch['depth'])#.cuda()
            real_A = img_r

            name = batch['name'][0]
        
            input_image = real_A
            image = net_G(input_image)
            save_image(image.data, result_dir / f"{name}.png")