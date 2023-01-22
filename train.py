from timeit import default_timer as timer
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

import utils
import data_preparation
import models
import inference
import visualization

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

def get_classified_loss(loss_fn, model, sketch, pos_img, neg_img, labels):
    s_logits, cs_logits = model(sketch)
    p_logits, cp_logits = model(pos_img)
    n_logits, _ = model(neg_img)
    return loss_fn(s_logits, p_logits, n_logits, cs_logits, cp_logits, labels)

def get_loss(loss_fn, model, elements):
    s_logits = model(elements[0]) # sketch
    p_logits = model(elements[1]) # pos image
    n_logits = model(elements[2]) # neg image

    if len(s_logits) > 3: # img
        return loss_fn(s_logits, p_logits, n_logits)
    elif len(s_logits) == 2: # img, class
        return loss_fn(s_logits[0], p_logits[0], n_logits[0], s_logits[1], p_logits[1], elements[3])
    elif len(s_logits) == 3: # img, class, class2
        return loss_fn(s_logits[0], p_logits[0], n_logits[0], s_logits[1], p_logits[1], s_logits[2], p_logits[2], elements[3], elements[4])

def triplet_train(model:nn.Module, epochs:int, train_dataloader:DataLoader, test_dataloader:DataLoader, loss_fn, optimizer, with_classification):
    start_time = timer()

    train_losses = []
    test_losses = []

    iteration_loss_frequency = 10000 // train_dataloader.batch_size if epochs <= 6 else 0
    itest_size = 1000 // test_dataloader.batch_size
    itrain_losses = []
    itest_losses = []

    for epoch in tqdm(range(epochs), desc="Epochs"):

        train_loss = 0
        test_loss = 0

        itrain_loss = 0

        # training batch loop
        model.train()
        for batch, tuple in enumerate(train_dataloader):# removed tqdm enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
            elements = list(tuple)
            for i in range(len(elements)):
                elements[i] = elements[i].to(device)

            loss = get_loss(loss_fn, model, elements)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

            if iteration_loss_frequency and batch % iteration_loss_frequency == 0:
                itrain_losses.append((train_loss.item() - itrain_loss) / iteration_loss_frequency)
                itrain_loss = train_loss.item()

                itest_loss = 0
                model.eval()
                with torch.inference_mode():
                    for batch, tuple in enumerate(test_dataloader): # removed tqdm
                        itest_loss += get_loss(loss_fn, model, elements)
                        if batch >= itest_size: break
                itest_losses.append(itest_loss.item() / itest_size)
                model.train()

        # testing batch_loop
        model.eval()
        with torch.inference_mode():

            for batch, tuple in enumerate(test_dataloader): # removed tqdm

                test_loss += get_loss(loss_fn, model, elements)

        train_losses.append(train_loss.item() / len(train_dataloader))
        test_losses.append(test_loss.item() / len(test_dataloader))

        print(f"Epoch {epoch+1} - Train loss: {train_losses[epoch]:.5f} | Test loss: {test_losses[epoch]:.5f}", flush=True)

    return {"train_losses": train_losses, "test_losses": test_losses, "itrain_losses": itrain_losses, "itest_losses":itest_losses, "iteration_loss_frequency": iteration_loss_frequency, "iteration_test_size": itest_size, "training_time": timer() - start_time}


# command line tool

msg = "Starts training a model"

parser = argparse.ArgumentParser(description=msg)

parser.add_argument("-e", "--epochs", type=int, default=1, help="Set number of epochs for training - default:10")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="Set batch_size for training - default:32")
parser.add_argument("-l", "--learning_rate", type=float, default=0.00001, help="Set learning rate")
parser.add_argument("-m", "--model", type=str, default='openResNet50m.pth', help="Choose a model - default:openResNet50m.pth")
parser.add_argument('--model_type', type=str, default='ModifiedResNet_with_classification', choices=['ModifiedResNet', 'ModifiedResNet_with_classification', 'DrawingGenerator', 'Photo2Sketch'], help='Type of model')
parser.add_argument("-d", "--dataset", type=str, default='SketchyV1', choices=['SketchyV1', 'SketchyV2', 'KaggleV1', 'KaggleV2', 'AugmentedKaggleV1', 'AugmentedKaggleV2', 'MixedDatasetV1', 'MixedDatasetV2'], help="Choose a dataset")
parser.add_argument("-s", "--dsize", type=float, default=1.0, help="Fraction of dataset used during training and testing")
parser.add_argument("--inference", action="store_true", help="If set extended inference will be executed after training")
parser.add_argument('--feature_folder', default=None, help="If None image features will be computed for inference otherwise loaded from data/image_features/[feature_folder]")
parser.add_argument("--no_training", action='store_true', help="If set no training will be executed")
parser.add_argument("-w", "--weight_decay", type=float, default=0.002, help="Weight decay for optimizer")
parser.add_argument('--img_type', type=str, default='photos', choices=['photos', 'anime_drawings', 'contour_drawings', 'images', 'artworks'], help="Image type")
parser.add_argument('--sketch_type', default='sketches_png', choices=['sketches_png', 'contour_drawings', 'opensketch_drawings', 'photo_sketch', 'adain_sketches', 'combination', 'dilated_opensketch_drawings'])
parser.add_argument('--loss_type', default='euclidean', choices=['euclidean', 'cosine'])
parser.add_argument('--loss_margin', type=float, default=0.2)

args = parser.parse_args()

if args.sketch_type == 'combination': args.sketch_type = ['contour_drawings', 'opensketch_drawings', 'dilated_opensketch_drawings']

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate # 5 * 10-4 used by clip with adam
WEIGHT_DECAY = args.weight_decay
LOSS_TYPE = args.loss_type
utils.MARGIN = args.loss_margin

MODEL = args.model
DATASET = args.dataset

model = utils.load_model(MODEL, dataset=DATASET, model_type=args.model_type)
model.freeze_layers()
model.to(device)

inference_dict = {}
training_dict = {}
param_dict = {}
data_dict= {}

img_format = 'jpg'
if 'drawings' in args.img_type: img_format = 'png'


# options have to be added
train_dataset, test_dataset = data_preparation.get_datasets(dataset=DATASET, size=args.dsize, sketch_type=args.sketch_type, img_type=args.img_type, img_format=img_format, transform=model.transform)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=min(4, os.cpu_count()), shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=min(4, os.cpu_count()), shuffle=False)

#optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE) # adam used by clip (hyper params in paper)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # adam with lr 10^-5, wd 0.002, betas default as in sketchy original paper


with_classification = 'with_classification' in type(model).__name__ and 'V2' in train_dataset.state_dict['dataset']
print("with classification: ", with_classification)

if LOSS_TYPE == 'euclidean':
    if with_classification: 
        if 'Sketchy' in train_dataset.state_dict['dataset']: loss_fn = utils.triplet_euclidean_loss_with_classification
        elif 'Kaggle' in train_dataset.state_dict['dataset']: loss_fn = utils.triplet_euclidean_loss_with_classification2
    else: loss_fn = utils.triplet_euclidean_loss
elif LOSS_TYPE == 'cosine':
    if with_classification: 
        if 'Sketchy' in train_dataset.state_dict['dataset']: loss_fn = utils.triplet_cosine_loss_with_classification
        elif 'Kaggle' in train_dataset.state_dict['dataset']: loss_fn = utils.triplet_cosine_loss_with_classification2
    else: loss_fn = utils.triplet_cosine_loss

param_dict = {"model": MODEL, "trained_layers": model.trained_layers, "dataset": DATASET, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "optimizer": type(optimizer).__name__,
            "loss_fn": type(loss_fn).__name__, "loss_margin": utils.MARGIN, "loss_type": LOSS_TYPE}
if with_classification:
    param_dict['loss_weights'] = [loss_fn.classification_weight, loss_fn.classification_weight2]

data_dict = train_dataset.state_dict

print(param_dict, flush=True)
print(data_dict, flush=True)

if not args.no_training: training_dict = triplet_train(model, EPOCHS, train_dataloader, test_dataloader, loss_fn, optimizer, with_classification)

if args.inference: inference_dict = inference.run_inference(model, test_dataset, args.feature_folder, LOSS_TYPE)

# saves model and/or dictionaries
folder = utils.save_model(model, data_dict, training_dict, param_dict, inference_dict)

# saves visualizations in result folder
visualization.visualize(folder, training_dict, inference_dict)