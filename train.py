from timeit import default_timer as timer
from tqdm.auto import tqdm
import argparse
from pathlib import Path

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
    s_logits, _ = model(sketch)
    p_logits, c_logits = model(pos_img)
    n_logits, _ = model(neg_img)
    return loss_fn(s_logits, p_logits, n_logits, c_logits, labels)

def triplet_train(model:nn.Module, epochs:int, train_dataloader:DataLoader, test_dataloader:DataLoader, loss_fn, optimizer, with_classification):
    start_time = timer()

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(epochs), desc="Epochs"):

        train_loss = 0
        test_loss = 0

        # training batch loop
        model.train()
        for batch, tuple in enumerate(train_dataloader):# removed tqdm enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
            if with_classification: sketch, pos_img, neg_img, labels = tuple
            else: sketch, pos_img, neg_img = tuple

            sketch, pos_img, neg_img = sketch.to(device), pos_img.to(device), neg_img.to(device)

            if with_classification:
                loss = get_classified_loss(loss_fn, model, sketch, pos_img, neg_img, labels)
            else: 
                s_logits, p_logits, n_logits = model(sketch), model(pos_img), model(neg_img)
                loss = loss_fn(s_logits, p_logits, n_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

        # testing batch_loop
        model.eval()
        with torch.inference_mode():

            for batch, tuple in enumerate(test_dataloader): # removed tqdm
                if with_classification: sketch, pos_img, neg_img, labels = tuple
                else: sketch, pos_img, neg_img = tuple

                sketch, pos_img, neg_img = sketch.to(device), pos_img.to(device), neg_img.to(device)

                if with_classification:
                    loss = get_classified_loss(loss_fn, model, sketch, pos_img, neg_img, labels)
                else: 
                    s_logits, p_logits, n_logits = model(sketch), model(pos_img), model(neg_img)
                    loss = loss_fn(s_logits, p_logits, n_logits)

                test_loss += loss

        train_losses.append(train_loss.item() / len(train_dataloader))
        test_losses.append(test_loss.item() / len(test_dataloader))

        print(f"Epoch {epoch+1} - Train loss: {train_losses[epoch]:.5f} | Test loss: {test_losses[epoch]:.5f}", flush=True)

    return {"train_losses": train_losses, "test_losses": test_losses, "training_time": timer() - start_time}


# command line tool

msg = "Starts training a model"

parser = argparse.ArgumentParser(description=msg)

parser.add_argument("-e", "--epochs", type=int, default=1, help="Set number of epochs for training - default:10")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="Set batch_size for training - default:32")
parser.add_argument("-l", "--learning_rate", type=float, default=0.00001, help="Set learning rate")
parser.add_argument("-m", "--model", type=str, default='openResNet50m.pth', help="Choose a model - default:openResNet50m.pth")
parser.add_argument("-d", "--dataset", type=str, default='Sketchy', choices=['Sketchy', 'SketchyV2', 'Kaggle'], help="Choose a dataset")
parser.add_argument("-s", "--dsize", type=float, default=1.0, help="Fraction of dataset used during training and testing")
parser.add_argument("--inference", action="store_true", help="If set extended inference will be executed after training")
parser.add_argument("-w", "--weight_decay", type=float, default=0.002, help="Weight decay for optimizer")

args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate # 5 * 10-4 used by clip with adam
WEIGHT_DECAY = args.weight_decay

MODEL = args.model
DATASET = args.dataset

with_inference = args.inference

model = utils.load_model(MODEL, dataset=DATASET)
model.freeze_layers()
model.to(device)

with_classification = 'with_classification' in type(model).__name__

inference_dict = {}
training_dict = {}
param_dict = {}
data_dict= {}


# options have to be added
train_dataset, test_dataset = data_preparation.get_datasets(dataset=DATASET, size=args.dsize, transform=model.transform)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True) #num_workers = os.cpu_count()
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False) #num_workers = os.cpu_count()

#optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE) # adam used by clip (hyper params in paper)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # adam with lr 10^-5, wd 0.002, betas default as in sketchy original paper

if with_classification: loss_fn = utils.triplet_euclidean_loss_with_classification
else: loss_fn = utils.triplet_euclidean_loss

param_dict = {"model": MODEL, "trained_layers": model.trained_layers, "dataset": DATASET, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "optimizer": type(optimizer).__name__, "loss_fn": type(loss_fn).__name__}
data_dict = train_dataset.state_dict

training_dict = triplet_train(model, EPOCHS, train_dataloader, test_dataloader, loss_fn, optimizer, with_classification)

if with_inference: inference_dict = inference.run_inference(model, test_dataset)

# save
folder = utils.save_model(model, data_dict, training_dict, param_dict, inference_dict)

# saves visualizations in result folder
visualization.visualize(folder, training_dict, inference_dict)