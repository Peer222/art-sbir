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

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE=32
EPOCHS=5
LEARNING_RATE=0.01


def triplet_train(model:nn.Module, epochs:int, train_dataloader:DataLoader, test_dataloader:DataLoader, loss_fn, optimizer):

    start_time = timer()

    train_losses = []
    test_losses = []

    # calculation of all images needed to be able to check retrieval results?
    test_top1 = 0
    test_top5 = 0
    test_top10 = 0
    test_avg_rank = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):

        train_loss = 0
        test_loss = 0

        # training batch loop
        model.train()
        
        for batch, (sketch, pos_img, neg_img) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):

            sketch, pos_img, neg_img = sketch.to(device), pos_img.to(device), neg_img.to(device)

            s_logits = model(sketch)
            p_logits = model(pos_img)
            n_logits = model(neg_img)

            loss = loss_fn(s_logits, p_logits, n_logits)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss


        # testing batch_loop

        model.eval()
        with torch.inference_mode():

            for batch, (sketch, pos_img, neg_img) in enumerate(tqdm(test_dataloader, desc="Evaluation", leave=False)):

                sketch, pos_img, neg_img = sketch.to(device), pos_img.to(device), neg_img.to(device)

                s_logits = model(sketch)
                p_logits = model(pos_img)
                n_logits = model(neg_img)

                loss = loss_fn(s_logits, p_logits, n_logits)

                test_loss += loss

        train_losses.append(train_loss / len(train_dataloader))
        test_losses.append(test_loss / len(test_dataloader))

        print(f"Epoch {epoch+1} - Train loss: {train_losses[epoch]:.3f} | Test loss: {test_losses[epoch]}")

    end_time = timer()
    training_time = end_time - start_time

    return {"train_losses": train_losses, "test_losses": test_losses, "training_time": training_time}


# command line tool

msg = "TODO"

parser = argparse.ArgumentParser(description=msg)

parser.add_argument("-e", "--epochs", nargs=1, type=int, default=10, help="Set number of epochs for training - default:10")
parser.add_argument("-b", "--batch_size", nargs=1, type=int, default=32, help="Set batch_size for training - default:32")
parser.add_argument("-l", "--learning_rate", nargs=1, type=float, default=0.01, help="Set learning rate - default:0.01")
parser.add_argument("-m", "--model", nargs=1, type=str, default='ResNet50m', choices=['ResNet50m'], help="Choose a model - default:ResNet50m WOP")
parser.add_argument("-d", "--dataset", nargs=1, type=str, default='SketchyS', choices=['SketchyS', 'SketchyL'], help="Choose a dataset - default:SketchyS WOP")
parser.add_argument("--inference", action="store_true", help="If set extended inference will be executed after training WOP")

args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate # 5 * 10-4 used by clip

MODEL = args.model
DATASET = args.dataset

with_inference = args.inference

model = utils.load_model("ResNet50m.pth")
model = model.to(device)
print(f'model {MODEL} loaded')

inference_dict = {}
training_dict = {}
param_dict = {}
data_dict= {}


# options have to be added
train_dataset, test_dataset = data_preparation.get_datasets(size=0.5)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False) #num_workers = os.cpu_count() - dataset already suffled by sklearn.train_test_split
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False) #num_workers = os.cpu_count()

optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE) # adam used by clip (hyper params in paper)

loss_fn = utils.triplet_loss


training_dict = triplet_train(model, EPOCHS, train_dataloader, test_dataloader, loss_fn, optimizer)

param_dict = {"model": MODEL, "dataset": DATASET, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE}
data_dict = train_dataset.state_dict

if with_inference:
    print("inference TODO")

    inference_dict = inference.run_inference(model, test_dataset)

#save
utils.save_model(model, data_dict, training_dict, param_dict, inference_dict)