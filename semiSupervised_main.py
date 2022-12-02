import argparse
from tqdm.auto import tqdm
from timeit import default_timer as timer


import torch
from torch import nn

import semiSupervised_utils

import utils
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_sketch_gen(hp):
    model = models.Photo2Sketch(hp)
    model.to(device)
    #model.load_state_dict(torch.load('./modelCVPR21/QMUL/model_photo2Sketch_QMUL_2Dattention_8000_.pth'))


    optimizer = torch.optim.Adam(params=model.parameters(), lr=hp.learning_rate, betas=(0.5, 0.999))


    step = 0
    current_loss = 1e+10

    dataloader_Train, dataloader_Test = semiSupervised_utils.get_dataloader(hp)

    train_losses = []
    test_losses = []

    start_time = timer()

    for i_epoch in tqdm(range(hp.max_epoch)):

        train_loss = 0
        test_loss = 0

        model.train()
        for i_batch, batch_data in enumerate(dataloader_Train):
            rgb_image = batch_data['photo'].to(device)
            sketch_vector = batch_data['sketch_vector'].to(device).permute(1, 0, 2).float()
            length_sketch = batch_data['length'].to(device) - 1
            sketch_name = batch_data['sketch_path'][0]

            sup_p2s_loss, kl_cost_rgb, total_loss = model.Image2Sketch_Train(rgb_image, sketch_vector, length_sketch, step, sketch_name)

            step += 1

            if total_loss.item() < current_loss:
                #save model
                pass

            current_loss = total_loss.item()

            train_loss += total_loss / hp.batchsize

        for i_batch, batch_data in tqdm(enumerate(dataloader_Test)):

            test_loss += total_loss / hp.batchsize

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    training_time = timer() - start_time


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Photo2Sketch')

    parser.add_argument('--setup', type=str, default='Sketchy')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--nThreads', type=int, default=8)

    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--eval_freq_iter', type=int, default=1000)


    parser.add_argument('--enc_rnn_size', default=256)
    parser.add_argument('--dec_rnn_size', default=512)
    parser.add_argument('--z_size', default=128)

    parser.add_argument('--num_mixture', default=20)
    parser.add_argument('--input_dropout_prob', default=0.9)
    parser.add_argument('--output_dropout_prob', default=0.9)
    parser.add_argument('--batch_size_sketch_rnn', default=100)

    parser.add_argument('--kl_weight_start', default=0.01)
    parser.add_argument('--kl_decay_rate', default=0.99995)
    parser.add_argument('--kl_tolerance', default=0.2)
    parser.add_argument('--kl_weight', default=1.0)

    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--decay_rate', default=0.9999)
    parser.add_argument('--min_learning_rate', default=0.00001)
    parser.add_argument('--grad_clip', default=1.)

    hp = parser.parse_args()

    print(hp)

    train_sketch_gen(hp)