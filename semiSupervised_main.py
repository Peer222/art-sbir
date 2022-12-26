import argparse
from tqdm.auto import tqdm
from timeit import default_timer as timer
import os
from pathlib import Path
from PIL import Image
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

import semiSupervised_utils

import utils
import models
import data_preparation
import visualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_sketch_gen(model, dataloader_train, dataloader_test, optimizer, hp):
    
    train_losses = {'total_loss': [], 'kl_loss': [], 'reconstruction_loss': []}
    test_losses = {'total_loss': [], 'kl_loss': [], 'reconstruction_loss': []}

    start_time = timer()

    step = 0
    current_loss = 1e+10

    result_path = None

    for i_epoch in tqdm(range(hp.max_epoch), desc='epoch'):

        train_loss = {'total_loss': 0, 'kl_loss': 0, 'reconstruction_loss': 0}
        test_loss = {'total_loss': 0, 'kl_loss': 0, 'reconstruction_loss': 0}

        model.train()
        for i_batch, batch_data in enumerate(dataloader_train):

            rgb_image = batch_data['photo'].to(device)
            sketch_vector = batch_data['sketch_vector'].to(device).permute(1, 0, 2).float()
            length_sketch = batch_data['length'].to(device) - 1

            # Encoder
            backbone_feature, rgb_encoded_dist = model.Image_Encoder(rgb_image)
            rgb_encoded_dist_z_vector = rgb_encoded_dist.rsample()

            prior_distribution = torch.distributions.Normal(torch.zeros_like(rgb_encoded_dist.mean), torch.ones_like(rgb_encoded_dist.stddev))
            kl_cost_rgb = torch.max(torch.distributions.kl_divergence(rgb_encoded_dist, prior_distribution).mean(), torch.tensor(hp.kl_tolerance).to(device))


            # decoder
            photo2sketch_output = model.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, sketch_vector, length_sketch + 1)

            ### end state already set in parse_svg but additionally to last line
            # end token added after sketch decoder -> otherwise error  why???
            sketch_vector = sketch_vector.permute(1, 0, 2)
            end_token = torch.stack([torch.tensor([0, 0, 0, 0, 1])] * sketch_vector.shape[0]).unsqueeze(1).to(device).float()
            x_target = torch.cat([sketch_vector, end_token], 1)

            curr_learning_rate = ((hp.learning_rate - hp.min_learning_rate) * (hp.decay_rate) ** step + hp.min_learning_rate)
            curr_kl_weight = (hp.kl_weight - (hp.kl_weight - hp.kl_weight_start) * (hp.kl_decay_rate) ** step)

            #sup_p2s_loss = semiSupervised_utils.sketch_reconstruction_loss(photo2sketch_output, x_target)  #TODO: Photo to Sketch Loss
            sup_p2s_loss = semiSupervised_utils.sketch_reconstruction_loss_withoutMask(photo2sketch_output, x_target)
            loss = sup_p2s_loss + curr_kl_weight*kl_cost_rgb

            semiSupervised_utils.set_learningRate(optimizer, curr_learning_rate)

            optimizer.zero_grad()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optimizer.step()

            step += 1
            
            loss_dict = {'reconstruction_loss': sup_p2s_loss.item(), 'kl_loss': kl_cost_rgb.item(), 'total_loss': loss.item()}
            train_loss = utils.process_losses(train_loss, loss_dict, hp.batchsize, 'add')

        train_losses = utils.process_losses(train_losses, train_loss, len(dataloader_train), 'append')

        print(f"Epoch:{i_epoch} ** Train ** sup_p2s_loss:{train_losses['reconstruction_loss'][i_epoch]} ** kl_cost_rgb:{train_losses['kl_loss'][i_epoch]} ** Total_loss:{train_losses['total_loss'][i_epoch]}", flush=True)

        model.eval()
        with torch.inference_mode():
            for i_batch, batch_data in enumerate(dataloader_test):
                rgb_image = batch_data['photo'].to(device)
                sketch_vector = batch_data['sketch_vector'].to(device).permute(1, 0, 2).float()
                length_sketch = batch_data['length'].to(device) - 1

                # Encoder
                backbone_feature, rgb_encoded_dist = model.Image_Encoder(rgb_image)
                rgb_encoded_dist_z_vector = rgb_encoded_dist.rsample()

                prior_distribution = torch.distributions.Normal(torch.zeros_like(rgb_encoded_dist.mean), torch.ones_like(rgb_encoded_dist.stddev))
                kl_cost_rgb = torch.max(torch.distributions.kl_divergence(rgb_encoded_dist, prior_distribution).mean(), torch.tensor(hp.kl_tolerance).to(device))


                # decoder
                photo2sketch_output = model.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, sketch_vector, length_sketch + 1)

                # end token added after sketch decoder -> otherwise error  why???
                sketch_vector = sketch_vector.permute(1, 0, 2)
                end_token = torch.stack([torch.tensor([0, 0, 0, 0, 1])] * sketch_vector.shape[0]).unsqueeze(1).to(device).float()
                x_target = torch.cat([sketch_vector, end_token], 1)

                curr_kl_weight = (hp.kl_weight - (hp.kl_weight - hp.kl_weight_start) * (hp.kl_decay_rate) ** step)

                #sup_p2s_loss = semiSupervised_utils.sketch_reconstruction_loss(photo2sketch_output, x_target)  #TODO: Photo to Sketch Loss
                sup_p2s_loss = semiSupervised_utils.sketch_reconstruction_loss_withoutMask(photo2sketch_output, x_target)
                loss = sup_p2s_loss + curr_kl_weight*kl_cost_rgb

                loss_dict = {'reconstruction_loss': sup_p2s_loss.item(), 'kl_loss': kl_cost_rgb.item(), 'total_loss': loss.item()}
                test_loss = utils.process_losses(test_loss, loss_dict, hp.batchsize, 'add')

        test_losses = utils.process_losses(test_losses, test_loss, len(dataloader_test), 'append')

        print(f"Epoch:{i_epoch} ** Test ** sup_p2s_loss:{test_losses['reconstruction_loss'][i_epoch]} ** kl_cost_rgb:{test_losses['kl_loss'][i_epoch]} ** Total_loss:{test_losses['total_loss'][i_epoch]}", flush=True)
        # total_losses not comparable due to changing curr_kl_weighting -> compare only by two seperate losses and may be add them 

        if (i_epoch+1) % hp.save_rate == 0:
            param_dict['epoch'] = i_epoch + 1
            training_dict = {"train_losses": train_losses, "test_losses": test_losses, "training_time": timer() - start_time}

            # if not result_path or (i_epoch+1) % 25 == 0: result_path = utils.save_model(model, dataset_train.state_dict, training_dict, param_dict, inference_dict={})
            result_path = utils.save_model(model, dataset_train.state_dict, training_dict, param_dict, inference_dict={}) # saving more often because of larger dataset
            model.to(device)

            create_sample_sketches(model, dataset_test, dataloader_test, hp, result_path, i_epoch + 1, max=25)
            visualization.build_all_loss_curves(train_losses, test_losses, result_path, i_epoch + 1)

    return {"train_losses": train_losses, "test_losses": test_losses, "training_time": timer() - start_time}


def create_sample_sketches(model, dataset_test, dataloader_test, hp, result_path, epoch, max=15):
    samples = []

    model.eval()
    with torch.inference_mode():
        for i_batch, batch_data in enumerate(dataloader_test):
            if i_batch * hp.batchsize > max: break

            rgb_image = batch_data['photo'].to(device)
            sketch_vector = batch_data['sketch_vector'].to(device).permute(1, 0, 2).float()
            length_sketch = batch_data['length'].to(device) - 1

            backbone_feature, rgb_encoded_dist = model.Image_Encoder(rgb_image)
            rgb_encoded_dist_z_vector = rgb_encoded_dist.rsample()

            photo2sketch_output, attention_plot = model.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, sketch_vector, length_sketch + 1, isTrain=False)

            for i in range(len(photo2sketch_output)):
                if i_batch * hp.batchsize + i > max: break
                sketch = photo2sketch_output[i]
                #image = rgb_image[i]

                svg_path = result_path / f'svgs_{epoch}'
                if not svg_path.is_dir(): svg_path.mkdir(parents=True, exist_ok=True)
                semiSupervised_utils.build_svg(sketch.cpu(), (256, 256), svg_path / f"sketch_{i}.svg")

                tuple_path = result_path / f'tuples_{epoch}'
                if not tuple_path.is_dir(): tuple_path.mkdir(parents=True, exist_ok=True)
                with open(tuple_path / (f"sketch_{i}" + '.json'), 'w') as f:
                    json.dump(sketch.cpu().tolist(), f)
                with open(tuple_path / f"original_sketch_{i}.json", 'w') as f:
                    json.dump(sketch_vector[i].cpu().tolist(), f)

                # masking from original sketch
                #sketch[length_sketch[i]:, 4 ] = 1.0
                #sketch[length_sketch[i]:, 2:4] = 0.0

                sketch_path = dataset_test.sketch_paths[i_batch * hp.batchsize + i] #Quickdraw
                image_path = dataset_test.photo_paths[i_batch * hp.batchsize + i] #QUickdraw

                image = Image.open(image_path) #quickdraw
                rasterized_sketch = semiSupervised_utils.batch_rasterize_relative(sketch.unsqueeze(0)).squeeze()
                original_sketch = Image.open(Path('data/sketchy/sketches_png') / sketch_path.parent.name / (sketch_path.stem + '.png')) #quickdraw
                samples.append((image, rasterized_sketch.cpu(), original_sketch)) #original_sketch quickdraw -> .cpu()

    visualization.show_triplets(samples, result_path / f'samples_{epoch}.png', mode='image')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Photo2Sketch')

    parser.add_argument('--setup', type=str, default='Sketchy')
    parser.add_argument('--batchsize', type=int, default=64) # previous 1 / paper used 64
    parser.add_argument('--nThreads', type=int, default=8)

    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--eval_freq_iter', type=int, default=1000)

    # has to be changed in utils load model as well
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

    parser.add_argument('--save_rate', type=int, default=30)

    hp = parser.parse_args()

    # if size is changed sketch vector folder has to be deleted 
    dataset_train, dataset_test = data_preparation.get_datasets(dataset='VectorizedSketchyV1', size=0.1, img_format='jpg', img_type='photos', transform=None, max_erase_count=1, only_valid=True)

    dataloader_train = DataLoader(dataset_train, batch_size=hp.batchsize, shuffle=True, num_workers=min(4, os.cpu_count()))
    dataloader_test = DataLoader(dataset_test, batch_size=hp.batchsize, shuffle=False, num_workers=min(4, os.cpu_count()))

    # initial model is used
    #initial_model = 'Photo2Sketch_QuickDrawDatasetV1_2022-12-20_05-20.pth'#'../../additional/semi_supervised_test/models/original_model_quickdraw_150000.pth'#'Photo2Sketch_QuickDrawDatasetV1_2022-12-20_05-20.pth'

    initial_model = 'Photo2Sketch_QuickDrawDatasetV1_2022-12-20_10-13.pth'
    model = utils.load_model(initial_model, 'VectorizedSketchyV1', max_seq_len=dataset_test.maximum_length) #models.Photo2Sketch(hp.z_size, hp.dec_rnn_size, hp.num_mixture, dataset_train.max_seq_len)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=hp.learning_rate, betas=(0.5, 0.999))

    param_dict = vars(hp)
    param_dict['loaded_model'] = initial_model
    param_dict['start_token'] = '[0, 0, 1, 0, 0]'

    #create_sample_sketches(model, dataset_test, dataloader_test, hp, Path('./results/Photo2Sketch_QuickDrawDatasetV1_2022-12-15_00-32'), 0, 10)
    #create_sample_sketches(model, dataset_test, dataloader_test, hp, Path('./results/test'), 'own_Quickdraw_withoutMask_on_Sketchy_photos', 30)

    training_dict = train_sketch_gen(model, dataloader_train, dataloader_test, optimizer, hp)

    inference_dict = {}

    #result_path = utils.save_model(model, dataset_train.state_dict, training_dict, param_dict, inference_dict)

