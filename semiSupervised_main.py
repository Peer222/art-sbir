import argparse
from tqdm.auto import tqdm
from timeit import default_timer as timer
import os
from pathlib import Path
from PIL import Image

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
    test_losses = {}

    start_time = timer()

    step = 0
    current_loss = 1e+10

    for i_epoch in tqdm(range(hp.max_epoch)):

        train_loss = []

        model.train()
        for i_batch, batch_data in tqdm(enumerate(dataloader_train)):

            rgb_image = batch_data['photo'].to(device)
            sketch_vector = batch_data['sketch_vector'].to(device).permute(1, 0, 2).float()
            length_sketch = batch_data['length'].to(device) - 1
            sketch_name = batch_data['sketch_path']

            # Encoder
            backbone_feature, rgb_encoded_dist = model.Image_Encoder(rgb_image)
            rgb_encoded_dist_z_vector = rgb_encoded_dist.rsample()

            prior_distribution = torch.distributions.Normal(torch.zeros_like(rgb_encoded_dist.mean), torch.ones_like(rgb_encoded_dist.stddev))
            kl_cost_rgb = torch.max(torch.distributions.kl_divergence(rgb_encoded_dist, prior_distribution).mean(), torch.tensor(hp.kl_tolerance).to(device))


            # decoder
            photo2sketch_output = model.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, sketch_vector, length_sketch + 1)

            ### end state already set in parse_svg but additionally to last line
            #end_token = torch.stack([torch.tensor([0, 0, 0, 0, 1])] * rgb_image.shape[0]).unsqueeze(0).to(device).float()
            #batch = torch.cat([sketch_vector, end_token], 0)
            #x_target = batch.permute(1, 0, 2)  # batch-> Seq_Len, Batch, Feature_dim
            x_target = sketch_vector.permute(1, 0, 2)

            curr_learning_rate = ((hp.learning_rate - hp.min_learning_rate) * (hp.decay_rate) ** step + hp.min_learning_rate)
            curr_kl_weight = (hp.kl_weight - (hp.kl_weight - hp.kl_weight_start) * (hp.kl_decay_rate) ** step)

            sup_p2s_loss = semiSupervised_utils.sketch_reconstruction_loss(photo2sketch_output, x_target)  #TODO: Photo to Sketch Loss
            loss = sup_p2s_loss + curr_kl_weight*kl_cost_rgb

            semiSupervised_utils.set_learningRate(optimizer, curr_learning_rate)

            optimizer.zero_grad()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optimizer.step()


            print(f'Step:{step} ** sup_p2s_loss:{sup_p2s_loss} ** kl_cost_rgb:{kl_cost_rgb} ** Total_loss:{loss}', flush=True)

            step += 1

            if loss.item() < current_loss:
                #save model
                pass

            current_loss = loss.item()

            if step % 5 == 0:
                train_losses['reconstruction_loss'].append(sup_p2s_loss.item() / hp.batchsize)
                train_losses['kl_loss'].append(kl_cost_rgb.item() / hp.batchsize)
                train_losses['total_loss'].append(loss.item() / hp.batchsize)

    return {"train_losses": train_losses, "test_losses": test_losses, "training_time": timer() - start_time}

def create_sample_sketches(model, dataloader_test, hp, result_path, max=10):
    samples = []

    model.eval()
    with torch.inference_mode():
        for i_batch, batch_data in enumerate(dataloader_test):
            if i_batch > max: break

            rgb_image = batch_data['photo'].to(device)
            sketch_vector = batch_data['sketch_vector'].to(device).permute(1, 0, 2).float()
            length_sketch = batch_data['length'].to(device) - 1
            sketch_path = batch_data['sketch_path']

            backbone_feature, rgb_encoded_dist = model.Image_Encoder(rgb_image)
            rgb_encoded_dist_z_vector = rgb_encoded_dist.rsample()

            photo2sketch_output = model.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, sketch_vector, length_sketch + 1)

            original_sketch = Image.open(Path('data/sketchy/sketches_png') / sketch_path.parent.name / (sketch_path.stem + '.png'))
            rasterized_sketch = semiSupervised_utils.batch_rasterize_relative(photo2sketch_output)
            samples.append((rgb_image, rasterized_sketch, original_sketch))

            semiSupervised_utils.build_svg(photo2sketch_output, result_path / sketch_path.name)

    visualization.show_triplets(samples, result_path / 'samples.png')


"""
def Image2Sketch_Train(self, rgb_image, sketch_vector, length_sketch, step, sketch_name):
                
        ##############################################################
        ##############################################################
        # Cross Modal the Decoding 
        ##############################################################
        ##############################################################

        if step%5 == 0:
        
            data = {}
            data['Reconstrcution_Loss'] = sup_p2s_loss
            data['kl_'] = kl_cost_rgb
            data['Total Loss'] = loss
        
            self.visualizer.plot_scalars(data, step)


        if step%1 == 0:

            folder_name = os.path.join('./CVPR_SSL/' + '_'.join(sketch_name.split('/')[-1].split('_')[:-1]))
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            sketch_vector_gt = sketch_vector.permute(1, 0, 2)

            save_sketch(sketch_vector_gt[0], sketch_name)


            with torch.no_grad():
                photo2sketch_gen, attention_plot  = \
                    self.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, sketch_vector, length_sketch+1, isTrain=False)

            sketch_vector_gt = sketch_vector.permute(1, 0, 2)


            for num, len in enumerate(length_sketch):
                photo2sketch_gen[num, len:, 4 ] = 1.0
                photo2sketch_gen[num, len:, 2:4] = 0.0

            save_sketch_gen(photo2sketch_gen[0], sketch_name)

            sketch_vector_gt_draw = semiSupervised_utils.batch_rasterize_relative(sketch_vector_gt)
            photo2sketch_gen_draw = semiSupervised_utils.batch_rasterize_relative(photo2sketch_gen)

            batch_redraw = []
            plot_attention = showAttention(attention_plot, rgb_image, sketch_vector_gt_draw, photo2sketch_gen_draw, sketch_name)
            # max_image = 5
            # for a, b, c, d in zip(sketch_vector_gt_draw[:max_image], rgb_image.cpu()[:max_image],
            #                       photo2sketch_gen_draw[:max_image], plot_attention[:max_image]):
            #     batch_redraw.append(torch.cat((1. - a, b, 1. - c,  d), dim=-1))
            #
            # torchvision.utils.save_image(torch.stack(batch_redraw), './Redraw_Photo2Sketch_'
            #                              + self.hp.setup + '/redraw_{}.jpg'.format(step),
            #                              nrow=1, normalize=False)

            # data = {'attention_1': [], 'attention_2':[]}
            # for x in attention_plot:
            #     data['attention_1'].append(x[0])
            #     data['attention_2'].append(x[2])
            #
            # data['attention_1'] = torch.stack(data['attention_1'])
            # data['attention_2'] = torch.stack(data['attention_2'])
            #
            # self.visualizer.vis_image(data, step)



        # return sup_p2s_loss, kl_cost_rgb, loss

        return 0, 0, 0
"""

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

    dataset_train, dataset_test = data_preparation.get_datasets(dataset='VectorizedSketchyV1', size=0.01, transform=utils.get_sketch_gen_transform())

    dataloader_train = DataLoader(dataset_train, batch_size=hp.batchsize, shuffle=False, num_workers=os.cpu_count())
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=os.cpu_count())


    model = models.Photo2Sketch(hp.z_size, hp.dec_rnn_size, hp.num_mixture, dataset_train.max_seq_len)
    model.to(device)
    #model.load_state_dict(torch.load('./modelCVPR21/QMUL/model_photo2Sketch_QMUL_2Dattention_8000_.pth'))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=hp.learning_rate, betas=(0.5, 0.999))

    param_dict = vars(hp)

    training_dict = train_sketch_gen(model, dataloader_train, dataloader_test, optimizer)

    inference_dict = {}

    result_path = utils.save_model(model, dataset_train.state_dict, training_dict, param_dict, inference_dict)

    create_sample_sketches(model, dataloader_test, hp, result_path)