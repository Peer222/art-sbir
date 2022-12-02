from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torchvision.models as backbone_

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Photo2Sketch(nn.Module):

    def __init__(self, hp):
        super(Photo2Sketch, self).__init__()
        self.Image_Encoder = EncoderCNN()
        # self.Image_Decoder = DecoderCNN()
        # self.Sketch_Encoder = EncoderRNN(hp)
        self.Sketch_Decoder = DecoderRNN2D(hp)
        self.hp = hp
        # self.apply(weights_init_normal)

    def freeze_weights(self):
        for name, x in self.named_parameters():
            x.requires_grad = False

    def unfreeze_weights(self):
        for name, x in self.named_parameters():
            x.requires_grad = True


class EncoderCNN(nn.Module):
    def __init__(self, hp=None):
        super(EncoderCNN, self).__init__()
        self.feature = backbone_.vgg16(pretrained=True).features
        self.pool_method = nn.AdaptiveMaxPool2d(1)
        self.fc_mu = nn.Linear(512, 128)
        self.fc_std = nn.Linear(512, 128)

    def forward(self, x):
        backbone_feature = self.feature(x)
        x = torch.flatten(self.pool_method(backbone_feature), start_dim=1)
        mean = self.fc_mu(x)
        log_var = self.fc_std(x)
        posterior_dist = torch.distributions.Normal(mean, torch.exp(0.5 * log_var))
        return backbone_feature, posterior_dist


class DecoderRNN2D(nn.Module):
    def __init__(self, hp):
        super(DecoderRNN2D, self).__init__()
        self.fc_hc = nn.Linear(hp.z_size, 2 * hp.dec_rnn_size)
        self.lstm = nn.LSTM(hp.dec_rnn_size + 5, hp.dec_rnn_size)
        self.fc_params = nn.Linear(hp.dec_rnn_size, 6 * hp.num_mixture + 3)
        self.hp = hp
        self.attention_cell = AttentionCell2D(hp.dec_rnn_size)


    def forward(self, backbone_feature, z_vector, sketch_vector=None, seq_len=None, isTrain=True):

        batch_size = z_vector.shape[0]
        start_token = torch.stack([torch.tensor([0, 0, 1, 0, 0])] * batch_size).unsqueeze(0).float().to(device)


        self.training = isTrain
        output_hiddens =  torch.FloatTensor(batch_size, self.hp.max_seq_len + 1, self.hp.dec_rnn_size).fill_(0).to(device)


        hidden, cell = torch.split(F.tanh(self.fc_hc(z_vector)), self.hp.dec_rnn_size, 1)
        hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())



        if self.training:
            batch_init = torch.cat([start_token, sketch_vector], 0)
            num_steps = sketch_vector.shape[0] + 1
            for i in range(num_steps):
                state_point = batch_init[i, :, ]
                att_feature, _ = self.attention_cell.forward(backbone_feature, hidden_cell[0].squeeze(0))
                concat_context = torch.cat([att_feature, state_point], 1).unsqueeze(0)  # batch_size x (num_channel + num_embedding)
                _, hidden_cell = self.lstm(concat_context, hidden_cell)
                output_hiddens[:, i, :] = hidden_cell[0].squeeze(0)  # LSTM hidden index (0: hidden, 1: Cell)

            y_output = self.fc_params(output_hiddens)

            """ Split the data"""
            z_pen_logits = y_output[:, :, 0:3]
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.chunk(y_output[:, :, 3:], 6, 2)
            z_pi = F.softmax(z_pi, dim=-1)
            z_sigma1 = torch.exp(z_sigma1)
            z_sigma2 = torch.exp(z_sigma2)
            z_corr = torch.tanh(z_corr)

            return [z_pi.reshape(-1, 20), z_mu1.reshape(-1, 20), z_mu2.reshape(-1, 20), \
               z_sigma1.reshape(-1, 20), z_sigma2.reshape(-1, 20), z_corr.reshape(-1, 20), z_pen_logits.reshape(-1, 3)]

        else:
            batch_gen_strokes = []
            state_point = start_token.squeeze(0)  # [GO] token
            num_steps = sketch_vector.shape[0] + 1
            # batch_init = torch.cat([start_token, sketch_vector], 0)
            attention_plot = []

            for i in range(num_steps):

                att_feature, attention = self.attention_cell.forward(backbone_feature, hidden_cell[0].squeeze(0))
                attention_plot.append(attention.view(batch_size, 1, 8, 8))
                # state_point = batch_init[i, :, :]
                concat_context = torch.cat([att_feature, state_point], 1).unsqueeze(0)  # batch_size x (num_channel + num_embedding)
                _, hidden_cell = self.lstm(concat_context, hidden_cell)
                y_output = self.fc_params(hidden_cell[0].permute(1, 0, 2))


                """ Split the data to get next output <Deterministic Prediction>"""
                z_pen_logits = y_output[:, :, 0:3]
                z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.chunk(y_output[:, :, 3:], 6, 2)
                z_pi = F.softmax(z_pi, dim=-1)
                z_sigma1 = torch.exp(z_sigma1)
                z_sigma2 = torch.exp(z_sigma2)
                z_corr = torch.tanh(z_corr)

                batch_size = z_pi.shape[0]
                z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, \
                z_corr, z_pen_logits = z_pi.reshape(-1, 20), z_mu1.reshape(-1,20), \
                                       z_mu2.reshape(-1, 20), z_sigma1.reshape(-1,20), \
                                       z_sigma2.reshape(-1, 20), z_corr.reshape(-1, 20), z_pen_logits.reshape(-1, 3)

                recons_output = torch.zeros(batch_size, 5).to(device)
                z_pi_idx = z_pi.argmax(dim=-1)
                z_pen_idx = z_pen_logits.argmax(-1)
                recons_output[:, 0] = z_mu1[range(z_mu1.shape[0]), z_pi_idx]
                recons_output[:, 1] = z_mu2[range(z_mu2.shape[0]), z_pi_idx]

                recons_output[range(z_mu1.shape[0]), z_pen_idx + 2] = 1.

                state_point = recons_output.data
                batch_gen_strokes.append(state_point)

            return torch.stack(batch_gen_strokes, dim=1), attention_plot



class AttentionCell2D(nn.Module):

    def __init__(self, hidden_size):
        super(AttentionCell2D, self).__init__()
        self.feature_layers = 512
        self.hidden_dim_de = hidden_size
        self.embedding_size = 256

        self.conv_h = nn.Linear(self.hidden_dim_de, self.embedding_size)
        self.conv_f = nn.Conv2d(self.feature_layers,
                                self.embedding_size, kernel_size=3, padding=1)

        self.conv_att = nn.Linear(self.embedding_size, 1)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, conv_f, h):#conv_f[10,512,6,40],h:[10,512]

        g_em = self.conv_h(h)  #10, 512
        g_em = g_em.unsqueeze(-1).permute(0, 2, 1) #[10, 1, 256]

        x_em = self.conv_f(conv_f) #[10, 256, 8, 25]
        x_em = x_em.view(x_em.shape[0], -1, conv_f.shape[2] * conv_f.shape[3]) #[10, 256, 200]
        x_em = x_em.permute(0, 2, 1) #[10, 200, 256]

        feat = torch.tanh(x_em + g_em) #[10, 200, 256]
        alpha = F.softmax(self.conv_att(feat), dim=1) # [10, 200, 1]
        alpha = alpha.permute(0, 2, 1)  # [10, 1, 200]

        orgfeat_embed = conv_f.view(conv_f.shape[0], -1, conv_f.shape[2] * conv_f.shape[3]) #[10, 512, 200]
        orgfeat_embed = orgfeat_embed.permute(0, 2, 1) #[10, 200,  512]

        att_out = torch.bmm(alpha, orgfeat_embed).squeeze(1) # [50, 1, 64] x [50, 64, 512] -> [50, 1, 512]

        return att_out, alpha



##############################################################################
### CLIP models ###

# https://github.com/openai/CLIP/blob/main/clip/model.py


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    # vision_heads = vision_cfg.width(64) * 32 // vision_cfg.head_width(64) from openclip model.py / RN50.config
    def __init__(self, layers, output_dim, heads=32, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.transform = transforms.Compose([
            transforms.Resize(size=input_resolution, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.CenterCrop(size=(input_resolution, input_resolution)),
            transforms.Lambda(lambda img : img.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        self.trained_layers = []

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)


    def freeze_layers(self):
        self.trained_layers.append('all')

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class ModifiedResNet_with_classification(ModifiedResNet):
    def __init__(self, layers, output_dim, heads=32, input_resolution=224, width=64, num_classes=125):
        super().__init__(layers, output_dim, heads, input_resolution, width)

        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        feature = super().forward(x)
        classes = self.classifier(feature)
        return feature, classes


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

