import torch
import numpy as np
import torch.nn.functional as F

def torch_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2

    z_1 = (norm1 / s1) ** 2
    z_2 = (norm2 / s2) ** 2
    z1_z2 = (norm1 * norm2) / s1s2

    z = z_1 + z_2 - 2 * rho * z1_z2
    neg_rho = 1 - rho ** 2
    result = torch.exp(-z / (2 * neg_rho))
    denom = 2 * np.pi * s1s2 * torch.sqrt(neg_rho)
    return result / denom


def sketch_reconstruction_loss(output, x_input):
    # x_input =
    # Ouput = Predicted 123 parameters from decoder = Batch*Max_seq_len, 20
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits] = output
    [x1_data, x2_data, eos_data, eoc_data, cont_data] = torch.chunk(x_input.reshape(-1, 5), 5, 1)
    pen_data = torch.cat([eos_data, eoc_data, cont_data], 1)
    mask = 1.0 - pen_data[:, 2]  # use training data for this

    result0 = torch_2d_normal(x1_data, x2_data, o_mu1, o_mu2, o_sigma1, o_sigma2,
                                   o_corr)
    epsilon = 1e-6

    result1 = torch.sum(result0 * o_pi, dim=1)  # ? unsqueeae(-1) ??
    result1 = -torch.log(result1 + epsilon)  # avoid log(0)

    if torch.isnan(result1).any():
        print('Catched')

    result2 = F.cross_entropy(o_pen_logits, pen_data.argmax(1), reduction='none')

    result = mask * result1 + mask * result2
    # result = result1 + result2

    return result.mean()


def sketch_reconstruction_loss_withoutMask(output, x_input):
    # x_input =
    # Ouput = Predicted 123 parameters from decoder = Batch*Max_seq_len, 20
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits] = output
    [x1_data, x2_data, eos_data, eoc_data, cont_data] = torch.chunk(x_input.reshape(-1, 5), 5, 1)
    pen_data = torch.cat([eos_data, eoc_data, cont_data], 1)
    #mask = 1.0 - pen_data[:, 2]  # use training data for this

    result0 = torch_2d_normal(x1_data, x2_data, o_mu1, o_mu2, o_sigma1, o_sigma2,
                                   o_corr)
    epsilon = 1e-6

    result1 = torch.sum(result0 * o_pi, dim=1)  # ? unsqueeae(-1) ??
    result1 = -torch.log(result1 + epsilon)  # avoid log(0)

    if torch.isnan(result1).any():
        print('Catched')

    result2 = F.cross_entropy(o_pen_logits, pen_data.argmax(1), reduction='none')

    #result = mask * result1 + mask * result2
    result = result1 + result2

    return result.mean()

def set_learningRate(optimizer, curr_learning_rate):
    for g in optimizer.param_groups:
        g['lr'] = curr_learning_rate
