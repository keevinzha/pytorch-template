# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/16 18:12
@Auth ： keevinzha
@File ：networks.py
@IDE ：PyCharm
"""


import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class Identity(nn.Module):
    """
    Used for none normalization.
    """
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """
    Return a normalization layer.

    :param norm_type: the name of the normalization layer: batch | instance | none
    :return: normalization layer
    """
    # todo add group norm, layer norm etc.
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = Identity
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """
    Return a learning rate scheduler.

    :param optimizer: the optimizer of the network
    :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions．　
                opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
        # todo replce consin with facebook codes
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    :param net (network): network to be initialized
    :param init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
    :param init_gain (floot): scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialize function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1): # conv and linear
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1: # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func) # apply the initialization function <init_func>


    def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
        """
        Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights.

        :param net (network): the network to be initialized
        :param init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
        :param init_gain (float): scaling factor for normal, xavier and orthogonal.
        :param gpu_ids (int list): which GPUs the network runs on: e.g., 0,1,2

        Return an initialized network.
        """
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        init_weights(net, init_type, init_gain=init_gain)
        return net


