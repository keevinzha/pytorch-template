# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/22 09:17
@Auth ： keevinzha
@File ：odls3d_model.py
@IDE ：PyCharm
"""

import torch
import torch.nn as nn

from timm.models.registry import register_model

from .base_model import BaseModel
from util.recon_tools import ifft1c, complex2real, real2complex, fft1c


class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels,
                                 eps=0.001,
                                 momentum=0.1,
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class SoftThresholding(nn.Module):
    def __init__(self, lamb):
        super(SoftThresholding, self).__init__()
        self.lamb = lamb

    def forward(self, x):
        return torch.mul(torch.sign(x), nn.functional.relu(torch.abs(x) - self.lamb))


class LowRankBlock(nn.Module):
    def __init__(self, ncoil):
        super().__init__()
        self.low_rank_block = nn.Sequential(
            Conv3d(2 * ncoil, 48, (1, 3, 3), 1, padding='same'),
            Conv3d(48, 48, (1, 3, 3), 1, padding='same'),
            Conv3d(48, 48, (1, 3, 3), 1, padding='same'),
            Conv3d(48, 48, (1, 3, 3), 1, padding='same'),
            Conv3d(48, 48, (1, 3, 3), 1, padding='same'),
            nn.Conv3d(48, 2 * ncoil, (1, 3, 3), 1, padding='same')
        )

    def forward(self, x):
        residual = x
        x = complex2real(x)
        x = self.low_rank_block(x)
        x = real2complex(x)
        out = x + residual
        return out


class SparseBlock(nn.Module):
    def __init__(self, ncoil):
        super().__init__()
        self.sparse_block = nn.Sequential(
            Conv3d(2 * ncoil, 48, (1, 3, 3), 1, padding='same'),
            Conv3d(48, 48, (1, 3, 3), 1, padding='same'),
            nn.Conv3d(48, 48, (1, 3, 3), 1, padding='same'),
            SoftThresholding(0.001),
            Conv3d(48, 48, (1, 3, 3), 1, padding='same'),
            Conv3d(48, 48, (1, 3, 3), 1, padding='same'),
            nn.Conv3d(48, 2 * ncoil, (1, 3, 3), 1, padding='same')
        )

    def forward(self, x):
        x = ifft1c(x, 2)
        residual = x
        x = complex2real(x)
        x = self.sparse_block(x)
        x = real2complex(x)
        x = fft1c(x, 2)
        out = x + residual
        return out


class ODLS3dBlock(nn.Module):
    def __init__(self, ncoil):
        super(ODLS3dBlock, self).__init__()
        self.lowrank_block = LowRankBlock(ncoil)
        self.sparse_block = SparseBlock(ncoil)

        self.lamb_sparse = torch.view_as_real(nn.Parameter(torch.tensor(1.0, dtype=torch.complex64)))
        self.lamb_lowrank = torch.view_as_real(nn.Parameter(torch.tensor(1.0, dtype=torch.complex64)))
        self.lamb_dc = torch.view_as_real(nn.Parameter(torch.tensor(1.0, dtype=torch.complex64)))

    def forward(self, iterate_input, mask_multicoil, original_input):
        lamb_sparse = torch.view_as_complex(self.lamb_sparse)
        lamb_lowrank = torch.view_as_complex(self.lamb_lowrank)
        lamb_dc = torch.view_as_complex(self.lamb_dc)
        out_lowrank = self.lowrank_block(iterate_input)
        out_sparse = self.sparse_block(iterate_input)
        out_combine = lamb_lowrank * out_lowrank + lamb_sparse * out_sparse
        lamb_factor = lamb_lowrank + lamb_sparse
        k_sample = mask_multicoil * (
            torch.divide((lamb_dc * original_input + out_combine), (lamb_dc + lamb_factor)))
        k_unsample = (1 - mask_multicoil) * (torch.divide(out_combine, lamb_factor))
        out_dc = k_sample + k_unsample
        return out_dc


class ODLS3dModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--phase_number', type=int, default=10, help='phase number')
        parser.add_argument('--ncoil', type=int, default=8, help='number of coils')
        return parser

    def __init__(self, opt):
        super(ODLS3dModel, self).__init__()
        self.phase_number = opt.phase_number
        self.ncoil = opt.ncoil
        layers = []
        for i in range(self.phase_number):
            layers.append(ODLS3dBlock(self.ncoil))
        self.layers = nn.ModuleList(layers)

    def forward(self, original_input, mask_multicoil):
        outputs = []
        out_k = original_input
        for i in range(self.phase_number):
            out_k = self.layers[i](out_k, mask_multicoil, original_input)
            outputs.append(out_k)
        return outputs








