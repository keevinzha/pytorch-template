# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/14 12:50
@Auth ： keevinzha
@File ：recon_tools.py
@IDE ：PyCharm
"""

import numpy as np
from numpy.fft import fft, fft2, ifft, ifft2, ifftshift, fftshift
import torch


def fft1c(x, axis=-1, norm='ortho'):
    """
    1D centered fft
    """
    if isinstance(x, np.ndarray):
        return fftshift(fft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)
    elif isinstance(x, torch.Tensor):
        return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=axis), dim=axis, norm=norm), dim=axis)


def ifft1c(x, axis=-1, norm='ortho'):
    """
    1D centered ifft
    """
    if isinstance(x, np.ndarray):
        return fftshift(ifft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)
    elif isinstance(x, torch.Tensor):
        return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=axis), dim=axis, norm=norm), dim=axis)


def fft2c(x, axes=(-2, -1), norm='ortho'):
    """
    2D centered fft
    """
    if isinstance(x, np.ndarray):
        return fftshift(fft2(ifftshift(x, axes=axes), axes=axes, norm=norm), axes=axes)
    elif isinstance(x, torch.Tensor):
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=axes), dim=axes, norm=norm), dim=axes)


def ifft2c(x, axes=(-2, -1), norm='ortho'):
    """
    2D centered ifft
    """
    if isinstance(x, np.ndarray):
        return fftshift(ifft2(ifftshift(x, axes=axes), axes=axes, norm=norm), axes=axes)
    elif isinstance(x, torch.Tensor):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=axes), dim=axes, norm=norm), dim=axes)


def sos(x, axis=-1):
    """
    Sum of squares
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.sum(np.abs(x)**2, axis=axis))
    elif isinstance(x, torch.Tensor):
        return torch.sqrt(torch.sum(torch.abs(x)**2, dim=axis))


def real2complex(x):
    """
    Convert double to complex
    """
    if isinstance(x, np.ndarray):
        return x[:, :x.shape[1]//2, ...] + 1j*x[:, x.shape[1]//2:, ...]
    elif isinstance(x, torch.Tensor):
        return x[:, :x.shape[1]//2, ...] + 1j*x[:, x.shape[1]//2:, ...]


def complex2real(x):
    """
    Convert complex to double
    """
    if isinstance(x, np.ndarray):
        return np.concatenate((np.real(x), np.imag(x)), axis=1).astype(np.float32)
    elif isinstance(x, torch.Tensor):
        return torch.concatenate((torch.real(x), torch.imag(x)), dim=1).type(torch.float32)

