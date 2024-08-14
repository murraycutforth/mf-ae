import argparse
import json
import logging
import typing

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, norm):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x


class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, activation, norm):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.norm = norm(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x


class ConvAutoencoderBaseline(nn.Module):
    """Basic 3D conv autoencoder class, uses strided convolutions to downsample inputs, and transposed convolutions to
    upsample them. The bottleneck can be either flat or not.
    """
    def __init__(self,
                 image_shape: tuple,
                 flat_bottleneck: bool = True,
                 latent_dim: int = 100,
                 activation: nn.Module = nn.SELU(),
                 norm: typing.Type[nn.Module] = nn.InstanceNorm3d,
                 feat_map_sizes: list|tuple = (32, 64, 128, 256),
                 ):
        super().__init__()
        self.flat_bottleneck = flat_bottleneck
        self.latent_dim = latent_dim
        self.activation = activation
        self.norm = norm

        self.encoder_outer = nn.Sequential(
            *[ConvBlockDown(in_channels, out_channels, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm) \
              for in_channels, out_channels in zip([1] + list(feat_map_sizes[:-1]), feat_map_sizes)]
        )

        final_shape = [s // (2 ** len(feat_map_sizes)) for s in image_shape]

        if flat_bottleneck:
            self.bottleneck_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_map_sizes[-1] * np.prod(final_shape), latent_dim),
            )
            self.encoder = nn.Sequential(
                self.encoder_outer,
                self.bottleneck_encoder,
            )
        else:
            self.encoder = self.encoder_outer

        self.decoder_outer = nn.Sequential(
            *[ConvBlockUp(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, activation=activation, norm=norm) \
              for in_channels, out_channels in zip(feat_map_sizes[::-1], list(feat_map_sizes[-2::-1]) + [1])]
        )

        if flat_bottleneck:
            self.bottleneck_decoder = nn.Sequential(
                nn.Linear(latent_dim, feat_map_sizes[-1] * np.prod(final_shape)),
                nn.Unflatten(1, (feat_map_sizes[-1], *final_shape)),
            )
            self.decoder = nn.Sequential(
                self.bottleneck_decoder,
                self.decoder_outer,
            )
        else:
            self.decoder = self.decoder_outer

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x
