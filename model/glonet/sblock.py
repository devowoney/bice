import torch
from torch import nn
from modules1 import *
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from NO import GF_Block
from NN import *

class FoTF(nn.Module):
    def __init__(self, shape_in, num_interactions=3, n_heads=8, patch_size=[8,8]):
        super(FoTF, self).__init__()
        T, C, H, W = shape_in
        self.n_heads=n_heads
        self.NN = Local_CNN_Branch(in_channels=C, out_channels=C)
        self.NO = GF_Block(
            img_size=[H,W],
            patch_size=patch_size,
            in_channels=C,
            out_channels=C,
            input_frames=T,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            uniform_drop=False,
            drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=None,
            dropcls=0.,
            n_heads=self.n_heads
        )
        self.up = nn.ConvTranspose2d(C, C, kernel_size=3, stride=1, padding=1)
        self.down = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1)
        self.num_interactions = num_interactions

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        gf_features = self.NO(x_raw)
        lc_features = self.NN(x_raw)

        for _ in range(self.num_interactions):
            gf_features_up = self.up(gf_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = gf_features_up + lc_features

            gf_features = self.NO(combined_features)
            lc_features = self.NN(combined_features)

            gf_features_down = self.down(gf_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = gf_features_down + lc_features

            gf_features = self.NO(combined_features)
            lc_features = self.NN(combined_features)

        return gf_features + lc_features

