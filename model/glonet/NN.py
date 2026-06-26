import torch
from torch import nn
from modules1 import *
import torch.nn.functional as F
import numpy as np
import torch.optim as optimizer


class Local_CNN_Branch(nn.Module):
    def __init__(self, in_channels = 2, out_channels = 2):
        super(Local_CNN_Branch, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.upconv = nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = self.upconv(x)
        x = x.view(B, T, C, x.shape[2], x.shape[3])
        return x

