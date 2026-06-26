import gc
import torch.distributed as dist
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#import deepspeed
import time
from multiprocessing import current_process
import torch
from torch import nn
#from modules import *
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from sblock import *
from dblock import *
from utils import *
class Glonet(nn.Module):
    def __init__(self, shape_in, hid_S=256, hid_T=128, N_S=2, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(Glonet, self).__init__()
        T, C, H, W = shape_in
        self.H1 = int(H / 2 ** (N_S / 2))  if H % 3 == 0 else int(H / 2 ** (N_S / 2))
        self.W1 = int(W / 2 ** (N_S / 2))
        print(self.W1)
        self.jump = ConvolutionalNetwork.residual(shape_in=shape_in).to('cuda:0')
        self.space = FoTF(shape_in=shape_in,num_interactions=3, n_heads=4, patch_size=[16,16]).to('cuda:0')
        self.maps = Encoder(C, hid_S, N_S).to('cuda:0')
        self.dynamics = TeDev(T*hid_S, hid_T, N_T, self.H1, self.W1, incep_ker, groups).to('cuda:0')
        self.mapsback = Decoder(hid_S, C, N_S).to('cuda:0')

        # Ensure all parameters are contiguous
        for param in self.parameters():
            param.data = param.data.contiguous()
            if param.grad is not None:
                param.grad = param.grad.contiguous()

    def forward(self, input_st_tensors):
        B, T, C, H, W = input_st_tensors.shape
        skip_feature = self.jump(input_st_tensors.to('cuda:0')).contiguous()
        spatial_feature = self.space(input_st_tensors.to('cuda:0')).contiguous()
        skip_feature=skip_feature.detach()
        spatial_feature=spatial_feature.detach()
        del input_st_tensors
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        spatial_feature = spatial_feature.reshape(-1, C, H, W).contiguous()
        #spatial_embed, spatial_skip_feature=deepspeed.checkpointing.checkpoint(self.latent_projection,spatial_feature)
        spatial_embed, spatial_skip_feature = self.maps(spatial_feature.to('cuda:0'))
        del spatial_feature
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        spatial_embed=spatial_embed.detach()
        spatial_skip_feature=spatial_skip_feature.detach()
        spatial_embed = spatial_embed.contiguous()
        spatial_skip_feature = spatial_skip_feature.contiguous()
        _, C_, H_, W_ = spatial_embed.shape
        spatial_embed = spatial_embed.view(B, T, C_, H_, W_).contiguous()
        spatialtemporal_embed = self.dynamics(spatial_embed.to('cuda:0')).contiguous()
        spatialtemporal_embed = spatialtemporal_embed.detach()
        #spatialtemporal_embed=deepspeed.checkpointing.checkpoint(self.TeDev_block,spatial_embed)
        spatialtemporal_embed = spatialtemporal_embed.reshape(B*T, C_, H_, W_).contiguous()
        del spatial_embed
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        #predictions=deepspeed.checkpointing.checkpoint(self.dec, spatialtemporal_embed, spatial_skip_feature)
        predictions = self.mapsback(spatialtemporal_embed.to('cuda:0'), spatial_skip_feature.to('cuda:0')).contiguous()
        del spatial_skip_feature, spatialtemporal_embed
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        predictions = predictions.detach()
        predictions = 0.05 * predictions.reshape(B, T, C, H, W).contiguous() + skip_feature.to('cuda:0')
        del skip_feature
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        return predictions.contiguous()


