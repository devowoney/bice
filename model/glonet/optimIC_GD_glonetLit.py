import hydra
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Dict, Any, Tuple
from hydra.utils import instantiate
import xarray as xr
import numpy as np
import os

# from blocks import *
# from NN import *

import sys
from pathlib import Path
from modelp2 import Glonet 

import logging
log = logging.getLogger(__name__)

import torch.utils.checkpoint as checkpoint_util




class GlonetGradientCheckpointing(Glonet) :
    def __init__(self, shape_in, hid_S=256, hid_T=128, N_S=2, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super().__init__(shape_in, hid_S, hid_T, N_S, N_T, incep_ker, groups)
    
    def forward(self, input_st_tensors):
        B, T, C, H, W = input_st_tensors.shape
        
        # Use gradient checkpointing for memory-intensive operations
        def compute_spatial_features(x):
            skip_feature = self.jump(x)
            spatial_feature = self.space(x)
            return skip_feature, spatial_feature
        
        def compute_latent_features(spatial_feature):
            spatial_feature = spatial_feature.reshape(-1, C, H, W)
            return self.maps(spatial_feature)
        
        def compute_temporal_features(spatial_embed):
            return self.dynamics(spatial_embed)
        
        def compute_predictions(spatialtemporal_embed, spatial_skip_feature):
            return self.mapsback(spatialtemporal_embed, spatial_skip_feature)
        
        # Apply gradient checkpointing to reduce memory usage
        skip_feature, spatial_feature = checkpoint_util.checkpoint(
            compute_spatial_features, input_st_tensors, use_reentrant=False
        )
        
        spatial_embed, spatial_skip_feature = checkpoint_util.checkpoint(
            compute_latent_features, spatial_feature, use_reentrant=False
        )
        
        # Reshape for temporal processing
        _, C_, H_, W_ = spatial_embed.shape
        spatial_embed = spatial_embed.view(B, T, C_, H_, W_)
        
        spatialtemporal_embed = checkpoint_util.checkpoint(
            compute_temporal_features, spatial_embed, use_reentrant=False
        )
        
        # Reshape back
        spatialtemporal_embed = spatialtemporal_embed.reshape(B*T, C_, H_, W_)
        
        predictions = checkpoint_util.checkpoint(
            compute_predictions, spatialtemporal_embed, spatial_skip_feature, use_reentrant=False
        )
        
        # Final computation
        predictions = 0.05 * predictions.reshape(B, T, C, H, W) + skip_feature
        
        return predictions



class GlonetGradientInitialCondition(pl.LightningModule) :
    
    def __init__(self, 
                 cfg = None) :
        super().__init__()
        
        # Store config for optimizer parameters
        self.cfg = cfg
        self.model_cfg = self.cfg.model
        

        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        
        # Initial conditions to be set on first batch
        self.init_input1 = None
        self.init_input2 = None
        self.init_input3 = None
        self.target1 = None
        self.target2 = None
        self.target3 = None
        self._initialized = False
        self.coords = None
        self.dataset = None  # Reference to the dataset instance
        
        # Loss function
        self.loss_fn = hydra.utils.instantiate(self.cfg.training.loss)
        
        # Load pytorch checkpoint
        log.info("Loading checkpoint...")
        self.checkpoint_1 = torch.load(self.model_cfg.checkpoint_paths.part_1, map_location=torch.device('cuda'))
        self.checkpoint_2 = torch.load(self.model_cfg.checkpoint_paths.part_2, map_location=torch.device('cuda'))
        self.checkpoint_3 = torch.load(self.model_cfg.checkpoint_paths.part_3, map_location=torch.device('cuda'))

        # Create new gradient checkpointing model instance
        if self.cfg.data.computing.enable_patching :
            patch_size = self.model_cfg.patch_size
            self.gradcheckp_model_1 = GlonetGradientCheckpointing(shape_in=(2, 5, patch_size[0], patch_size[1]))
            self.gradcheckp_model_2 = GlonetGradientCheckpointing(shape_in=(2, 40, patch_size[0], patch_size[1]))            
            self.gradcheckp_model_3 = GlonetGradientCheckpointing(shape_in=(2, 40, patch_size[0], patch_size[1]))
        else :
            self.gradcheckp_model_1 = GlonetGradientCheckpointing(shape_in=(2, 5, 672, 1440))
            self.gradcheckp_model_2 = GlonetGradientCheckpointing(shape_in=(2, 40, 672, 1440))            
            self.gradcheckp_model_3 = GlonetGradientCheckpointing(shape_in=(2, 40, 672, 1440))
            
        # Load the same weights from the original model
        self.gradcheckp_model_1.load_state_dict(self.checkpoint_1['model_state_dict'])
        self.gradcheckp_model_1.train()
        self.gradcheckp_model_2.load_state_dict(self.checkpoint_2['model_state_dict'])
        self.gradcheckp_model_2.train()
        self.gradcheckp_model_3.load_state_dict(self.checkpoint_3['model_state_dict'])
        self.gradcheckp_model_3.train()
        
        # Freeze model parameters — we only optimize the initial condition
        for param in self.gradcheckp_model_1.parameters():
            param.requires_grad = False
        for param in self.gradcheckp_model_2.parameters():
            param.requires_grad = False
        for param in self.gradcheckp_model_3.parameters():
            param.requires_grad = False
            
        log.info(f"Frozen {sum(1 for _ in self.gradcheckp_model_1.parameters()) * 3} parameters for models part1, 2 and 3")


    def forward(self, 
                x1 : torch.Tensor = None,
                x2 : torch.Tensor = None,
                x3 : torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] :
        """Forward pass through the saved model with forecast window iterations."""
        
        if x1 is not None and not x1.requires_grad:
            log.warning(f"Input tensor does not require grad: {x1.requires_grad}")
        
        try:
            # !!! Not explicit device handling : model is hardcoded to cuda in JIT
            with torch.enable_grad():
                
                return self.gradcheckp_model_1(x1), self.gradcheckp_model_2(x2), self.gradcheckp_model_3(x3)
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[Error]: CUDA device issue in JIT model: {e}")
                print(f"[Error]: Your JIT model was saved with hardcoded CUDA devices.")
                print(f"[Error]: You need to retrain/resave the model without hardcoded devices,")
                print(f"[Error]: or run on a machine with CUDA available.")
                raise RuntimeError("JIT model has hardcoded CUDA device but CUDA is not available") from e
            else:
                raise e

    def step(self, 
             batch : tuple) -> float :
        """Single optimization step."""
    
        y1_hat, y2_hat, y3_hat = self.forward(self.init_input1, 
                                              self.init_input2, 
                                              self.init_input3)
        
        loss = (self.loss_fn(y1_hat, self.target1) + 
                self.loss_fn(y2_hat, self.target2) + 
                self.loss_fn(y3_hat, self.target3))
        
        return loss

    def training_step(self, batch) -> float :
        """Define training step - initialize on first call."""
        
        # Store reference dataset
        if not self._initialized:
            # Store reference to dataset via trainer.datamodule
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
                self.dataset = self.trainer.datamodule.train_dataset
                log.info(f"Dataset reference stored in Lightning module")
                
        # Initialize on first batch            
            x1, x2, x3, y1, y2, y3 = batch
            self.init_input1 = nn.Parameter(x1.detach().clone(), requires_grad=True)
            self.init_input2 = nn.Parameter(x2.detach().clone(), requires_grad=True)
            self.init_input3 = nn.Parameter(x3.detach().clone(), requires_grad=True)
            self.target1 = y1
            self.target2 = y2
            self.target3 = y3
            self._initialized = True
            
            # Re-initialize optimizer with the new parameters
            self.trainer.strategy.setup_optimizers(self.trainer)
        
        # For JIT models, we might need to explicitly enable gradients
        with torch.enable_grad():
            train_loss = self.step(batch)

            # Log metrics
            self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
            
            return train_loss
    
    def on_train_epoch_end(self) -> None :
        """Called at the end of each training epoch to log gradient norms."""
        
        if self._initialized :
            # Calculate gradient norms for each initial condition
            grad_norm_1 = self.init_input1.grad.norm().item() if self.init_input1.grad is not None else 0.0
            grad_norm_2 = self.init_input2.grad.norm().item() if self.init_input2.grad is not None else 0.0
            grad_norm_3 = self.init_input3.grad.norm().item() if self.init_input3.grad is not None else 0.0
            
            # Log gradient norms
            self.log('grad_norm/input1', grad_norm_1, on_epoch=True, prog_bar=False)
            self.log('grad_norm/input2', grad_norm_2, on_epoch=True, prog_bar=False)
            self.log('grad_norm/input3', grad_norm_3, on_epoch=True, prog_bar=False)
            
            log.info(f"Epoch {self.current_epoch} - Gradient Norms: "
                    f"input1={grad_norm_1:.6f}, input2={grad_norm_2:.6f}, "
                    f"input3={grad_norm_3:.6f}")
        
    def on_train_end(self) -> None :
        """Save optimized initial condition at the end of the training as netCDF"""
        
        if self._initialized :
            # Convert tensors to numpy arrays and squeeze batch dimension
            # Shape: [batch, time, channels, height, width] -> [time, channels, height, width]
            init1_np = self.init_input1.detach().cpu().numpy().squeeze(0)
            init2_np = self.init_input2.detach().cpu().numpy().squeeze(0)
            init3_np = self.init_input3.detach().cpu().numpy().squeeze(0)
            
            # Get dimensions: [time, channels, height, width]
            time, channel, height, width = init1_np.shape
            time2, channel2, height2, width2 = init2_np.shape
            
            self.coords = self.dataset.current_coords
            
            # Extract coordinates from stored batch coordinates
            if self.coords is not None:
                # Get coordinates from the batch
                # Check if coordinates are numpy arrays or have multiple dimensions (batched)
                time_coords = self.coords['time']
                lat_coords = self.coords['lat']
                lon_coords = self.coords['lon']
            else:
                # Fallback: create simple index-based coordinates
                log.warning(f"No coordinates found in the batch; using default indices instead.")
                time_coords = np.arange(time)
                lat_coords = np.arange(height)
                lon_coords = np.arange(width)
            
            # Create channel coordinates
            ch_coords_1 = np.arange(channel)
            ch_coords_2 = np.arange(channel2)
            
            # Create coordinate dictionaries
            coords1 = {
                'time': time_coords,
                'ch': ch_coords_1,
                'lat': lat_coords,
                'lon': lon_coords
            }
            coords2 = {
                'time': time_coords,
                'ch': ch_coords_2,
                'lat': lat_coords,
                'lon': lon_coords
            }
            
            # Create separate xarray Datasets for each initial condition
            ds1 = xr.Dataset({
                'data': (['time', 'ch', 'lat', 'lon'], init1_np)
            }, coords=coords1)
            
            ds2 = xr.Dataset({
                'data': (['time', 'ch', 'lat', 'lon'], init2_np)
            }, coords=coords2)
            
            ds3 = xr.Dataset({
                'data': (['time', 'ch', 'lat', 'lon'], init3_np)
            }, coords=coords2)
            
            # Add metadata attributes to each dataset
            for i, ds in enumerate([ds1, ds2, ds3], start=1):
                ds.attrs['description'] = f'Optimized initial condition part {i} from gradient descent'
                ds.attrs['creation_date'] = str(np.datetime64('now'))
                ds.attrs['model_config'] = str(self.cfg.model)
                ds.attrs['training_epochs'] = self.current_epoch
                ds['data'].attrs['long_name'] = f'Optimized initial condition part {i}'
                ds['data'].attrs['units'] = 'model_units'
            
            # Prepare save paths
            save_path = self.cfg.model.get('output_path', 'optimized_ic')
            if save_path.endswith('.pt'):
                base_path = save_path.replace('.pt', '')
            elif save_path.endswith('.nc'):
                base_path = save_path.replace('.nc', '')
            elif save_path.endswith('/'):
                base_path = save_path.replace('/', '')
            else:
                base_path = save_path
            
            # Create output directory if it doesn't exist
            os.makedirs(base_path, exist_ok=True)
            
            # Save each dataset to separate netCDF files with compression
            encoding = {'data': {'zlib': True, 'complevel': 4}}
            
            path1 = f"{base_path}/optimal_input1.nc"
            path2 = f"{base_path}/optimal_input2.nc"
            path3 = f"{base_path}/optimal_input3.nc"
            
            ds1.to_netcdf(path1, encoding=encoding)
            ds2.to_netcdf(path2, encoding=encoding)
            ds3.to_netcdf(path3, encoding=encoding)
            
            log.info(f"Saved optimized initial conditions (netCDF format):")
            log.info(f"  Optimized input 1: {path1} - shape {init1_np.shape}")
            log.info(f"  Optimized input 2: {path2} - shape {init2_np.shape}")
            log.info(f"  Optimized input 3: {path3} - shape {init3_np.shape}")
            
    def on_after_backward(self):
        # mask ic gradients so land stays zero
        with torch.no_grad():
            self.init_input1.grad *= self.dataset.ocean_mask_1.to(self.device)
            self.init_input2.grad *= self.dataset.ocean_mask_2.to(self.device)
            self.init_input3.grad *= self.dataset.ocean_mask_3.to(self.device)
        
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler"""
    
        optimizer = instantiate(self.cfg.training.optimizer, 
                                params=self.parameters())
        
        # Check if scheduler is configured
        if 'scheduler' in self.cfg.training:
            scheduler = instantiate(self.cfg.training.scheduler, 
                                    optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
            
        return optimizer
    
