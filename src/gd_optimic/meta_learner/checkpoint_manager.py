"""
Checkpoint management for meta-learner weights.

Provides:
    - MetaLearnerCheckpointManager: Save/load meta-learner weights
    - Support for training, fine-tuning, and inference modes
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from .network_s import UNetMetaGrad2D

logger = logging.getLogger(__name__)


class MetaLearnerCheckpointManager:
    """
    Manage meta-learner checkpoint saving and loading.
    
    Supports three modes:
    - 'training': Train meta-learner from scratch
    - 'fine_tune': Load pre-trained weights and fine-tune
    - 'inference': Load pre-trained weights, freeze meta-learner
    """
    
    def __init__(self, checkpoint_dir: Path, device: str = "cuda"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save/load meta-learner checkpoints
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track meta-learner checkpoint location
        self.meta_learner_dir = self.checkpoint_dir / "meta_learner"
        self.meta_learner_dir.mkdir(parents=True, exist_ok=True)
    
    def save_meta_learner(
        self,
        network_s: UNetMetaGrad2D,
        meta_optimizer,
        iteration: int,
        meta_loss: float,
        meta_loss_history: list = None,
    ) -> Path:
        """
        Save meta-learner checkpoint.
        
        Args:
            network_s: UNetMetaGrad2D network to save
            meta_optimizer: Meta-optimizer (Adam) for meta-learner parameters
            iteration: Current iteration (for naming)
            meta_loss: Current meta-loss value
            meta_loss_history: Optional history of meta-losses
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.meta_learner_dir / f"meta_learner_iter{iteration}.pt"
        
        checkpoint_data = {
            "iteration": iteration,
            "network_s_state": network_s.state_dict(),
            "meta_optimizer_state": meta_optimizer.state_dict(),
            "meta_loss": meta_loss,
            "meta_loss_history": meta_loss_history or [],
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved meta-learner checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_meta_learner(
        self,
        network_s: UNetMetaGrad2D,
        meta_optimizer,
        checkpoint_path: Optional[Path] = None,
        iteration: Optional[int] = None,
    ) -> Dict:
        """
        Load meta-learner checkpoint.
        
        Args:
            network_s: UNetMetaGrad2D network to load into
            meta_optimizer: Meta-optimizer to load state into
            checkpoint_path: Explicit path to checkpoint (if None, use latest)
            iteration: Load specific iteration (if checkpoint_path is None)
            
        Returns:
            Dictionary with checkpoint metadata:
                - iteration: Iteration number
                - meta_loss: Meta-loss value at checkpoint
                - meta_loss_history: History of meta-losses
        """
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self._find_latest_checkpoint(iteration)
        
        if checkpoint_path is None:
            logger.warning("No meta-learner checkpoint found. Starting from scratch.")
            return {"iteration": 0, "meta_loss": float("inf"), "meta_loss_history": []}
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}. Starting from scratch.")
            return {"iteration": 0, "meta_loss": float("inf"), "meta_loss_history": []}
        
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        
        # Load network weights
        network_s.load_state_dict(checkpoint_data["network_s_state"])
        
        # Load optimizer state (if available)
        if "meta_optimizer_state" in checkpoint_data:
            meta_optimizer.load_state_dict(checkpoint_data["meta_optimizer_state"])
        
        logger.info(f"Loaded meta-learner checkpoint: {checkpoint_path}")
        logger.info(f"  Iteration: {checkpoint_data.get('iteration', 'unknown')}")
        logger.info(f"  Meta-loss: {checkpoint_data.get('meta_loss', 'unknown'):.6f}")
        
        return {
            "iteration": checkpoint_data.get("iteration", 0),
            "meta_loss": checkpoint_data.get("meta_loss", float("inf")),
            "meta_loss_history": checkpoint_data.get("meta_loss_history", []),
        }
    
    def load_latest_meta_learner(
        self,
        network_s: UNetMetaGrad2D,
        meta_optimizer,
    ) -> Dict:
        """
        Load latest meta-learner checkpoint from directory.
        
        Args:
            network_s: UNetMetaGrad2D network to load into
            meta_optimizer: Meta-optimizer to load state into
            
        Returns:
            Dictionary with checkpoint metadata
        """
        return self.load_meta_learner(network_s, meta_optimizer, checkpoint_path=None, iteration=None)
    
    def _find_latest_checkpoint(self, iteration: Optional[int] = None) -> Optional[Path]:
        """
        Find latest meta-learner checkpoint.
        
        Args:
            iteration: Specific iteration to load (if None, find latest)
            
        Returns:
            Path to checkpoint, or None if no checkpoint found
        """
        checkpoints = list(self.meta_learner_dir.glob("meta_learner_iter*.pt"))
        
        if not checkpoints:
            return None
        
        if iteration is not None:
            # Load specific iteration
            checkpoint_path = self.meta_learner_dir / f"meta_learner_iter{iteration}.pt"
            if checkpoint_path.exists():
                return checkpoint_path
            else:
                logger.warning(f"Checkpoint for iteration {iteration} not found.")
                return None
        
        # Return latest checkpoint by iteration number
        def get_iteration_from_path(path: Path) -> int:
            """Extract iteration number from checkpoint filename."""
            return int(path.stem.split("iter")[-1])
        
        latest_checkpoint = max(checkpoints, key=get_iteration_from_path)
        return latest_checkpoint
    
    def freeze_meta_learner(self, network_s: UNetMetaGrad2D):
        """
        Freeze all meta-learner parameters (for inference mode).
        
        Args:
            network_s: UNetMetaGrad2D network to freeze
        """
        for param in network_s.parameters():
            param.requires_grad = False
        logger.info("Meta-learner parameters frozen for inference mode.")
    
    def unfreeze_meta_learner(self, network_s: UNetMetaGrad2D):
        """
        Unfreeze all meta-learner parameters (for training/fine-tuning mode).
        
        Args:
            network_s: UNetMetaGrad2D network to unfreeze
        """
        for param in network_s.parameters():
            param.requires_grad = True
        logger.info("Meta-learner parameters unfrozen for training/fine-tuning mode.")
    
    def get_checkpoint_info(self) -> Dict:
        """
        Get information about available meta-learner checkpoints.
        
        Returns:
            Dictionary with checkpoint information
        """
        checkpoints = sorted(
            self.meta_learner_dir.glob("meta_learner_iter*.pt"),
            key=lambda p: int(p.stem.split("iter")[-1])
        )
        
        info = {
            "total_checkpoints": len(checkpoints),
            "checkpoint_dir": str(self.meta_learner_dir),
            "checkpoints": []
        }
        
        for checkpoint_path in checkpoints:
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            info["checkpoints"].append({
                "path": str(checkpoint_path),
                "iteration": checkpoint_data.get("iteration"),
                "meta_loss": checkpoint_data.get("meta_loss"),
            })
        
        return info
