"""
Memory optimization utilities for LLaMA implementation
Includes buffer reuse, gradient checkpointing, and memory-efficient operations
"""
from __future__ import annotations
from typing import Dict, Optional, List, Tuple
import sys
sys.path.append("build")
import bten
import numpy as np
from collections import deque

class BufferPool:
    """
    Reusable buffer pool to avoid repeated allocations
    """
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pools: Dict[Tuple[int, int, bool], deque] = {}
    
    def get(self, h: int, w: int, is_cuda: bool = True) -> bten.TensorF:
        """Get a buffer from the pool or create a new one"""
        key = (h, w, is_cuda)
        
        if key not in self.pools:
            self.pools[key] = deque()
        
        pool = self.pools[key]
        
        if pool:
            return pool.popleft()
        else:
            return bten.TensorF(h, w, is_cuda)
    
    def put(self, tensor: bten.TensorF):
        """Return a buffer to the pool"""
        key = (tensor.shape[0], tensor.shape[1], tensor.is_cuda)
        
        if key not in self.pools:
            self.pools[key] = deque()
        
        pool = self.pools[key]
        
        if len(pool) < self.max_size:
            pool.append(tensor)
    
    def clear(self):
        """Clear all buffers from the pool"""
        self.pools.clear()

# Global buffer pool instance
_buffer_pool = BufferPool()

def get_buffer(h: int, w: int, is_cuda: bool = True) -> bten.TensorF:
    """Get a buffer from the global pool"""
    return _buffer_pool.get(h, w, is_cuda)

def return_buffer(tensor: bten.TensorF):
    """Return a buffer to the global pool"""
    _buffer_pool.put(tensor)

def clear_buffer_pool():
    """Clear the global buffer pool"""
    _buffer_pool.clear()


class GradientCheckpoint:
    """
    Gradient checkpointing to reduce memory usage during training
    Only stores activations at checkpointed layers, recomputes others during backward
    """
    def __init__(self, checkpoint_every: int = 4):
        self.checkpoint_every = checkpoint_every
        self.checkpoints: List[dict] = []
        self.enabled = False
    
    def enable(self):
        self.enabled = True
        self.checkpoints.clear()
    
    def disable(self):
        self.enabled = False
        self.checkpoints.clear()
    
    def should_checkpoint(self, layer_idx: int) -> bool:
        """Determine if this layer should be checkpointed"""
        if not self.enabled:
            return False
        return layer_idx % self.checkpoint_every == 0
    
    def save_checkpoint(self, layer_idx: int, activations: dict):
        """Save activations for a checkpointed layer"""
        if self.should_checkpoint(layer_idx):
            self.checkpoints.append({
                'layer_idx': layer_idx,
                'activations': activations
            })
    
    def get_checkpoint(self, layer_idx: int) -> Optional[dict]:
        """Get saved activations for a checkpointed layer"""
        for checkpoint in self.checkpoints:
            if checkpoint['layer_idx'] == layer_idx:
                return checkpoint['activations']
        return None

# Global gradient checkpoint instance
_gradient_checkpoint = GradientCheckpoint()

def enable_gradient_checkpointing(checkpoint_every: int = 4):
    """Enable gradient checkpointing"""
    _gradient_checkpoint.checkpoint_every = checkpoint_every
    _gradient_checkpoint.enable()

def disable_gradient_checkpointing():
    """Disable gradient checkpointing"""
    _gradient_checkpoint.disable()

def is_checkpointed(layer_idx: int) -> bool:
    """Check if a layer is checkpointed"""
    return _gradient_checkpoint.should_checkpoint(layer_idx)

