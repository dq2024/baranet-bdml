"""
Extended AGTensor with LLaMA-specific operations
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Union
import sys; sys.path.append("build")
import bten
from contextlib import contextmanager

_grad_enabled = True

def is_grad_enabled():
    return _grad_enabled
    
@contextmanager
def no_grad():
    global _grad_enabled
    old = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = old

# Import base AGTensor class
from agtensor import AGTensor

# Add LLaMA-specific methods to AGTensor
def silu(self):
    """
    SiLU (Swish) activation: y = x * sigmoid(x)
    """
    out_data = self.data.silu()
    out = AGTensor(out_data, (self,), 'silu', requires_grad=self.requires_grad and is_grad_enabled())
    
    if out.requires_grad:
        def _backward():
            grad_input = self.data.silu_back(out.grad)
            if self.grad is None:
                self.grad = grad_input
            else:
                self.grad = self.grad + grad_input
        out._backward = _backward
    
    return out

def rmsnorm(self, weight: AGTensor, eps: float = 1e-6):
    """
    RMS Normalization: y = x / sqrt(mean(x^2) + eps) * weight
    weight shape: (1, hidden_dim)
    """
    out_data = self.data.rmsnorm(weight.data, eps)
    out = AGTensor(out_data, (self, weight), 'rmsnorm', 
                   requires_grad=(self.requires_grad or weight.requires_grad) and is_grad_enabled())
    
    if out.requires_grad:
        def _backward():
            # Create gradient tensor for weight
            grad_weight_data = bten.TensorF(weight.shape[0], weight.shape[1], weight.is_cuda)
            grad_weight_data.fill(0.0)
            
            # Compute gradients
            grad_input = self.data.rmsnorm_back(weight.data, out.grad, grad_weight_data, eps)
            
            if self.requires_grad:
                if self.grad is None:
                    self.grad = grad_input
                else:
                    self.grad = self.grad + grad_input
            
            if weight.requires_grad:
                if weight.grad is None:
                    weight.grad = grad_weight_data
                else:
                    weight.grad = weight.grad + grad_weight_data
        
        out._backward = _backward
    
    return out

def rope(self, position_offset: int = 0, head_dim: int = 64, theta: float = 10000.0):
    """
    Rotary Position Embeddings (RoPE)
    Applies rotary embeddings to query or key tensors
    """
    out_data = self.data.rope(position_offset, head_dim, theta)
    out = AGTensor(out_data, (self,), 'rope', requires_grad=self.requires_grad and is_grad_enabled())
    
    if out.requires_grad:
        def _backward():
            grad_input = out.grad.rope_back(position_offset, head_dim, theta)
            if self.grad is None:
                self.grad = grad_input
            else:
                self.grad = self.grad + grad_input
        out._backward = _backward
    
    return out

def softmax(self, causal: bool = False, seq_offset: int = 0):
    """
    Softmax activation
    If causal=True, applies causal masking (for autoregressive attention)
    """
    out_data = self.data.softmax(causal, seq_offset)
    out = AGTensor(out_data, (self,), 'softmax', requires_grad=self.requires_grad and is_grad_enabled())
    
    if out.requires_grad:
        # Store output for backward pass
        out._softmax_output = out_data
        
        def _backward():
            grad_input = out._softmax_output.softmax_back(out.grad)
            if self.grad is None:
                self.grad = grad_input
            else:
                self.grad = self.grad + grad_input
        out._backward = _backward
    
    return out

# Add methods to AGTensor class
AGTensor.silu = silu
AGTensor.rmsnorm = rmsnorm
AGTensor.rope = rope
AGTensor.softmax = softmax
