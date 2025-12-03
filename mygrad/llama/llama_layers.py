"""
LLaMA Model Layers
Attention, MLP, and Decoder block implementations
"""
import numpy as np
import sys; sys.path.append("build")
import bten
from mygrad.engine import AGTensor, no_grad
from agtensor_llama import *  # Import LLaMA-specific operations
from llama_config import LLaMAConfig
from typing import Optional, Tuple

class RMSNorm:
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, is_cuda: bool = True):
        self.hidden_size = hidden_size
        self.eps = eps
        self.is_cuda = is_cuda
        
        # Initialize weight to ones
        weight_np = np.ones((1, hidden_size), dtype=np.float32)
        self.weight = AGTensor(weight_np, requires_grad=True, is_cuda=is_cuda)
    
    def __call__(self, x: AGTensor) -> AGTensor:
        return x.rmsnorm(self.weight, self.eps)
    
    def parameters(self):
        return [self.weight]


class RotaryEmbedding:
    """Rotary Position Embeddings"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
    
    def __call__(self, x: AGTensor, position_offset: int = 0) -> AGTensor:
        """Apply RoPE to input tensor"""
        return x.rope(position_offset=position_offset, head_dim=self.dim, theta=self.base)


class GroupedQueryAttention:
    """
    Grouped-Query Attention (GQA) as used in TinyLLaMA
    
    In GQA, multiple query heads share the same key/value heads.
    This reduces the memory and compute cost of attention.
    """
    
    def __init__(self, config: LLaMAConfig, is_cuda: bool = True):
        self.config = config
        self.is_cuda = is_cuda
        
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        # Linear projections
        # Q: (hidden_size, hidden_size)
        # K, V: (hidden_size, num_kv_heads * head_dim)
        # O: (hidden_size, hidden_size)
        self.q_proj = self._init_linear(self.hidden_size, self.hidden_size)
        self.k_proj = self._init_linear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = self._init_linear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = self._init_linear(self.hidden_size, self.hidden_size)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        self.scale = 1.0 / np.sqrt(self.head_dim)
    
    def _init_linear(self, in_features: int, out_features: int) -> AGTensor:
        """Initialize a linear layer weight matrix"""
        # Xavier/Glorot initialization
        std = np.sqrt(2.0 / (in_features + out_features))
        weight_np = np.random.randn(in_features, out_features).astype(np.float32) * std
        return AGTensor(weight_np, requires_grad=True, is_cuda=self.is_cuda)
    
    def _split_heads(self, x: AGTensor, num_heads: int) -> AGTensor:
        """
        Reshape from (batch, hidden_size) to (batch * num_heads, head_dim)
        
        Note: Since we only support 2D tensors, we'll reshape carefully.
        Input: (batch_size, num_heads * head_dim)
        Output: (batch_size * num_heads, head_dim)
        """
        batch_size = x.shape[0]
        hidden_size = x.shape[1]
        head_dim = hidden_size // num_heads
        
        # We need to reshape and transpose
        # This is tricky with only 2D tensors...
        # For now, we'll work with the flat representation
        # TODO: Implement proper reshape when needed
        return x
    
    def _repeat_kv(self, kv: AGTensor, n_rep: int) -> AGTensor:
        """
        Repeat key/value tensors for grouped-query attention
        Each KV head is repeated n_rep times to match query heads
        """
        if n_rep == 1:
            return kv
        
        # For GQA: we need to repeat each KV head n_rep times
        # Input: (batch, num_kv_heads * head_dim)
        # Output: (batch, num_heads * head_dim)
        batch_size = kv.shape[0]
        kv_size = kv.shape[1]
        head_dim = self.head_dim
        num_kv_heads = kv_size // head_dim
        
        # Manually repeat - this is inefficient but works with our constraints
        # In practice, you'd want a custom CUDA kernel for this
        repeated_np = np.zeros((batch_size, num_kv_heads * n_rep * head_dim), dtype=np.float32)
        kv_np = kv.numpy()
        
        for b in range(batch_size):
            for kv_head in range(num_kv_heads):
                src_start = kv_head * head_dim
                src_end = src_start + head_dim
                src_data = kv_np[b, src_start:src_end]
                
                for rep in range(n_rep):
                    dst_head = kv_head * n_rep + rep
                    dst_start = dst_head * head_dim
                    dst_end = dst_start + head_dim
                    repeated_np[b, dst_start:dst_end] = src_data
        
        return AGTensor(repeated_np, requires_grad=kv.requires_grad, is_cuda=self.is_cuda)
    
    def __call__(
        self,
        hidden_states: AGTensor,
        position_offset: int = 0,
        use_causal_mask: bool = True
    ) -> AGTensor:
        """
        Forward pass of grouped-query attention
        
        Args:
            hidden_states: (batch_size, hidden_size)
            position_offset: Starting position for RoPE (for KV cache)
            use_causal_mask: Whether to use causal masking
        """
        batch_size = hidden_states.shape[0]
        
        # Project to Q, K, V
        query = hidden_states @ self.q_proj  # (batch, hidden_size)
        key = hidden_states @ self.k_proj    # (batch, num_kv_heads * head_dim)
        value = hidden_states @ self.v_proj  # (batch, num_kv_heads * head_dim)
        
        # Apply rotary embeddings to Q and K
        query = self.rotary_emb(query, position_offset)
        key = self.rotary_emb(key, position_offset)
        
        # Repeat K and V for grouped-query attention
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            key = self._repeat_kv(key, n_rep)
            value = self._repeat_kv(value, n_rep)
        
        # Compute attention scores: Q @ K^T
        # query: (batch, hidden_size), key: (batch, hidden_size)
        attn_scores = query @ key.T  # (batch, batch)
        
        # Scale scores
        attn_scores = attn_scores * self.scale
        
        # Apply softmax with causal mask if needed
        attn_probs = attn_scores.softmax(causal=use_causal_mask, seq_offset=position_offset)
        
        # Compute attention output: softmax(QK^T) @ V
        attn_output = attn_probs @ value  # (batch, hidden_size)
        
        # Output projection
        output = attn_output @ self.o_proj
        
        return output
    
    def parameters(self):
        return [self.q_proj, self.k_proj, self.v_proj, self.o_proj]


class MLP:
    """
    Feed-Forward Network with SiLU activation (SwiGLU variant)
    """
    
    def __init__(self, config: LLaMAConfig, is_cuda: bool = True):
        self.config = config
        self.is_cuda = is_cuda
        
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        
        # Gate, Up, and Down projections for SwiGLU
        self.gate_proj = self._init_linear(hidden_size, intermediate_size)
        self.up_proj = self._init_linear(hidden_size, intermediate_size)
        self.down_proj = self._init_linear(intermediate_size, hidden_size)
    
    def _init_linear(self, in_features: int, out_features: int) -> AGTensor:
        """Initialize a linear layer weight matrix"""
        std = np.sqrt(2.0 / (in_features + out_features))
        weight_np = np.random.randn(in_features, out_features).astype(np.float32) * std
        return AGTensor(weight_np, requires_grad=True, is_cuda=self.is_cuda)
    
    def __call__(self, x: AGTensor) -> AGTensor:
        """
        SwiGLU: (SiLU(gate(x)) * up(x)) @ down
        """
        gate_out = x @ self.gate_proj
        gate_out = gate_out.silu()  # Apply SiLU activation
        
        up_out = x @ self.up_proj
        
        # Element-wise multiplication
        intermediate = gate_out * up_out
        
        # Down projection
        output = intermediate @ self.down_proj
        
        return output
    
    def parameters(self):
        return [self.gate_proj, self.up_proj, self.down_proj]


class LLaMADecoderLayer:
    """Single transformer decoder layer"""
    
    def __init__(self, config: LLaMAConfig, is_cuda: bool = True):
        self.config = config
        self.is_cuda = is_cuda
        
        # Pre-attention norm
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            is_cuda=is_cuda
        )
        
        # Self-attention
        self.self_attn = GroupedQueryAttention(config, is_cuda=is_cuda)
        
        # Pre-MLP norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            is_cuda=is_cuda
        )
        
        # Feed-forward network
        self.mlp = MLP(config, is_cuda=is_cuda)
    
    def __call__(
        self,
        hidden_states: AGTensor,
        position_offset: int = 0,
        use_causal_mask: bool = True
    ) -> AGTensor:
        """
        Forward pass with pre-norm and residual connections
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_offset, use_causal_mask)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def parameters(self):
        params = []
        params.extend(self.input_layernorm.parameters())
        params.extend(self.self_attn.parameters())
        params.extend(self.post_attention_layernorm.parameters())
        params.extend(self.mlp.parameters())
        return params
