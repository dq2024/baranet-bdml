"""
PyTorch implementation of TinyLLaMA for benchmarking comparison

This implements the same architecture as our custom CUDA implementation
to ensure fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # x: (batch, seq_len, num_heads, head_dim)
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        
        # Split into first and second half
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotation
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        
        return torch.cat((y1, y2), dim=-1)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with KV cache support"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.rope = RotaryEmbedding(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        # Repeat KV heads for grouped query attention
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)
        
        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1
            )
        scores = scores.masked_fill(attention_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Output projection
        return self.o_proj(attn_output)


class MLP(nn.Module):
    """MLP with SiLU activation (SwiGLU variant)"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate_proj(x) * SiLU(up_proj(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaMADecoderLayer(nn.Module):
    """Single LLaMA decoder layer"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class LLaMAModel(nn.Module):
    """Full LLaMA model"""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        rms_norm_eps: float = 1e-6,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        
        head_dim = hidden_size // num_attention_heads
        
        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            LLaMADecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input_ids: (batch_size,) or (batch_size, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.embed_tokens(input_ids)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final norm
        x = self.norm(x)
        
        # LM head
        logits = self.lm_head(x)
        
        # Return only the last token logits for generation
        return logits[:, -1, :]

    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_tinyllama_pytorch(device: str = 'cuda') -> LLaMAModel:
    """Create TinyLLaMA 1.1B model matching our custom implementation"""
    model = LLaMAModel(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        max_seq_len=2048,
    )
    model = model.to(device)
    model.eval()
    
    print(f"Created PyTorch TinyLLaMA model")
    print(f"Total parameters: {model.count_parameters():,}")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_tinyllama_pytorch()
    
    # Test forward pass
    input_ids = torch.randint(0, 32000, (1,), device='cuda')
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("Model test passed!")