"""
Complete TinyLLaMA Model Implementation
"""
import numpy as np
import sys; sys.path.append("build")
import bten
from mygrad.engine import AGTensor, no_grad
from mygrad.agtensor_llama import *
from mygrad.llama.llama_config import LLaMAConfig
from mygrad.llama.llama_layers import LLaMADecoderLayer, RMSNorm
from typing import Optional, List

class LLaMAModel:
    """
    TinyLLaMA Language Model
    """
    
    def __init__(self, config: LLaMAConfig, is_cuda: bool = True):
        self.config = config
        self.is_cuda = is_cuda
        
        # Token embeddings
        # Shape: (vocab_size, hidden_size)
        self.embed_tokens = self._init_embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = [
            LLaMADecoderLayer(config, is_cuda=is_cuda)
            for _ in range(config.num_hidden_layers)
        ]
        
        # Final layer norm
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            is_cuda=is_cuda
        )
        
        # Language model head (tied with embeddings in original LLaMA)
        self.lm_head = self._init_linear(config.hidden_size, config.vocab_size)
        
        print(f"Initialized LLaMA model with {config.num_hidden_layers} layers")
        print(f"Total parameters: {self.count_parameters():,}")
    
    def _init_embedding(self, vocab_size: int, hidden_size: int) -> AGTensor:
        """Initialize embedding matrix"""
        # Small random initialization
        std = 0.02
        emb_np = np.random.randn(vocab_size, hidden_size).astype(np.float32) * std
        return AGTensor(emb_np, requires_grad=True, is_cuda=self.is_cuda)
    
    def _init_linear(self, in_features: int, out_features: int) -> AGTensor:
        """Initialize linear layer"""
        std = np.sqrt(2.0 / (in_features + out_features))
        weight_np = np.random.randn(in_features, out_features).astype(np.float32) * std
        return AGTensor(weight_np, requires_grad=True, is_cuda=self.is_cuda)
    
    def embed(self, input_ids: np.ndarray) -> AGTensor:
        """
        Convert token IDs to embeddings
        
        Args:
            input_ids: (batch_size,) array of token IDs
        
        Returns:
            embeddings: (batch_size, hidden_size)
        """
        batch_size = len(input_ids)
        hidden_size = self.config.hidden_size
        
        # Gather embeddings for each token
        embeddings_np = np.zeros((batch_size, hidden_size), dtype=np.float32)
        embed_matrix = self.embed_tokens.numpy()
        
        for i, token_id in enumerate(input_ids):
            embeddings_np[i] = embed_matrix[token_id]
        
        return AGTensor(embeddings_np, requires_grad=False, is_cuda=self.is_cuda)
    
    def __call__(
        self,
        input_ids: np.ndarray,
        position_offset: int = 0,
        use_causal_mask: bool = True
    ) -> AGTensor:
        """
        Forward pass
        
        Args:
            input_ids: (batch_size,) token IDs
            position_offset: Starting position (for KV cache)
            use_causal_mask: Whether to use causal masking
        
        Returns:
            logits: (batch_size, vocab_size)
        """
        # Embed tokens
        hidden_states = self.embed(input_ids)
        
        # Pass through all layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_offset, use_causal_mask)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Project to vocabulary
        logits = hidden_states @ self.lm_head
        
        return logits
    
    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_causal_mask: bool = True
    ) -> List[int]:
        """
        Autoregressive generation
        
        Args:
            input_ids: (seq_len,) initial token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
            use_causal_mask: Whether to use causal masking
        
        Returns:
            generated_ids: List of generated token IDs
        """
        generated = list(input_ids)
        
        with no_grad():
            for _ in range(max_new_tokens):
                # Get logits for current sequence
                # For simplicity, process entire sequence each time
                # (In practice, you'd use KV cache)
                current_ids = np.array(generated, dtype=np.uint32)
                logits = self(current_ids, position_offset=0, use_causal_mask=use_causal_mask)
                
                # Get logits for last position
                next_token_logits = logits.numpy()[-1]  # (vocab_size,)
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    indices_to_remove = next_token_logits < np.partition(next_token_logits, -top_k)[-top_k]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Convert logits to probabilities
                probs = np.exp(next_token_logits - np.max(next_token_logits))
                probs = probs / np.sum(probs)
                
                # Sample next token
                next_token = np.random.choice(len(probs), p=probs)
                
                # Stop if EOS token
                if next_token == self.config.eos_token_id:
                    break
                
                generated.append(int(next_token))
        
        return generated
    
    def parameters(self):
        """Get all model parameters"""
        params = [self.embed_tokens]
        
        for layer in self.layers:
            params.extend(layer.parameters())
        
        params.extend(self.norm.parameters())
        params.append(self.lm_head)
        
        return params
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        total = 0
        for param in self.parameters():
            total += param.shape[0] * param.shape[1]
        return total
    
    def save_checkpoint(self, path: str):
        """Save model weights"""
        checkpoint = {}
        params = self.parameters()
        
        for i, param in enumerate(params):
            checkpoint[f'param_{i}'] = param.numpy()
        
        np.savez(path, **checkpoint)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model weights"""
        checkpoint = np.load(path)
        params = self.parameters()
        
        for i, param in enumerate(params):
            param_data = checkpoint[f'param_{i}']
            # Copy data back to tensor
            if self.is_cuda:
                temp = bten.TensorF(param.shape[0], param.shape[1], True)
                temp.copy_from_numpy(param_data)
                param.data = temp
            else:
                param.data.copy_from_numpy(param_data)
        
        print(f"Loaded checkpoint from {path}")


def create_tinyllama_model(is_cuda: bool = True) -> LLaMAModel:
    """Create a TinyLLaMA 1.1B model"""
    config = LLaMAConfig()
    return LLaMAModel(config, is_cuda=is_cuda)
