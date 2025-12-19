"""
LLaMA Model Configuration
Matches TinyLLaMA 1.1B architecture
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLaMAConfig:
    """Configuration for TinyLLaMA model"""
    
    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632  # FFN hidden dimension
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4  # Grouped-query attention
    
    # Context and embedding
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # Derived dimensions
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_key_value_groups(self) -> int:
        """Number of query heads per key/value head in GQA"""
        return self.num_attention_heads // self.num_key_value_heads
    
    # Inference settings
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
    
    @classmethod
    def from_pretrained(cls, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Load config from HuggingFace model"""
        if "TinyLlama-1.1B" in model_name:
            return cls()  # Use default TinyLLaMA config
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'max_position_embeddings': self.max_position_embeddings,
            'rms_norm_eps': self.rms_norm_eps,
            'rope_theta': self.rope_theta,
        }

# Preset configs
TINYLLAMA_1_1B_CONFIG = LLaMAConfig()

def get_config(name: str = "tinyllama-1.1b") -> LLaMAConfig:
    """Get a preset configuration"""
    configs = {
        "tinyllama-1.1b": TINYLLAMA_1_1B_CONFIG,
        "tinyllama": TINYLLAMA_1_1B_CONFIG,
    }
    
    if name.lower() in configs:
        return configs[name.lower()]
    else:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")
