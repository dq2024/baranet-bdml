"""
Weight Loading Utilities
Convert HuggingFace LLaMA weights to our format
"""
import numpy as np
import torch
from typing import Dict, Any
import sys
sys.path.append("build")

from llama_model import LLaMAModel
from llama_config import LLaMAConfig

def load_huggingface_weights(
    model: LLaMAModel,
    hf_model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda"
) -> None:
    """
    Load pre-trained weights from HuggingFace model
    
    Args:
        model: Our LLaMAModel instance
        hf_model_path: HuggingFace model identifier
        device: Device to load HF model on
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")
    
    print(f"Loading HuggingFace model from {hf_model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float32,
        device_map=device
    )
    hf_state_dict = hf_model.state_dict()
    
    print(f"Converting {len(hf_state_dict)} parameters...")
    
    # Map HuggingFace parameter names to our model
    weight_map = _create_weight_mapping(model.config)
    
    for hf_key, our_param in weight_map.items():
        if hf_key in hf_state_dict:
            hf_weight = hf_state_dict[hf_key].cpu().numpy()
            
            # Handle shape mismatches (transpose if needed)
            if hf_weight.shape != (our_param.shape[0], our_param.shape[1]):
                if hf_weight.shape == (our_param.shape[1], our_param.shape[0]):
                    hf_weight = hf_weight.T
                else:
                    print(f"Warning: Shape mismatch for {hf_key}: "
                          f"HF {hf_weight.shape} vs ours {our_param.shape}")
                    continue
            
            # Copy weight to our parameter
            our_param.data.copy_from_numpy(hf_weight.astype(np.float32))
            print(f"✓ Loaded {hf_key} -> {hf_weight.shape}")
        else:
            print(f"✗ Missing {hf_key} in HuggingFace model")
    
    print("✓ Weight loading complete!")

def _create_weight_mapping(config: LLaMAConfig) -> Dict[str, Any]:
    """
    Create mapping from HuggingFace parameter names to our parameters
    
    HuggingFace LLaMA naming convention:
    - model.embed_tokens.weight
    - model.layers.{i}.self_attn.q_proj.weight
    - model.layers.{i}.self_attn.k_proj.weight
    - model.layers.{i}.self_attn.v_proj.weight
    - model.layers.{i}.self_attn.o_proj.weight
    - model.layers.{i}.mlp.gate_proj.weight
    - model.layers.{i}.mlp.up_proj.weight
    - model.layers.{i}.mlp.down_proj.weight
    - model.layers.{i}.input_layernorm.weight
    - model.layers.{i}.post_attention_layernorm.weight
    - model.norm.weight
    - lm_head.weight
    """
    # This is a placeholder - you'll need to populate with actual parameter references
    # from your model instance
    weight_map = {}
    
    # TODO: Implement the actual mapping
    # Example structure:
    # weight_map["model.embed_tokens.weight"] = model.embed_tokens
    # for i in range(config.num_hidden_layers):
    #     layer = model.layers[i]
    #     weight_map[f"model.layers.{i}.self_attn.q_proj.weight"] = layer.self_attn.q_proj
    #     ...
    
    return weight_map

def save_converted_weights(model: LLaMAModel, output_path: str):
    """Save model weights in our format"""
    model.save_checkpoint(output_path)

def load_converted_weights(model: LLaMAModel, input_path: str):
    """Load model weights from our format"""
    model.load_checkpoint(input_path)

def download_and_convert_weights(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_path: str = "tinyllama_weights.npz"
):
    """
    Download HuggingFace weights and convert to our format
    
    Usage:
        python weight_loader.py
    """
    # Create model with matching config
    config = LLaMAConfig.from_pretrained(model_name)
    model = LLaMAModel(config, is_cuda=True)
    
    # Load HF weights
    load_huggingface_weights(model, model_name)
    
    # Save in our format
    save_converted_weights(model, output_path)
    
    print(f"\n✓ Weights saved to {output_path}")
    print(f"  To load: model.load_checkpoint('{output_path}')")

def verify_weight_loading(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Verify that loaded weights produce same outputs as HuggingFace
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("Please install transformers and torch for verification")
        return
    
    print("Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_model.eval()
    
    print("Loading our model...")
    config = LLaMAConfig.from_pretrained(model_name)
    our_model = LLaMAModel(config, is_cuda=True)
    load_huggingface_weights(our_model, model_name)
    
    # Test with random input
    input_ids = np.random.randint(0, config.vocab_size, size=4, dtype=np.uint32)
    
    # HF forward pass
    with torch.no_grad():
        hf_input = torch.tensor(input_ids).unsqueeze(0)
        hf_output = hf_model(hf_input).logits[0].cpu().numpy()
    
    # Our forward pass
    from agtensor import no_grad
    with no_grad():
        our_output = our_model(input_ids).numpy()
    
    # Compare outputs
    diff = np.abs(hf_output - our_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nOutput comparison:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < 0.01:
        print("✓ Outputs match closely!")
    else:
        print("✗ Large difference detected - check weight loading")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert HuggingFace LLaMA weights")
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output",
        default="tinyllama_weights.npz",
        help="Output file path"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify weight loading against HuggingFace"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_weight_loading(args.model)
    else:
        download_and_convert_weights(args.model, args.output)
