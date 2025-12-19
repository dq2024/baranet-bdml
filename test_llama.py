"""
Test script for TinyLLaMA implementation
Tests individual components and full model
"""
import numpy as np
import sys
sys.path.append("build")

from mygrad.engine import AGTensor, no_grad
from mygrad.agtensor_llama import *
from mygrad.llama.llama_config import LLaMAConfig
from mygrad.llama.llama_layers import RMSNorm, RotaryEmbedding, GroupedQueryAttention, MLP, LLaMADecoderLayer
from mygrad.llama.llama_model import LLaMAModel, create_tinyllama_model

def test_rmsnorm():
    """Test RMSNorm layer"""
    print("\n" + "="*50)
    print("Testing RMSNorm...")
    print("="*50)
    
    batch_size, hidden_size = 4, 64
    x_np = np.random.randn(batch_size, hidden_size).astype(np.float32)
    x = AGTensor(x_np, requires_grad=True, is_cuda=True)
    
    norm = RMSNorm(hidden_size, is_cuda=True)
    y = norm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Input mean: {x_np.mean():.4f}, std: {x_np.std():.4f}")
    
    y_np = y.numpy()
    print(f"Output mean: {y_np.mean():.4f}, std: {y_np.std():.4f}")
    
    # Test backward
    loss = y.mean()
    loss.backward()
    
    print(f"Gradient computed: {x.grad is not None}")
    print("✓ RMSNorm test passed")

def test_rope():
    """Test Rotary Position Embeddings"""
    print("\n" + "="*50)
    print("Testing RoPE...")
    print("="*50)
    
    batch_size = 4
    hidden_size = 64
    head_dim = 32
    
    x_np = np.random.randn(batch_size, hidden_size).astype(np.float32)
    x = AGTensor(x_np, requires_grad=True, is_cuda=True)
    
    rope = RotaryEmbedding(head_dim)
    y = rope(x, position_offset=0)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Head dim: {head_dim}")
    
    # Test backward
    loss = y.mean()
    loss.backward()
    
    print(f"Gradient computed: {x.grad is not None}")
    print("✓ RoPE test passed")

def test_silu():
    """Test SiLU activation"""
    print("\n" + "="*50)
    print("Testing SiLU...")
    print("="*50)
    
    x_np = np.array([[[-2, -1, 0, 1, 2]]], dtype=np.float32).reshape(1, 5)
    x = AGTensor(x_np, requires_grad=True, is_cuda=True)
    
    y = x.silu()
    
    print(f"Input: {x_np}")
    print(f"Output: {y.numpy()}")
    
    # Test backward
    loss = y.mean()
    loss.backward()
    
    print(f"Gradient computed: {x.grad is not None}")
    print("✓ SiLU test passed")

def test_softmax():
    """Test softmax"""
    print("\n" + "="*50)
    print("Testing Softmax...")
    print("="*50)
    
    x_np = np.array([[1, 2, 3, 4]], dtype=np.float32)
    x = AGTensor(x_np, requires_grad=True, is_cuda=True)
    
    # Regular softmax
    y = x.softmax(causal=False)
    y_np = y.numpy()
    
    print(f"Input: {x_np}")
    print(f"Softmax output: {y_np}")
    print(f"Sum of probabilities: {y_np.sum():.4f}")
    
    # Test backward
    loss = y.mean()
    loss.backward()
    
    print(f"Gradient computed: {x.grad is not None}")
    print("✓ Softmax test passed")

def test_mlp():
    """Test MLP layer"""
    print("\n" + "="*50)
    print("Testing MLP...")
    print("="*50)
    
    config = LLaMAConfig()
    config.hidden_size = 128
    config.intermediate_size = 256
    
    batch_size = 4
    x_np = np.random.randn(batch_size, config.hidden_size).astype(np.float32)
    x = AGTensor(x_np, requires_grad=True, is_cuda=True)
    
    mlp = MLP(config, is_cuda=True)
    y = mlp(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of parameters: {len(mlp.parameters())}")
    
    # Test backward
    loss = y.mean()
    loss.backward()
    
    print(f"Gradient computed: {x.grad is not None}")
    print("✓ MLP test passed")

def test_decoder_layer():
    """Test full decoder layer"""
    print("\n" + "="*50)
    print("Testing Decoder Layer...")
    print("="*50)
    
    config = LLaMAConfig()
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.intermediate_size = 256
    
    batch_size = 2
    x_np = np.random.randn(batch_size, config.hidden_size).astype(np.float32)
    x = AGTensor(x_np, requires_grad=True, is_cuda=True)
    
    layer = LLaMADecoderLayer(config, is_cuda=True)
    y = layer(x, position_offset=0)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of parameters: {len(layer.parameters())}")
    
    print("✓ Decoder layer test passed")

def test_small_model():
    """Test a small LLaMA model"""
    print("\n" + "="*50)
    print("Testing Small LLaMA Model...")
    print("="*50)
    
    # Create a tiny config for testing
    config = LLaMAConfig()
    config.vocab_size = 1000
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    
    model = LLaMAModel(config, is_cuda=True)
    
    # Test forward pass
    input_ids = np.array([1, 2, 3, 4], dtype=np.uint32)
    
    with no_grad():
        logits = model(input_ids)
    
    print(f"Input IDs: {input_ids}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: ({len(input_ids)}, {config.vocab_size})")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(input_ids[:2], max_new_tokens=5)
    print(f"Generated sequence: {generated}")
    
    print("✓ Small model test passed")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" RUNNING TINYLLAMA IMPLEMENTATION TESTS")
    print("="*70)
    
    try:
        test_rmsnorm()
        test_rope()
        test_silu()
        test_softmax()
        test_mlp()
        test_decoder_layer()
        test_small_model()
        
        print("\n" + "="*70)
        print(" ✓ ALL TESTS PASSED!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
