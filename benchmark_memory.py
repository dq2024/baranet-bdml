"""
Benchmark memory usage of LLaMA implementations

Usage:
    python scripts/benchmark_memory.py --batch-size 1
"""

import sys
sys.path.append("build")

import numpy as np
import argparse
from typing import Dict, Any
import json

def get_gpu_memory_usage():
    """Get current GPU memory usage in bytes"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
    except:
        pass
    
    # Fallback to nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True
        )
        return int(result.stdout.strip()) * 1024 * 1024  # Convert MB to bytes
    except:
        return 0

def benchmark_python_memory(batch_size: int) -> Dict[str, Any]:
    """Benchmark Python implementation memory"""
    print("\n" + "="*60)
    print("Benchmarking Python Implementation Memory")
    print("="*60)
    
    from agtensor import no_grad
    from llama_model import create_tinyllama_model
    from llama_config import LLaMAConfig
    
    # Measure before
    mem_before = get_gpu_memory_usage()
    
    # Create model
    print("Creating model...")
    model = create_tinyllama_model(is_cuda=True)
    
    # Measure after model creation
    mem_after_model = get_gpu_memory_usage()
    model_size = mem_after_model - mem_before
    
    # Create input
    config = LLaMAConfig()
    input_ids = np.random.randint(0, config.vocab_size, size=batch_size, dtype=np.uint32)
    
    # Forward pass
    print("Running forward pass...")
    with no_grad():
        logits = model(input_ids)
    
    # Measure peak memory
    mem_peak = get_gpu_memory_usage()
    activation_memory = mem_peak - mem_after_model
    
    result = {
        'model_memory_mb': model_size / (1024**2),
        'activation_memory_mb': activation_memory / (1024**2),
        'peak_memory_mb': mem_peak / (1024**2),
        'num_parameters': model.count_parameters(),
        'batch_size': batch_size,
    }
    
    return result

def benchmark_cpp_memory(batch_size: int) -> Dict[str, Any]:
    """Benchmark C++ implementation memory"""
    print("\n" + "="*60)
    print("Benchmarking C++ Implementation Memory")
    print("="*60)
    
    import bten
    
    try:
        # Measure before
        mem_before = get_gpu_memory_usage()
        
        # Create model
        print("Creating model...")
        config = bten.LLaMAConfig.tinyllama_1_1b()
        model = bten.LLaMAModelCpp(config, True)
        
        # Measure after model creation
        mem_after_model = model.get_gpu_memory_usage()
        model_size = mem_after_model - mem_before
        
        # Create input
        input_ids = np.random.randint(0, config.vocab_size, size=batch_size, dtype=np.uint32)
        
        # Forward pass
        print("Running forward pass...")
        logits = model.forward(input_ids)
        
        # Measure peak memory
        mem_peak = model.get_gpu_memory_usage()
        activation_memory = mem_peak - mem_after_model
        
        result = {
            'model_memory_mb': model_size / (1024**2),
            'activation_memory_mb': activation_memory / (1024**2),
            'peak_memory_mb': mem_peak / (1024**2),
            'num_parameters': model.count_parameters(),
            'batch_size': batch_size,
        }
        
        return result
        
    except AttributeError:
        print("C++ LLaMA bindings not available!")
        return None

def benchmark_pytorch_memory(batch_size: int) -> Dict[str, Any]:
    """Benchmark PyTorch implementation memory"""
    print("\n" + "="*60)
    print("Benchmarking PyTorch Implementation Memory")
    print("="*60)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        print("PyTorch/Transformers not available!")
        return None
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure before
    mem_before = torch.cuda.memory_allocated()
    
    # Create model
    print("Loading model...")
    config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_config(config)
    model = model.cuda()
    model.eval()
    
    # Measure after model creation
    mem_after_model = torch.cuda.memory_allocated()
    model_size = mem_after_model - mem_before
    
    # Create input
    input_ids = torch.randint(0, config.vocab_size, (batch_size,), device='cuda')
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        _ = model(input_ids)
    
    # Measure peak memory
    mem_peak = torch.cuda.max_memory_allocated()
    activation_memory = mem_peak - mem_after_model
    
    result = {
        'model_memory_mb': model_size / (1024**2),
        'activation_memory_mb': activation_memory / (1024**2),
        'peak_memory_mb': mem_peak / (1024**2),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'batch_size': batch_size,
    }
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return result

def print_results(name: str, results: Dict[str, Any]):
    """Print memory results"""
    if results is None:
        print(f"\n{name}: Not available")
        return
    
    print(f"\n{'='*60}")
    print(f"{name} Memory Usage")
    print('='*60)
    print(f"Batch size:         {results['batch_size']}")
    print(f"Parameters:         {results['num_parameters'] / 1e6:.1f}M")
    print(f"Model memory:       {results['model_memory_mb']:.2f} MB")
    print(f"Activation memory:  {results['activation_memory_mb']:.2f} MB")
    print(f"Peak memory:        {results['peak_memory_mb']:.2f} MB")
    print(f"Memory efficiency:  {results['model_memory_mb'] / results['peak_memory_mb'] * 100:.1f}%")

def compare_memory(results: Dict[str, Dict[str, Any]]):
    """Compare memory usage across implementations"""
    print("\n" + "="*60)
    print("Memory Comparison")
    print("="*60)
    
    if 'pytorch' in results and results['pytorch'] is not None:
        baseline = results['pytorch']['peak_memory_mb']
        
        print(f"\nMemory vs PyTorch:")
        for name, result in results.items():
            if name == 'pytorch' or result is None:
                continue
            
            ratio = result['peak_memory_mb'] / baseline
            diff = result['peak_memory_mb'] - baseline
            
            print(f"  {name:20s}: {ratio:.2f}x ({diff:+.2f} MB)")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Benchmark LLaMA memory usage')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--skip-pytorch', action='store_true', help='Skip PyTorch benchmark')
    parser.add_argument('--skip-python', action='store_true', help='Skip Python implementation')
    parser.add_argument('--skip-cpp', action='store_true', help='Skip C++ implementation')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    print("TinyLLaMA Memory Benchmark")
    print("="*60)
    print(f"Batch size: {args.batch_size}")
    
    results = {}
    
    # Benchmark implementations
    if not args.skip_python:
        try:
            results['python'] = benchmark_python_memory(args.batch_size)
        except Exception as e:
            print(f"Python benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            results['python'] = None
    
    if not args.skip_cpp:
        try:
            results['cpp'] = benchmark_cpp_memory(args.batch_size)
        except Exception as e:
            print(f"C++ benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            results['cpp'] = None
    
    if not args.skip_pytorch:
        try:
            results['pytorch'] = benchmark_pytorch_memory(args.batch_size)
        except Exception as e:
            print(f"PyTorch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            results['pytorch'] = None
    
    # Print results
    for name, result in results.items():
        print_results(name.capitalize(), result)
    
    # Compare
    compare_memory(results)
    
    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
