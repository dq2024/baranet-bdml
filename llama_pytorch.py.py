import sys
sys.path.append("build")

import numpy as np
import time
import argparse
from typing import Dict, Any
import json

# Import our implementations
import bten
from benchmark_latency import create_tinyllama_pytorch
from mygrad.llama.llama_config import LLaMAConfig

def benchmark_python_impl(batch_size: int, num_warmup: int = 5, num_iterations: int = 100) -> Dict[str, Any]:
    """Benchmark Python implementation"""
    print("\n" + "="*60)
    print("Benchmarking Python Implementation")
    print("="*60)
    
    import torch
    
    # Create model
    print("Creating model...")
    model = create_tinyllama_pytorch(device='cuda')
    
    # Create input
    input_ids = torch.randint(0, 32000, (batch_size,), device='cuda')
    
    # Warmup
    print(f"Warmup: {num_warmup} iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids)
    
    # Benchmark
    print(f"Benchmarking: {num_iterations} iterations...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(input_ids)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    # Compute statistics
    times = np.array(times)
    result = {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'tokens_per_second': batch_size * 1000 / np.mean(times),
        'num_parameters': model.count_parameters(),
    }
    
    return result

def benchmark_cpp_impl(batch_size: int, num_warmup: int = 5, num_iterations: int = 100) -> Dict[str, Any]:
    """Benchmark C++ implementation"""
    print("\n" + "="*60)
    print("Benchmarking C++ Implementation")
    print("="*60)
    
    # Check if C++ bindings are available
    try:
        config = bten.LLaMAConfig.tinyllama_1_1b()
        model = bten.LLaMAModelCpp(config, True)
    except AttributeError:
        print("C++ LLaMA bindings not available!")
        print("You need to add llama_bindings.hpp to your bindings.cu")
        return None
    
    print("Model created successfully!")
    print(f"Parameters: {model.count_parameters() / 1e6:.1f}M")
    
    # Run benchmark
    print(f"Running benchmark...")
    result = model.benchmark(batch_size, num_warmup, num_iterations)
    
    return result

def benchmark_pytorch(batch_size: int, num_warmup: int = 5, num_iterations: int = 100) -> Dict[str, Any]:
    """Benchmark PyTorch implementation"""
    print("\n" + "="*60)
    print("Benchmarking PyTorch Implementation")
    print("="*60)
    
    try:
        import torch
    except ImportError:
        print("PyTorch not available!")
        return None
    
    try:
        from benchmark_latency import create_tinyllama_pytorch
    except ImportError:
        print("benchmark_latency.py not found! Make sure it's in the same directory.")
        return None
    
    # Create TinyLLaMA model
    print("Creating model...")
    model = create_tinyllama_pytorch(device='cuda')
    
    # Create input
    input_ids = torch.randint(0, 32000, (batch_size,), device='cuda')
    
    # Warmup
    print(f"Warmup: {num_warmup} iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids)
    
    # Benchmark
    print(f"Benchmarking: {num_iterations} iterations...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_ids)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    # Compute statistics
    times = np.array(times)
    result = {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'tokens_per_second': batch_size * 1000 / np.mean(times),
        'num_parameters': model.count_parameters(),
    }
    
    return result

def print_results(name: str, results: Dict[str, Any]):
    """Print benchmark results"""
    if results is None:
        print(f"\n{name}: Not available")
        return
    
    # Check if results dict is empty or invalid
    if not results or not isinstance(results, dict):
        print(f"\n{name}: Invalid results")
        return
    
    print(f"\n{'='*60}")
    print(f"{name} Results")
    print('='*60)
    
    # Handle different key names safely
    mean_time = results.get('mean_time_ms', results.get('forward_time_ms', 0))
    if mean_time == 0:
        print("No timing data available")
        return
        
    std_time = results.get('std_time_ms', 0)
    min_time = results.get('min_time_ms', 0)
    max_time = results.get('max_time_ms', 0)
    throughput = results.get('tokens_per_second', 0)
    params = results.get('num_parameters', 0)
    
    print(f"Mean time:      {mean_time:.2f}" + (f" ± {std_time:.2f}" if std_time > 0 else "") + " ms")
    if min_time > 0:
        print(f"Min time:       {min_time:.2f} ms")
    if max_time > 0:
        print(f"Max time:       {max_time:.2f} ms")
    print(f"Throughput:     {throughput:.2f} tokens/sec")
    print(f"Parameters:     {params / 1e6:.1f}M")
    
    if 'memory_used_bytes' in results:
        print(f"GPU Memory:     {results['memory_used_bytes'] / (1024**3):.2f} GB")

def compare_results(results: Dict[str, Dict[str, Any]]):
    """Compare results across implementations"""
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    
    # Find baseline (prefer pytorch, fallback to first available)
    baseline = None
    baseline_name = None
    baseline_throughput = None
    
    if 'pytorch' in results and results['pytorch'] is not None:
        baseline_name = 'pytorch'
        baseline = results['pytorch'].get('mean_time_ms') or results['pytorch'].get('forward_time_ms')
        baseline_throughput = results['pytorch']['tokens_per_second']
    else:
        # Use first available as baseline
        for name, result in results.items():
            if result is not None and isinstance(result, dict):
                baseline_name = name
                baseline = result.get('mean_time_ms') or result.get('forward_time_ms')
                baseline_throughput = result.get('tokens_per_second', 0)
                if baseline and baseline > 0:
                    break
    
    if baseline is None or baseline == 0:
        print("No valid baseline found")
        return
    
    print(f"\nSpeedup vs {baseline_name}:")
    for name, result in results.items():
        if name == baseline_name or result is None or not isinstance(result, dict):
            continue
        
        mean_time = result.get('mean_time_ms') or result.get('forward_time_ms', 0)
        throughput = result.get('tokens_per_second', 0)
        
        if mean_time == 0 or throughput == 0:
            continue
            
        speedup = baseline / mean_time
        throughput_ratio = throughput / baseline_throughput
        
        print(f"  {name:20s}: {speedup:.2f}x faster ({throughput_ratio:.2f}x throughput)")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Benchmark LLaMA implementations')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--skip-pytorch', action='store_true', help='Skip PyTorch benchmark')
    parser.add_argument('--skip-python', action='store_true', help='Skip Python implementation')
    parser.add_argument('--skip-cpp', action='store_true', help='Skip C++ implementation')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    print("TinyLLaMA Benchmark")
    print("="*60)
    print(f"Configuration:")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Warmup:        {args.warmup}")
    print(f"  Iterations:    {args.iterations}")
    
    results = {}
    
    # Benchmark implementations
    if not args.skip_python:
        try:
            results['python'] = benchmark_python_impl(args.batch_size, args.warmup, args.iterations)
        except Exception as e:
            print(f"Python benchmark failed: {e}")
            results['python'] = None
    
    if not args.skip_cpp:
        try:
            results['cpp'] = benchmark_cpp_impl(args.batch_size, args.warmup, args.iterations)
        except Exception as e:
            print(f"C++ benchmark failed: {e}")
            results['cpp'] = None
    
    if not args.skip_pytorch:
        try:
            results['pytorch'] = benchmark_pytorch(args.batch_size, args.warmup, args.iterations)
        except Exception as e:
            print(f"PyTorch benchmark failed: {e}")
            results['pytorch'] = None
    
    # Print results
    for name, result in results.items():
        print_results(name.capitalize(), result)
    
    # Compare
    compare_results(results)
    
    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()