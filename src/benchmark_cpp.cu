/**
 * Standalone C++ benchmark for LLaMA model
 * 
 * Compile: 
 *   nvcc -O3 -std=c++17 benchmark_cpp.cu -o benchmark_cpp
 * 
 * Run:
 *   ./benchmark_cpp
 */

#include "llama_config.hpp"
#include "llama_model.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace llama;

void print_benchmark_results(const std::string& name, const BenchmarkResult& result) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << name << " Benchmark Results\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Forward pass time:    " << result.forward_time_ms << " ms\n";
    std::cout << "Throughput:           " << result.tokens_per_second << " tokens/sec\n";
    std::cout << "GPU memory:           " << (result.memory_used_bytes / (1024*1024)) << " MB\n";
    std::cout << "Parameters:           " << (result.num_parameters / 1000000) << "M\n";
    std::cout << std::string(60, '=') << "\n";
}

int main(int argc, char** argv) {
    std::cout << "TinyLLaMA C++ Benchmark\n";
    std::cout << "=======================\n\n";
    
    // Parse arguments
    int batch_size = 1;
    int num_warmup = 5;
    int num_iterations = 100;
    
    if (argc > 1) batch_size = std::atoi(argv[1]);
    if (argc > 2) num_warmup = std::atoi(argv[2]);
    if (argc > 3) num_iterations = std::atoi(argv[3]);
    
    std::cout << "Configuration:\n";
    std::cout << "  Batch size:    " << batch_size << "\n";
    std::cout << "  Warmup runs:   " << num_warmup << "\n";
    std::cout << "  Iterations:    " << num_iterations << "\n\n";
    
    // Create TinyLLaMA config
    LLaMAConfig config = LLaMAConfig::tinyllama_1_1b();
    std::cout << "Model config:\n" << config.to_string() << "\n";
    
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\nGPU: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB\n\n";
    
    try {
        // Create model
        std::cout << "Creating model...\n";
        LLaMAModel<float> model(config, true);
        std::cout << "Model created successfully!\n";
        std::cout << "Parameters: " << (model.count_parameters() / 1000000) << "M\n\n";
        
        // Run benchmark
        std::cout << "Running benchmark...\n";
        auto result = benchmark_model(model, batch_size, num_warmup, num_iterations);
        
        // Print results
        print_benchmark_results("TinyLLaMA C++", result);
        
        // Additional metrics
        float params_gb = (model.count_parameters() * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
        float mem_gb = result.memory_used_bytes / (1024.0f * 1024.0f * 1024.0f);
        
        std::cout << "\nAdditional Metrics:\n";
        std::cout << "  Model size:        " << params_gb << " GB (FP32)\n";
        std::cout << "  Peak memory:       " << mem_gb << " GB\n";
        std::cout << "  Memory efficiency: " << (params_gb / mem_gb * 100) << "%\n";
        std::cout << "  FLOPS estimate:    " << (model.count_parameters() * 2 * batch_size / (result.forward_time_ms / 1000.0f) / 1e9) << " GFLOPS\n";
        
        std::cout << "\nBenchmark complete!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
