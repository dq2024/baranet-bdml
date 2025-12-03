#pragma once

#include "llama_config.hpp"
#include "llama_layers.hpp"
#include "ops/op_elemwise.cuh"
#include "utils/tensor.cuh"
#include <vector>
#include <memory>
#include <chrono>

namespace llama {

/**
 * Complete LLaMA model for inference
 * 
 * This is the C++ implementation suitable for benchmarking.
 * No Python overhead - pure C++/CUDA execution.
 */
template <typename T>
class LLaMAModel {
public:
    LLaMAModel(const LLaMAConfig& config, bool on_device = true)
        : config_(config), on_device_(on_device),
          embed_tokens_(config.vocab_size, config.hidden_size, on_device),
          final_norm_(config.hidden_size, config.rms_norm_eps, on_device),
          lm_head_(config.hidden_size, config.vocab_size, on_device) {
        
        config_.validate();
        
        // Initialize embedding with small random values
        op_uniform_fill(embed_tokens_, (T)-0.02, (T)0.02);
        
        // Create decoder layers
        for (int i = 0; i < config_.num_hidden_layers; i++) {
            layers_.emplace_back(
                std::make_unique<LLaMADecoderLayer<T>>(config_, on_device_)
            );
        }
        
        // Initialize LM head
        op_uniform_fill(lm_head_, (T)-0.02, (T)0.02);
    }
    
    /**
     * Forward pass: token IDs -> logits
     * 
     * @param input_ids: (batch_size,) tensor of token IDs (uint32)
     * @param logits: (batch_size, vocab_size) output tensor
     * @param position_offset: Starting position for RoPE
     * @param use_causal_mask: Whether to use causal masking
     */
    void forward(
        const Tensor<uint32_t>& input_ids,
        Tensor<T>& logits,
        int position_offset = 0,
        bool use_causal_mask = true
    ) {
        int batch_size = input_ids.h;
        int hidden_size = config_.hidden_size;
        
        // Embed tokens
        Tensor<T> hidden_states(batch_size, hidden_size, on_device_);
        embed(input_ids, hidden_states);
        
        // Pass through all decoder layers
        Tensor<T> layer_input = hidden_states;
        Tensor<T> layer_output(batch_size, hidden_size, on_device_);
        
        for (int i = 0; i < config_.num_hidden_layers; i++) {
            layers_[i]->forward(layer_input, layer_output, position_offset, use_causal_mask);
            layer_input = layer_output;
        }
        
        // Final norm
        Tensor<T> normed(batch_size, hidden_size, on_device_);
        final_norm_.forward(layer_output, normed);
        
        // Project to vocabulary
        op_mm(normed, lm_head_, logits);
    }
    
    /**
     * Timed forward pass for benchmarking
     * Returns elapsed time in milliseconds
     */
    float forward_timed(
        const Tensor<uint32_t>& input_ids,
        Tensor<T>& logits,
        int position_offset = 0,
        bool use_causal_mask = true
    ) {
        if (!on_device_) {
            // CPU timing
            auto start = std::chrono::high_resolution_clock::now();
            forward(input_ids, logits, position_offset, use_causal_mask);
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<float, std::milli> duration = end - start;
            return duration.count();
        } else {
            // GPU timing with CUDA events
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            forward(input_ids, logits, position_offset, use_causal_mask);
            cudaEventRecord(stop);
            
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            return milliseconds;
        }
    }
    
    /**
     * Get current GPU memory usage in bytes
     */
    size_t get_gpu_memory_usage() const {
        if (!on_device_) return 0;
        
        size_t free_byte, total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);
        return total_byte - free_byte;
    }
    
    /**
     * Count total parameters
     */
    size_t count_parameters() const {
        return config_.count_parameters();
    }
    
    // Component accessors for weight loading
    Tensor<T>& embed_tokens() { return embed_tokens_; }
    std::vector<std::unique_ptr<LLaMADecoderLayer<T>>>& layers() { return layers_; }
    RMSNorm<T>& final_norm() { return final_norm_; }
    Tensor<T>& lm_head() { return lm_head_; }
    
    const LLaMAConfig& config() const { return config_; }
    
private:
    LLaMAConfig config_;
    bool on_device_;
    
    // Model components
    Tensor<T> embed_tokens_;  // (vocab_size, hidden_size)
    std::vector<std::unique_ptr<LLaMADecoderLayer<T>>> layers_;
    RMSNorm<T> final_norm_;
    Tensor<T> lm_head_;  // (hidden_size, vocab_size)
    
    /**
     * Embed token IDs to hidden states
     */
    void embed(const Tensor<uint32_t>& input_ids, Tensor<T>& embeddings) {
        int batch_size = input_ids.h;
        int hidden_size = config_.hidden_size;
        
        // Copy embedding table to host for lookup
        // (In production, you'd want a custom CUDA kernel for this)
        Tensor<T> embed_host(config_.vocab_size, hidden_size, false);
        Tensor<T> output_host(batch_size, hidden_size, false);
        
        if (on_device_) {
            embed_tokens_.toHost(embed_host);
        } else {
            embed_host = embed_tokens_;
        }
        
        // Copy input IDs to host
        Tensor<uint32_t> ids_host(batch_size, 1, false);
        if (on_device_) {
            input_ids.toHost(ids_host);
        } else {
            ids_host = input_ids;
        }
        
        // Lookup embeddings
        for (int i = 0; i < batch_size; i++) {
            uint32_t token_id = Index(ids_host, i, 0);
            for (int j = 0; j < hidden_size; j++) {
                Index(output_host, i, j) = Index(embed_host, token_id, j);
            }
        }
        
        // Copy back to device
        if (on_device_) {
            output_host.toDevice(embeddings);
        } else {
            embeddings = output_host;
        }
    }
};

/**
 * Benchmark utilities
 */
struct BenchmarkResult {
    float forward_time_ms;
    float tokens_per_second;
    size_t memory_used_bytes;
    size_t num_parameters;
};

template <typename T>
BenchmarkResult benchmark_model(
    LLaMAModel<T>& model,
    int batch_size,
    int num_warmup = 5,
    int num_iterations = 100
) {
    BenchmarkResult result = {0};
    
    // Create input
    //Tensor<uint32_t> input_ids(batch_size, 1, model.config().on_device);
    bool on_device = true; // Assume GPU for benchmarking
    Tensor<uint32_t> input_ids(batch_size, 1, on_device);
    // op_const_fill(input_ids, (uint32_t)1);  // Fill with token ID 1
    // Fill manually (op_const_fill may not work for uint32)
    Tensor<uint32_t> host_ids(batch_size, 1, false);
    for (int i = 0; i < batch_size; i++) {
        Index(host_ids, i, 0) = (uint32_t)1;
    }
    if (on_device) {
        host_ids.toDevice(input_ids);
    } else {
        input_ids = host_ids;
    }
    
    //Tensor<T> logits(batch_size, model.config().vocab_size, model.config().on_device);
    Tensor<T> logits(batch_size, model.config().vocab_size, on_device);

    
    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        model.forward(input_ids, logits);
    }
    
    // Measure memory before benchmark
    size_t mem_before = model.get_gpu_memory_usage();
    
    // Benchmark
    float total_time = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        float time = model.forward_timed(input_ids, logits);
        total_time += time;
    }
    
    // Measure memory after
    size_t mem_after = model.get_gpu_memory_usage();
    
    // Compute metrics
    result.forward_time_ms = total_time / num_iterations;
    result.tokens_per_second = (batch_size * 1000.0f) / result.forward_time_ms;
    result.memory_used_bytes = mem_after;
    result.num_parameters = model.count_parameters();
    
    return result;
}

} // namespace llama
