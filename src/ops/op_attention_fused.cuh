#pragma once

#include "utils/tensor.cuh"
#include <cuda_runtime.h>

/**
 * Fused Attention Kernel (simplified for 2D tensor structure)
 * Fuses QK^T, scaling, softmax, and V multiplication into a single kernel
 * Reduces memory traffic and temporary allocations
 * 
 * Note: This is a simplified version that works with the current 2D tensor structure
 * For full Flash Attention, we'd need 3D/4D tensor support
 */
template <typename T>
__global__ void op_attention_fused_kernel(
    const Tensor<T> Q,        // (batch, hidden_size)
    const Tensor<T> K,        // (batch, hidden_size)
    const Tensor<T> V,        // (batch, hidden_size)
    Tensor<T> output,         // (batch, hidden_size)
    T scale,                  // 1.0 / sqrt(head_dim)
    bool causal,              // Use causal masking
    int seq_offset            // Position offset for KV cache
) {
    int row = blockIdx.x;
    if (row >= Q.h) return;
    
    int tid = threadIdx.x;
    int seq_len = Q.h;  // For self-attention, batch_size = seq_len
    int hidden_dim = Q.w;
    
    extern __shared__ T shared_mem[];
    T* scores = shared_mem;
    T* sum_shared = &shared_mem[seq_len];
    
    // Compute attention scores: Q[row] @ K^T
    T max_score = -INFINITY;
    
    // Find max for numerical stability
    for (int j = tid; j < seq_len; j += blockDim.x) {
        T score = 0.0;
        for (int d = 0; d < hidden_dim; d++) {
            score += Index(Q, row, d) * Index(K, j, d);
        }
        score *= scale;
        
        // Apply causal mask if needed
        if (causal && j > (row + seq_offset)) {
            score = -INFINITY;
        }
        
        scores[j] = score;
        if (score > max_score) {
            max_score = score;
        }
    }
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < seq_len) {
            if (scores[tid + stride] > scores[tid]) {
                scores[tid] = scores[tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        max_score = scores[0];
    }
    __syncthreads();
    
    // Compute exp and sum
    T local_sum = 0.0;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        T exp_score = exp(scores[j] - max_score);
        scores[j] = exp_score;
        local_sum += exp_score;
    }
    
    // Store local sum in shared memory for reduction
    T* sum_shared = &shared_mem[seq_len];
    sum_shared[tid] = local_sum;
    __syncthreads();
    
    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }
    
    T sum_exp = sum_shared[0];
    
    T inv_sum = 1.0 / sum_exp;
    
    // Compute output: sum(softmax(QK^T) * V)
    for (int d = tid; d < hidden_dim; d += blockDim.x) {
        T out_val = 0.0;
        for (int j = 0; j < seq_len; j++) {
            T attn_weight = scores[j] * inv_sum;
            out_val += attn_weight * Index(V, j, d);
        }
        Index(output, row, d) = out_val;
    }
}


template <typename T>
void op_attention_fused(
    const Tensor<T>& Q,
    const Tensor<T>& K,
    const Tensor<T>& V,
    Tensor<T>& output,
    T scale,
    bool causal = false,
    int seq_offset = 0
) {
    if (Q.h != K.h || Q.h != V.h || Q.h != output.h) {
        throw std::runtime_error("Attention: batch size mismatch");
    }
    if (Q.w != K.w || Q.w != V.w || Q.w != output.w) {
        throw std::runtime_error("Attention: hidden size mismatch");
    }
    if (!Q.on_device || !K.on_device || !V.on_device || !output.on_device) {
        throw std::runtime_error("Attention: only GPU implementation available");
    }
    
    int batch_size = Q.h;
    int threads = 256;
    int blocks = batch_size;
    // Shared memory: scores array + reduction array
    size_t shared_mem_size = (batch_size + threads) * sizeof(T);
    
    op_attention_fused_kernel<<<blocks, threads, shared_mem_size>>>(
        Q, K, V, output, scale, causal, seq_offset
    );
}

