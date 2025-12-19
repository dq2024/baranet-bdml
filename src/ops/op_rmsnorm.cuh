#pragma once

#include "utils/tensor.cuh"
#include <cuda_runtime.h>

// RMSNorm kernel: y = x * weight / sqrt(mean(x^2) + eps)
// This is applied row-wise (each token sequence position independently)
template <typename T>
__global__ void op_rmsnorm_kernel(
    const Tensor<T> input,      // shape: (batch_size, hidden_dim)
    const Tensor<T> weight,     // shape: (1, hidden_dim) 
    Tensor<T> output,           // shape: (batch_size, hidden_dim)
    T eps
) {
    int row = blockIdx.x;
    if (row >= input.h) return;
    
    int tid = threadIdx.x;
    int hidden_dim = input.w;
    
    // Shared memory for reduction
    extern __shared__ T shared_mem[];
    
    // Compute sum of squares for this row
    T sum_sq = 0;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        T val = Index(input, row, i);
        sum_sq += val * val;
    }
    
    // Store partial sum in shared memory
    shared_mem[tid] = sum_sq;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute RMS
    T mean_sq = shared_mem[0] / hidden_dim;
    T rms = sqrt(mean_sq + eps);
    
    // Normalize and apply weight
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        T normalized = Index(input, row, i) / rms;
        T w = Index(weight, 0, i);
        Index(output, row, i) = normalized * w;
    }
}

// Backward pass for RMSNorm
template <typename T>
__global__ void op_rmsnorm_backward_kernel(
    const Tensor<T> input,       // shape: (batch_size, hidden_dim)
    const Tensor<T> weight,      // shape: (1, hidden_dim)
    const Tensor<T> grad_output, // shape: (batch_size, hidden_dim)
    Tensor<T> grad_input,        // shape: (batch_size, hidden_dim)
    Tensor<T> grad_weight,       // shape: (1, hidden_dim)
    T eps
) {
    int row = blockIdx.x;
    if (row >= input.h) return;
    
    int tid = threadIdx.x;
    int hidden_dim = input.w;
    
    extern __shared__ T shared_mem[];
    T* sum_sq_shared = shared_mem;
    T* sum_grad_shared = &shared_mem[blockDim.x];
    
    // Recompute RMS (forward pass values)
    T sum_sq = 0;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        T val = Index(input, row, i);
        sum_sq += val * val;
    }
    
    sum_sq_shared[tid] = sum_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_sq_shared[tid] += sum_sq_shared[tid + stride];
        }
        __syncthreads();
    }
    
    T mean_sq = sum_sq_shared[0] / hidden_dim;
    T rms = sqrt(mean_sq + eps);
    T inv_rms = 1.0 / rms;
    T inv_rms3 = inv_rms * inv_rms * inv_rms;
    
    // Compute sum of grad_output * input for this row
    T sum_grad = 0;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        T g_out = Index(grad_output, row, i);
        T w = Index(weight, 0, i);
        T x = Index(input, row, i);
        sum_grad += g_out * w * x;
    }
    
    sum_grad_shared[tid] = sum_grad;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_grad_shared[tid] += sum_grad_shared[tid + stride];
        }
        __syncthreads();
    }
    
    T sum_grad_total = sum_grad_shared[0];
    
    // Compute gradients
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        T x = Index(input, row, i);
        T g_out = Index(grad_output, row, i);
        T w = Index(weight, 0, i);
        
        // Gradient w.r.t input
        T grad_norm = g_out * w * inv_rms;
        T grad_rms = -inv_rms3 * x * sum_grad_total / hidden_dim;
        Index(grad_input, row, i) = grad_norm + grad_rms;
        
        // Gradient w.r.t weight (accumulate across batch)
        T normalized = x * inv_rms;
        atomicAdd(&Index(grad_weight, 0, i), g_out * normalized);
    }
}

template <typename T>
void op_rmsnorm(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    Tensor<T>& output,
    T eps = 1e-6
) {
    if (input.h != output.h || input.w != output.w) {
        throw std::runtime_error("RMSNorm: input/output shape mismatch");
    }
    if (weight.h != 1 || weight.w != input.w) {
        throw std::runtime_error("RMSNorm: weight shape must be (1, hidden_dim)");
    }
    if (!input.on_device || !weight.on_device || !output.on_device) {
        throw std::runtime_error("RMSNorm: only GPU implementation available");
    }
    
    int threads = 256;
    int blocks = input.h;
    size_t shared_mem_size = threads * sizeof(T);
    
    op_rmsnorm_kernel<<<blocks, threads, shared_mem_size>>>(
        input, weight, output, eps
    );
}

template <typename T>
void op_rmsnorm_backward(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& grad_output,
    Tensor<T>& grad_input,
    Tensor<T>& grad_weight,
    T eps = 1e-6
) {
    if (input.h != grad_output.h || input.w != grad_output.w) {
        throw std::runtime_error("RMSNorm backward: shape mismatch");
    }
    if (!input.on_device) {
        throw std::runtime_error("RMSNorm backward: only GPU implementation available");
    }
    
    // Zero out grad_weight since we're accumulating
    op_const_fill(grad_weight, (T)0);
    
    int threads = 256;
    int blocks = input.h;
    size_t shared_mem_size = threads * 2 * sizeof(T);
    
    op_rmsnorm_backward_kernel<<<blocks, threads, shared_mem_size>>>(
        input, weight, grad_output, grad_input, grad_weight, eps
    );
}
