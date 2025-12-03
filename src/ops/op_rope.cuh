#pragma once

#include "utils/tensor.cuh"
#include <cuda_runtime.h>
#include <cmath>

// Rotary Position Embeddings (RoPE)
// Applied to query and key tensors in attention
// input shape: (batch_size, seq_len, num_heads, head_dim)
// We'll work with reshaped tensors: (batch_size * seq_len, num_heads * head_dim)

template <typename T>
__global__ void op_rope_kernel(
    const Tensor<T> input,      // shape: (batch_size, hidden_dim)
    Tensor<T> output,           // shape: (batch_size, hidden_dim)
    int position_offset,        // starting position in sequence
    int head_dim,              // dimension per head (usually hidden_dim / num_heads)
    T theta                     // base for rotation (usually 10000.0)
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch element
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // feature dimension
    
    if (row >= input.h || col >= input.w) return;
    
    // Each pair of dimensions gets rotated together
    int pair_idx = col / 2;  // which pair (0, 1, 2, ...)
    int within_pair = col % 2;  // 0 or 1 within the pair
    
    // Position is the row index plus offset
    int position = row + position_offset;
    
    // Compute rotation angle for this dimension pair
    T freq = 1.0 / pow(theta, (T)(2 * (pair_idx % (head_dim / 2))) / head_dim);
    T angle = position * freq;
    
    T cos_angle = cos(angle);
    T sin_angle = sin(angle);
    
    // Get the pair of values
    int col_even = (col / 2) * 2;
    int col_odd = col_even + 1;
    
    T x0 = Index(input, row, col_even);
    T x1 = Index(input, row, col_odd);
    
    // Apply rotation
    if (within_pair == 0) {
        Index(output, row, col) = x0 * cos_angle - x1 * sin_angle;
    } else {
        Index(output, row, col) = x0 * sin_angle + x1 * cos_angle;
    }
}

// Simplified version that assumes contiguous head_dim sections
template <typename T>
__global__ void op_rope_kernel_simple(
    const Tensor<T> input,
    Tensor<T> output,
    int position_offset,
    int head_dim,
    T theta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = input.h * input.w;
    
    if (idx >= total_elements) return;
    
    int row = idx / input.w;
    int col = idx % input.w;
    
    // Which head and dimension within head
    int head_idx = col / head_dim;
    int dim_in_head = col % head_dim;
    
    // Only apply to pairs (every 2 dimensions)
    int pair_idx = dim_in_head / 2;
    int within_pair = dim_in_head % 2;
    
    int position = row + position_offset;
    
    // Frequency for this dimension pair
    T freq = 1.0 / pow(theta, (T)(2 * pair_idx) / head_dim);
    T angle = position * freq;
    
    T cos_angle = cos(angle);
    T sin_angle = sin(angle);
    
    // Get paired dimension
    int base_col = head_idx * head_dim + (pair_idx * 2);
    T x0 = Index(input, row, base_col);
    T x1 = Index(input, row, base_col + 1);
    
    // Apply rotation
    if (within_pair == 0) {
        Index(output, row, col) = x0 * cos_angle - x1 * sin_angle;
    } else {
        Index(output, row, col) = x0 * sin_angle + x1 * cos_angle;
    }
}

template <typename T>
void op_rope(
    const Tensor<T>& input,
    Tensor<T>& output,
    int position_offset = 0,
    int head_dim = 64,
    T theta = 10000.0
) {
    if (input.h != output.h || input.w != output.w) {
        throw std::runtime_error("RoPE: input/output shape mismatch");
    }
    if (!input.on_device || !output.on_device) {
        throw std::runtime_error("RoPE: only GPU implementation available");
    }
    if (input.w % head_dim != 0) {
        throw std::runtime_error("RoPE: hidden_dim must be divisible by head_dim");
    }
    
    int total_elements = input.h * input.w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    op_rope_kernel_simple<<<blocks, threads>>>(
        input, output, position_offset, head_dim, theta
    );
}

// Backward pass for RoPE
// The backward of a rotation is just rotating in the opposite direction
template <typename T>
__global__ void op_rope_backward_kernel(
    const Tensor<T> grad_output,
    Tensor<T> grad_input,
    int position_offset,
    int head_dim,
    T theta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = grad_output.h * grad_output.w;
    
    if (idx >= total_elements) return;
    
    int row = idx / grad_output.w;
    int col = idx % grad_output.w;
    
    int head_idx = col / head_dim;
    int dim_in_head = col % head_dim;
    int pair_idx = dim_in_head / 2;
    int within_pair = dim_in_head % 2;
    
    int position = row + position_offset;
    
    T freq = 1.0 / pow(theta, (T)(2 * pair_idx) / head_dim);
    T angle = position * freq;
    
    T cos_angle = cos(angle);
    T sin_angle = sin(angle);
    
    int base_col = head_idx * head_dim + (pair_idx * 2);
    T g0 = Index(grad_output, row, base_col);
    T g1 = Index(grad_output, row, base_col + 1);
    
    // Backward rotation (transpose of forward rotation matrix)
    if (within_pair == 0) {
        Index(grad_input, row, col) = g0 * cos_angle + g1 * sin_angle;
    } else {
        Index(grad_input, row, col) = -g0 * sin_angle + g1 * cos_angle;
    }
}

template <typename T>
void op_rope_backward(
    const Tensor<T>& grad_output,
    Tensor<T>& grad_input,
    int position_offset = 0,
    int head_dim = 64,
    T theta = 10000.0
) {
    if (grad_output.h != grad_input.h || grad_output.w != grad_input.w) {
        throw std::runtime_error("RoPE backward: shape mismatch");
    }
    if (!grad_output.on_device) {
        throw std::runtime_error("RoPE backward: only GPU implementation available");
    }
    
    int total_elements = grad_output.h * grad_output.w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    op_rope_backward_kernel<<<blocks, threads>>>(
        grad_output, grad_input, position_offset, head_dim, theta
    );
}
