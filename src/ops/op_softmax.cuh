#pragma once

#include "utils/tensor.cuh"
#include <cuda_runtime.h>

// Softmax applied row-wise: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_i - max(x)))
// This is numerically stable version
template <typename T>
__global__ void op_softmax_kernel(
    const Tensor<T> input,
    Tensor<T> output
) {
    int row = blockIdx.x;
    if (row >= input.h) return;
    
    int tid = threadIdx.x;
    int seq_len = input.w;
    
    extern __shared__ T shared_mem[];
    T* max_shared = shared_mem;
    T* sum_shared = &shared_mem[blockDim.x];
    
    // Find max value in this row (for numerical stability)
    T local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        T val = Index(input, row, i);
        if (val > local_max) {
            local_max = val;
        }
    }
    
    max_shared[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (max_shared[tid + stride] > max_shared[tid]) {
                max_shared[tid] = max_shared[tid + stride];
            }
        }
        __syncthreads();
    }
    
    T max_val = max_shared[0];
    
    // Compute exp(x - max) and sum
    T local_sum = 0;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        T val = exp(Index(input, row, i) - max_val);
        Index(output, row, i) = val;  // Store temporarily
        local_sum += val;
    }
    
    sum_shared[tid] = local_sum;
    __syncthreads();
    
    // Reduce to find total sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }
    
    T sum_val = sum_shared[0];
    
    // Normalize
    for (int i = tid; i < seq_len; i += blockDim.x) {
        Index(output, row, i) = Index(output, row, i) / sum_val;
    }
}

// Softmax with causal mask (for autoregressive attention)
// Mask ensures position i can only attend to positions <= i
template <typename T>
__global__ void op_softmax_causal_kernel(
    const Tensor<T> input,
    Tensor<T> output,
    int seq_offset  // for KV cache: offset of current position
) {
    int row = blockIdx.x;
    if (row >= input.h) return;
    
    int tid = threadIdx.x;
    int seq_len = input.w;
    
    // Current query position
    int query_pos = row + seq_offset;
    
    extern __shared__ T shared_mem[];
    T* max_shared = shared_mem;
    T* sum_shared = &shared_mem[blockDim.x];
    
    // Find max value (only for valid positions)
    T local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        // Causal mask: can only attend to key positions <= query position
        if (i <= query_pos) {
            T val = Index(input, row, i);
            if (val > local_max) {
                local_max = val;
            }
        }
    }
    
    max_shared[tid] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (max_shared[tid + stride] > max_shared[tid]) {
                max_shared[tid] = max_shared[tid + stride];
            }
        }
        __syncthreads();
    }
    
    T max_val = max_shared[0];
    
    // Compute exp and sum (with mask)
    T local_sum = 0;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        if (i <= query_pos) {
            T val = exp(Index(input, row, i) - max_val);
            Index(output, row, i) = val;
            local_sum += val;
        } else {
            Index(output, row, i) = 0;  // Masked positions
        }
    }
    
    sum_shared[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }
    
    T sum_val = sum_shared[0];
    
    // Normalize
    for (int i = tid; i < seq_len; i += blockDim.x) {
        if (i <= query_pos) {
            Index(output, row, i) = Index(output, row, i) / sum_val;
        }
    }
}

template <typename T>
void op_softmax(
    const Tensor<T>& input,
    Tensor<T>& output,
    bool causal = false,
    int seq_offset = 0
) {
    if (input.h != output.h || input.w != output.w) {
        throw std::runtime_error("Softmax: input/output shape mismatch");
    }
    if (!input.on_device || !output.on_device) {
        throw std::runtime_error("Softmax: only GPU implementation available");
    }
    
    int threads = 256;
    int blocks = input.h;
    size_t shared_mem_size = threads * 2 * sizeof(T);
    
    if (causal) {
        op_softmax_causal_kernel<<<blocks, threads, shared_mem_size>>>(
            input, output, seq_offset
        );
    } else {
        op_softmax_kernel<<<blocks, threads, shared_mem_size>>>(
            input, output
        );
    }
}

// Backward pass for softmax
// If y = softmax(x), then:
// dy/dx_i = y_i * (dy_i - sum_j(y_j * dy_j))
template <typename T>
__global__ void op_softmax_backward_kernel(
    const Tensor<T> output,       // softmax output from forward pass
    const Tensor<T> grad_output,  // gradient w.r.t. output
    Tensor<T> grad_input          // gradient w.r.t. input
) {
    int row = blockIdx.x;
    if (row >= output.h) return;
    
    int tid = threadIdx.x;
    int seq_len = output.w;
    
    extern __shared__ T shared_mem[];
    
    // Compute sum of (y_j * dy_j) for this row
    T local_sum = 0;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_sum += Index(output, row, i) * Index(grad_output, row, i);
    }
    
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    T sum_y_dy = shared_mem[0];
    
    // Compute gradient: y_i * (dy_i - sum)
    for (int i = tid; i < seq_len; i += blockDim.x) {
        T y_i = Index(output, row, i);
        T dy_i = Index(grad_output, row, i);
        Index(grad_input, row, i) = y_i * (dy_i - sum_y_dy);
    }
}

template <typename T>
void op_softmax_backward(
    const Tensor<T>& output,
    const Tensor<T>& grad_output,
    Tensor<T>& grad_input
) {
    if (output.h != grad_output.h || output.w != grad_output.w) {
        throw std::runtime_error("Softmax backward: shape mismatch");
    }
    if (!output.on_device) {
        throw std::runtime_error("Softmax backward: only GPU implementation available");
    }
    
    int threads = 256;
    int blocks = output.h;
    size_t shared_mem_size = threads * sizeof(T);
    
    op_softmax_backward_kernel<<<blocks, threads, shared_mem_size>>>(
        output, grad_output, grad_input
    );
}
