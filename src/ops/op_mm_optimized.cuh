#pragma once

#include "utils/tensor.cuh"
#include "ops/op_mm.cuh"
#include <cuda_runtime.h>

#define MM_TILE_SIZE 16

/**
 * Optimized Matrix Multiplication with Shared Memory Tiling
 * Reduces global memory accesses by using shared memory for tiles
 */
template <typename T>
__global__ void op_mm_tiled_kernel(
    const Tensor<T> A,
    const Tensor<T> B,
    Tensor<T> C
) {
    // Tile indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Shared memory for tiles
    __shared__ T As[MM_TILE_SIZE][MM_TILE_SIZE];
    __shared__ T Bs[MM_TILE_SIZE][MM_TILE_SIZE];
    
    int row = by * MM_TILE_SIZE + ty;
    int col = bx * MM_TILE_SIZE + tx;
    
    T sum = 0.0;
    
    // Process tiles
    for (int tile = 0; tile < (A.w + MM_TILE_SIZE - 1) / MM_TILE_SIZE; tile++) {
        // Load tile from A
        int a_row = row;
        int a_col = tile * MM_TILE_SIZE + tx;
        if (a_row < A.h && a_col < A.w) {
            As[ty][tx] = Index(A, a_row, a_col);
        } else {
            As[ty][tx] = 0.0;
        }
        
        // Load tile from B
        int b_row = tile * MM_TILE_SIZE + ty;
        int b_col = col;
        if (b_row < B.h && b_col < B.w) {
            Bs[ty][tx] = Index(B, b_row, b_col);
        } else {
            Bs[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < MM_TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < C.h && col < C.w) {
        Index(C, row, col) = sum;
    }
}

/**
 * Optimized matrix multiplication using tiling
 */
template <typename T>
void op_mm_optimized(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    if (A.h != C.h || B.w != C.w || A.w != B.h) {
        throw std::runtime_error("Matrix multiply: shape mismatch");
    }
    if (!A.on_device || !B.on_device || !C.on_device) {
        throw std::runtime_error("Matrix multiply: only GPU implementation available");
    }
    
    dim3 block_size(MM_TILE_SIZE, MM_TILE_SIZE);
    dim3 grid_size((C.w + MM_TILE_SIZE - 1) / MM_TILE_SIZE,
                   (C.h + MM_TILE_SIZE - 1) / MM_TILE_SIZE);
    
    op_mm_tiled_kernel<<<grid_size, block_size>>>(A, B, C);
}

/**
 * Fused: RMSNorm + Residual Addition
 * Reduces memory traffic by combining operations
 */
template <typename T>
__global__ void op_rmsnorm_residual_kernel(
    const Tensor<T> input,
    const Tensor<T> residual,
    const Tensor<T> weight,
    Tensor<T> output,
    T eps
) {
    int row = blockIdx.x;
    if (row >= input.h) return;
    
    int tid = threadIdx.x;
    int hidden_dim = input.w;
    
    extern __shared__ T shared_mem[];
    
    // Compute sum of squares
    T sum_sq = 0;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        T val = Index(input, row, i);
        sum_sq += val * val;
    }
    
    shared_mem[tid] = sum_sq;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    T mean_sq = shared_mem[0] / hidden_dim;
    T rms = sqrt(mean_sq + eps);
    T inv_rms = 1.0 / rms;
    
    // Normalize, apply weight, and add residual in one pass
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        T normalized = Index(input, row, i) * inv_rms;
        T weighted = normalized * Index(weight, 0, i);
        T res = Index(residual, row, i);
        Index(output, row, i) = weighted + res;
    }
}

template <typename T>
void op_rmsnorm_residual(
    const Tensor<T>& input,
    const Tensor<T>& residual,
    const Tensor<T>& weight,
    Tensor<T>& output,
    T eps = 1e-6
) {
    if (input.h != output.h || input.w != output.w) {
        throw std::runtime_error("RMSNorm+Residual: shape mismatch");
    }
    if (!input.on_device || !output.on_device) {
        throw std::runtime_error("RMSNorm+Residual: only GPU implementation available");
    }
    
    int threads = 256;
    int blocks = input.h;
    size_t shared_mem_size = threads * sizeof(T);
    
    op_rmsnorm_residual_kernel<<<blocks, threads, shared_mem_size>>>(
        input, residual, weight, output, eps
    );
}

