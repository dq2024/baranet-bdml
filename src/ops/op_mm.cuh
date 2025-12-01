#pragma once

#define MM_BLOCK_DIM 32 

#include "utils/check_error.cuh"
#include "utils/tensor.cuh"
template <typename AT, typename BT, typename OT>
static void ensure_mm_shape_device(const Tensor<AT> &a, const Tensor<BT> &b, const Tensor<OT> &out)
{
    if (a.h != out.h || b.w != out.w || a.w != b.h)
        throw std::runtime_error("a,b,out tensor shape mismatch a:" +
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());

    if (a.on_device != b.on_device || a.on_device != out.on_device)
        throw std::runtime_error("a,b,out tensor device mismatch a:" + 
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());
}

template <typename T>
__global__ void op_mm_kernel(Tensor<T> A, Tensor<T> B, Tensor<T> C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < C.h && col < C.w) {
        T sum = (T)0;
        for (int k = 0; k < A.w; k++) {
            sum += Index(A, row, k) * Index(B, k, col);
        }
        Index(C, row, col) = sum;
    }
}

//compute C = A@B
template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    ensure_mm_shape_device(A,B,C);
    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    dim3 block_size(MM_BLOCK_DIM, MM_BLOCK_DIM);
        dim3 grid_size((C.w + block_size.x - 1) / block_size.x, (C.h + block_size.y - 1) / block_size.y);
        
        op_mm_kernel<<<grid_size, block_size>>>(A, B, C);
}
