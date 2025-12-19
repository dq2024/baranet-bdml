#pragma once

#include "utils/tensor.cuh"
#include <cuda_runtime.h>

// SiLU (Swish) activation: y = x * sigmoid(x) = x / (1 + exp(-x))
template <typename T>
class SiLUFunc {
public:
    __host__ __device__ T operator()(T x) {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        return x / ((T)1.0 + exp(-x));
    }
};

// Backward pass for SiLU
// d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
template <typename T>
class SiLUBackFunc {
public:
    __host__ __device__ T operator()(T x, T grad_output) {
        T sigmoid_x = (T)1.0 / ((T)1.0 + exp(-x));
        T grad_sigmoid = sigmoid_x * ((T)1.0 - sigmoid_x);
        // d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid'(x)
        //                      = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        T grad_input = sigmoid_x + x * grad_sigmoid;
        return grad_output * grad_input;
    }
};

// Forward pass using existing elementwise infrastructure
template <typename T>
void op_silu(const Tensor<T>& input, Tensor<T>& output) {
    if (input.h != output.h || input.w != output.w) {
        throw std::runtime_error("SiLU: input/output shape mismatch");
    }
    if (input.on_device != output.on_device) {
        throw std::runtime_error("SiLU: device mismatch");
    }
    
    SiLUFunc<T> f;
    if (input.on_device) {
        op_elemwise_unary_gpu(f, input, output);
    } else {
        op_elemwise_unary_cpu(f, input, output);
    }
}

// Backward pass
template <typename T>
void op_silu_backward(
    const Tensor<T>& input,
    const Tensor<T>& grad_output,
    Tensor<T>& grad_input
) {
    if (input.h != grad_output.h || input.w != grad_output.w) {
        throw std::runtime_error("SiLU backward: shape mismatch");
    }
    if (input.h != grad_input.h || input.w != grad_input.w) {
        throw std::runtime_error("SiLU backward: grad_input shape mismatch");
    }
    if (input.on_device != grad_output.on_device || input.on_device != grad_input.on_device) {
        throw std::runtime_error("SiLU backward: device mismatch");
    }
    
    SiLUBackFunc<T> f;
    if (input.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, input, grad_output, grad_input);
    } else {
        op_elemwise_binary_w_bcast_cpu(f, input, grad_output, grad_input);
    }
}
