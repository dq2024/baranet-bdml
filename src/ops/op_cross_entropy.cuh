#pragma once
#include "utils/tensor.cuh"


template <typename T, typename S>
__global__ void op_cross_entropy_kernel(const Tensor<T> logits, const Tensor<S> targets,
                               Tensor<T> d_logits, Tensor<T> losses)
{
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < logits.h) {

        T max_logit = Index(logits, batch_idx, 0);
        for (int i = 1; i < logits.w; i++) {
            T val = Index(logits, batch_idx, i);
            if (val > max_logit) {
                max_logit = val;
            }
        }

        T sum = 0;
        for (int i = 0; i < logits.w; i++) {
            sum += exp(Index(logits, batch_idx, i)- max_logit);
        }

        S target = Index(targets, batch_idx, 0);

        T logit = Index(logits, batch_idx, target);
        Index(losses, batch_idx, 0) = -logit + max_logit + log(sum);

        for (int j = 0; j < logits.w; j++) {
            T softmax_val = exp(Index(logits, batch_idx, j) - max_logit) / sum;
            T gradient = softmax_val;

            if (j == target) {
                gradient -= 1;
            }
            T inv = 1.0 / logits.h;
            gradient *= inv;
            
            Index(d_logits, batch_idx, j) = gradient;
        }
    }
}

//This function calculates the cross_entropy loss from the "logits" tensor for a batch of training innput
//and the batch's corresponding "target" label tensor and returns the average loss of the batch.
//It also returns the gradient of the logits tensor.
template <typename T, typename S>
T op_cross_entropy_loss(const Tensor<T> &logits, const Tensor<S> &targets,
                               Tensor<T> &d_logits)
{
    if (logits.h != d_logits.h || logits.w != d_logits.w)
    {
        throw std::runtime_error("op_cross_entropy_loss: d_logits shape mismatch");
    }

    if (targets.h != logits.h || targets.w != 1)
    {
        throw std::runtime_error("op_cross_entropy_loss: targets shape mismatch");
    }
    if (logits.on_device != d_logits.on_device || logits.on_device != targets.on_device)
    {
        throw std::runtime_error("op_cross_entropy_loss: device mismatch");
    }

    //Lab-1: please add your code here
    //You need to define separate GPU kernel function(s) and launch them here
    //In order to calculate d_logits, you should derive what its values should be 
    //symbolically.
    Tensor<T> losses(logits.h, 1, true);


    dim3 block_size(32);
    dim3 grid_size((logits.h + block_size.x - 1) / block_size.x);

    op_cross_entropy_kernel<<<grid_size, block_size>>>(logits, targets, d_logits, losses);

    Tensor<T> total_loss_tensor(1, 1, true);
    Tensor<int> dummy(1, 1, true);
    SumAccumFunc<T, int> sum_func;
    op_reduction_gpu(sum_func, losses, total_loss_tensor, dummy, false);

    T loss;
    cudaMemcpy(&loss, total_loss_tensor.rawp, sizeof(T), cudaMemcpyDeviceToHost);

    return loss / logits.h;

    
    
}
