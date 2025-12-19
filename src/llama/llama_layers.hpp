#pragma once

#include "utils/tensor.cuh"
#include "llama_config.hpp"
#include "ops/op_rmsnorm.cuh"
#include "ops/op_rope.cuh"
#include "ops/op_silu.cuh"
#include "ops/op_softmax.cuh"
#include "ops/op_mm.cuh"
#include <vector>
#include <memory>

namespace llama {

/**
 * RMSNorm layer
 */
template <typename T>
class RMSNorm {
public:
    RMSNorm(int hidden_size, float eps, bool on_device)
        : hidden_size_(hidden_size), eps_(eps), on_device_(on_device),
          weight_(1, hidden_size, on_device) {
        // Initialize weight to ones
        op_const_fill(weight_, (T)1.0);
    }
    
    void forward(const Tensor<T>& input, Tensor<T>& output) {
        op_rmsnorm(input, weight_, output, eps_);
    }
    
    Tensor<T>& weight() { return weight_; }
    const Tensor<T>& weight() const { return weight_; }
    
private:
    int hidden_size_;
    float eps_;
    bool on_device_;
    Tensor<T> weight_;
};

/**
 * Grouped-Query Attention layer
 */
template <typename T>
class GroupedQueryAttention {
public:
    GroupedQueryAttention(const LLaMAConfig& config, bool on_device)
        : config_(config), on_device_(on_device),
          q_proj_(config.hidden_size, config.hidden_size, on_device),
          k_proj_(config.hidden_size, config.num_key_value_heads * config.head_dim(), on_device),
          v_proj_(config.hidden_size, config.num_key_value_heads * config.head_dim(), on_device),
          o_proj_(config.hidden_size, config.hidden_size, on_device),
          scale_(1.0f / std::sqrt(config.head_dim())) {
    }
    
    void forward(
        const Tensor<T>& hidden_states,
        Tensor<T>& output,
        int position_offset = 0,
        bool use_causal_mask = true
    ) {
        int batch_size = hidden_states.h;
        int hidden_size = config_.hidden_size;
        int kv_size = config_.num_key_value_heads * config_.head_dim();
        
        // Allocate workspace tensors
        Tensor<T> query(batch_size, hidden_size, on_device_);
        Tensor<T> key(batch_size, kv_size, on_device_);
        Tensor<T> value(batch_size, kv_size, on_device_);
        
        // Q, K, V projections
        op_mm(hidden_states, q_proj_, query);
        op_mm(hidden_states, k_proj_, key);
        op_mm(hidden_states, v_proj_, value);
        
        // Apply RoPE to Q and K
        Tensor<T> query_rope(batch_size, hidden_size, on_device_);
        Tensor<T> key_rope(batch_size, kv_size, on_device_);
        op_rope(query, query_rope, position_offset, config_.head_dim(), config_.rope_theta);
        op_rope(key, key_rope, position_offset, config_.head_dim(), config_.rope_theta);
        
        // Repeat K,V for GQA (if needed)
        Tensor<T> key_repeated(batch_size, hidden_size, on_device_);
        Tensor<T> value_repeated(batch_size, hidden_size, on_device_);
        
        if (config_.num_kv_groups() > 1) {
            repeat_kv(key_rope, key_repeated, config_.num_kv_groups());
            repeat_kv(value, value_repeated, config_.num_kv_groups());
        } else {
            key_repeated = key_rope;
            value_repeated = value;
        }
        
        // Attention scores: Q @ K^T
        Tensor<T> key_t = key_repeated.transpose();
        Tensor<T> attn_scores(batch_size, batch_size, on_device_);
        op_mm(query_rope, key_t, attn_scores);
        
        // Scale scores
        op_multiply(attn_scores, (T)scale_, attn_scores);
        
        // Softmax with optional causal mask
        Tensor<T> attn_probs(batch_size, batch_size, on_device_);
        op_softmax(attn_scores, attn_probs, use_causal_mask, position_offset);
        
        // Attention output: attn_probs @ V
        Tensor<T> attn_out(batch_size, hidden_size, on_device_);
        op_mm(attn_probs, value_repeated, attn_out);
        
        // Output projection
        op_mm(attn_out, o_proj_, output);
    }
    
    // Weight accessors
    Tensor<T>& q_proj() { return q_proj_; }
    Tensor<T>& k_proj() { return k_proj_; }
    Tensor<T>& v_proj() { return v_proj_; }
    Tensor<T>& o_proj() { return o_proj_; }
    
private:
    const LLaMAConfig& config_;
    bool on_device_;
    Tensor<T> q_proj_;
    Tensor<T> k_proj_;
    Tensor<T> v_proj_;
    Tensor<T> o_proj_;
    T scale_;
    
    // Helper: repeat K/V heads for GQA
    void repeat_kv(const Tensor<T>& kv, Tensor<T>& repeated, int n_rep) {
        if (n_rep == 1) {
            // Just copy
            repeated = kv;
            return;
        }
        
        // TODO: Implement efficient repeat on GPU
        // For now, do it on CPU (slow but correct)
        int batch_size = kv.h;
        int kv_size = kv.w;
        int head_dim = config_.head_dim();
        int num_kv_heads = kv_size / head_dim;
        
        // Copy to host
        Tensor<T> kv_host(batch_size, kv_size, false);
        Tensor<T> repeated_host(batch_size, num_kv_heads * n_rep * head_dim, false);
        
        if (kv.on_device) {
            kv.toHost(kv_host);
        } else {
            kv_host = kv;
        }
        
        // Repeat on CPU
        for (int b = 0; b < batch_size; b++) {
            for (int kv_h = 0; kv_h < num_kv_heads; kv_h++) {
                for (int r = 0; r < n_rep; r++) {
                    int dst_head = kv_h * n_rep + r;
                    for (int d = 0; d < head_dim; d++) {
                        int src_idx = kv_h * head_dim + d;
                        int dst_idx = dst_head * head_dim + d;
                        Index(repeated_host, b, dst_idx) = Index(kv_host, b, src_idx);
                    }
                }
            }
        }
        
        // Copy back to device
        if (on_device_) {
            repeated_host.toDevice(repeated);
        } else {
            repeated = repeated_host;
        }
    }
};

/**
 * MLP with SwiGLU activation
 */
template <typename T>
class MLP {
public:
    MLP(const LLaMAConfig& config, bool on_device)
        : config_(config), on_device_(on_device),
          gate_proj_(config.hidden_size, config.intermediate_size, on_device),
          up_proj_(config.hidden_size, config.intermediate_size, on_device),
          down_proj_(config.intermediate_size, config.hidden_size, on_device) {
    }
    
    void forward(const Tensor<T>& input, Tensor<T>& output) {
        int batch_size = input.h;
        int intermediate_size = config_.intermediate_size;
        
        // Gate and Up projections
        Tensor<T> gate_out(batch_size, intermediate_size, on_device_);
        Tensor<T> up_out(batch_size, intermediate_size, on_device_);
        
        op_mm(input, gate_proj_, gate_out);
        op_mm(input, up_proj_, up_out);
        
        // SiLU activation on gate
        Tensor<T> gate_activated(batch_size, intermediate_size, on_device_);
        op_silu(gate_out, gate_activated);
        
        // Element-wise multiply
        Tensor<T> intermediate(batch_size, intermediate_size, on_device_);
        op_multiply(gate_activated, up_out, intermediate);
        
        // Down projection
        op_mm(intermediate, down_proj_, output);
    }
    
    // Weight accessors
    Tensor<T>& gate_proj() { return gate_proj_; }
    Tensor<T>& up_proj() { return up_proj_; }
    Tensor<T>& down_proj() { return down_proj_; }
    
private:
    const LLaMAConfig& config_;
    bool on_device_;
    Tensor<T> gate_proj_;
    Tensor<T> up_proj_;
    Tensor<T> down_proj_;
};

/**
 * Single LLaMA decoder layer
 */
template <typename T>
class LLaMADecoderLayer {
public:
    LLaMADecoderLayer(const LLaMAConfig& config, bool on_device)
        : config_(config), on_device_(on_device),
          input_layernorm_(config.hidden_size, config.rms_norm_eps, on_device),
          self_attn_(config, on_device),
          post_attention_layernorm_(config.hidden_size, config.rms_norm_eps, on_device),
          mlp_(config, on_device) {
    }
    
    void forward(
        const Tensor<T>& hidden_states,
        Tensor<T>& output,
        int position_offset = 0,
        bool use_causal_mask = true
    ) {
        int batch_size = hidden_states.h;
        int hidden_size = config_.hidden_size;
        
        // Pre-attention norm
        Tensor<T> normed(batch_size, hidden_size, on_device_);
        input_layernorm_.forward(hidden_states, normed);
        
        // Self-attention
        Tensor<T> attn_out(batch_size, hidden_size, on_device_);
        self_attn_.forward(normed, attn_out, position_offset, use_causal_mask);
        
        // Residual connection
        Tensor<T> after_attn(batch_size, hidden_size, on_device_);
        op_add(hidden_states, attn_out, after_attn);
        
        // Pre-MLP norm
        Tensor<T> normed2(batch_size, hidden_size, on_device_);
        post_attention_layernorm_.forward(after_attn, normed2);
        
        // MLP
        Tensor<T> mlp_out(batch_size, hidden_size, on_device_);
        mlp_.forward(normed2, mlp_out);
        
        // Residual connection
        op_add(after_attn, mlp_out, output);
    }
    
    // Component accessors
    RMSNorm<T>& input_layernorm() { return input_layernorm_; }
    GroupedQueryAttention<T>& self_attn() { return self_attn_; }
    RMSNorm<T>& post_attention_layernorm() { return post_attention_layernorm_; }
    MLP<T>& mlp() { return mlp_; }
    
private:
    const LLaMAConfig& config_;
    bool on_device_;
    RMSNorm<T> input_layernorm_;
    GroupedQueryAttention<T> self_attn_;
    RMSNorm<T> post_attention_layernorm_;
    MLP<T> mlp_;
};

} // namespace llama
