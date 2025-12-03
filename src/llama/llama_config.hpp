#pragma once

#include <string>
#include <stdexcept>

namespace llama {

/**
 * Configuration for LLaMA models (matches TinyLLaMA 1.1B)
 */
struct LLaMAConfig {
    // Model dimensions
    int vocab_size = 32000;
    int hidden_size = 2048;
    int intermediate_size = 5632;  // FFN hidden dimension
    int num_hidden_layers = 22;
    int num_attention_heads = 32;
    int num_key_value_heads = 4;   // For Grouped-Query Attention
    
    // Context and positional
    int max_position_embeddings = 2048;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    
    // Derived dimensions
    int head_dim() const {
        return hidden_size / num_attention_heads;
    }
    
    int num_kv_groups() const {
        return num_attention_heads / num_key_value_heads;
    }
    
    // Token IDs
    int pad_token_id = 0;
    int bos_token_id = 1;
    int eos_token_id = 2;
    
    // Validate configuration
    void validate() const {
        if (hidden_size % num_attention_heads != 0) {
            throw std::runtime_error(
                "hidden_size must be divisible by num_attention_heads");
        }
        if (num_attention_heads % num_key_value_heads != 0) {
            throw std::runtime_error(
                "num_attention_heads must be divisible by num_key_value_heads");
        }
    }
    
    // Factory methods
    static LLaMAConfig tinyllama_1_1b() {
        LLaMAConfig config;
        // Default values already match TinyLLaMA 1.1B
        config.validate();
        return config;
    }
    
    static LLaMAConfig from_sizes(int vocab, int hidden, int layers, int heads) {
        LLaMAConfig config;
        config.vocab_size = vocab;
        config.hidden_size = hidden;
        config.num_hidden_layers = layers;
        config.num_attention_heads = heads;
        config.num_key_value_heads = heads / 4;  // Assume GQA ratio of 4
        config.intermediate_size = hidden * 3;    // Rough estimate
        config.validate();
        return config;
    }
    
    // Print configuration
    std::string to_string() const {
        return "LLaMAConfig(\n"
               "  vocab_size=" + std::to_string(vocab_size) + ",\n"
               "  hidden_size=" + std::to_string(hidden_size) + ",\n"
               "  num_layers=" + std::to_string(num_hidden_layers) + ",\n"
               "  num_heads=" + std::to_string(num_attention_heads) + ",\n"
               "  num_kv_heads=" + std::to_string(num_key_value_heads) + ",\n"
               "  intermediate_size=" + std::to_string(intermediate_size) + "\n"
               ")";
    }
    
    // Count total parameters (approximate)
    size_t count_parameters() const {
        size_t params = 0;
        
        // Embeddings
        params += vocab_size * hidden_size;
        
        // Each layer
        size_t per_layer = 0;
        per_layer += hidden_size * hidden_size;  // Q proj
        per_layer += hidden_size * (num_key_value_heads * head_dim());  // K proj
        per_layer += hidden_size * (num_key_value_heads * head_dim());  // V proj
        per_layer += hidden_size * hidden_size;  // O proj
        per_layer += hidden_size * intermediate_size;  // Gate proj
        per_layer += hidden_size * intermediate_size;  // Up proj
        per_layer += intermediate_size * hidden_size;  // Down proj
        per_layer += hidden_size * 2;  // RMSNorm weights (2 per layer)
        
        params += per_layer * num_hidden_layers;
        
        // Final norm and LM head
        params += hidden_size;  // Final norm
        params += hidden_size * vocab_size;  // LM head
        
        return params;
    }
};

} // namespace llama
