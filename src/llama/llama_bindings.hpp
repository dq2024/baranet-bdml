/**
 * Python bindings for C++ LLaMA model
 * Add this to your bindings.cu file
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "llama_config.hpp"
#include "llama_model.hpp"

namespace py = pybind11;

// Wrapper for C++ LLaMAModel
template <typename T>
class PyLLaMAModel {
public:
    PyLLaMAModel(const llama::LLaMAConfig& config, bool on_device = true)
        : model_(config, on_device) {}
    
    py::array forward(py::array_t<uint32_t> input_ids) {
        auto buf = input_ids.request();
        int batch_size = buf.shape[0];
        
        // Create input tensor
        Tensor<uint32_t> input(batch_size, 1, model_.config().on_device);
        input.copy_from_numpy(input_ids);
        
        // Create output tensor
        int vocab_size = model_.config().vocab_size;
        Tensor<T> logits(batch_size, vocab_size, model_.config().on_device);
        
        // Forward pass
        model_.forward(input, logits);
        
        // Convert to numpy
        return logits.to_numpy();
    }
    
    float forward_timed(py::array_t<uint32_t> input_ids) {
        auto buf = input_ids.request();
        int batch_size = buf.shape[0];
        
        Tensor<uint32_t> input(batch_size, 1, model_.config().on_device);
        input.copy_from_numpy(input_ids);
        
        int vocab_size = model_.config().vocab_size;
        Tensor<T> logits(batch_size, vocab_size, model_.config().on_device);
        
        return model_.forward_timed(input, logits);
    }
    
    py::dict benchmark(int batch_size, int num_warmup = 5, int num_iterations = 100) {
        auto result = llama::benchmark_model(model_, batch_size, num_warmup, num_iterations);
        
        py::dict d;
        d["forward_time_ms"] = result.forward_time_ms;
        d["tokens_per_second"] = result.tokens_per_second;
        d["memory_used_bytes"] = result.memory_used_bytes;
        d["num_parameters"] = result.num_parameters;
        return d;
    }
    
    size_t count_parameters() const {
        return model_.count_parameters();
    }
    
    size_t get_gpu_memory_usage() const {
        return model_.get_gpu_memory_usage();
    }
    
private:
    llama::LLaMAModel<T> model_;
};

// Bind to Python
void bind_llama_model(py::module_ &m) {
    // Config
    py::class_<llama::LLaMAConfig>(m, "LLaMAConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &llama::LLaMAConfig::vocab_size)
        .def_readwrite("hidden_size", &llama::LLaMAConfig::hidden_size)
        .def_readwrite("intermediate_size", &llama::LLaMAConfig::intermediate_size)
        .def_readwrite("num_hidden_layers", &llama::LLaMAConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &llama::LLaMAConfig::num_attention_heads)
        .def_readwrite("num_key_value_heads", &llama::LLaMAConfig::num_key_value_heads)
        .def_readwrite("max_position_embeddings", &llama::LLaMAConfig::max_position_embeddings)
        .def_readwrite("rms_norm_eps", &llama::LLaMAConfig::rms_norm_eps)
        .def_readwrite("rope_theta", &llama::LLaMAConfig::rope_theta)
        .def("head_dim", &llama::LLaMAConfig::head_dim)
        .def("num_kv_groups", &llama::LLaMAConfig::num_kv_groups)
        .def("validate", &llama::LLaMAConfig::validate)
        .def("count_parameters", &llama::LLaMAConfig::count_parameters)
        .def_static("tinyllama_1_1b", &llama::LLaMAConfig::tinyllama_1_1b)
        .def("__repr__", &llama::LLaMAConfig::to_string);
    
    // Model
    py::class_<PyLLaMAModel<float>>(m, "LLaMAModelCpp")
        .def(py::init<const llama::LLaMAConfig&, bool>(),
             py::arg("config"), py::arg("on_device") = true,
             "Create LLaMA model (C++ implementation)")
        .def("forward", &PyLLaMAModel<float>::forward,
             py::arg("input_ids"),
             "Forward pass: input_ids -> logits")
        .def("forward_timed", &PyLLaMAModel<float>::forward_timed,
             py::arg("input_ids"),
             "Timed forward pass, returns time in ms")
        .def("benchmark", &PyLLaMAModel<float>::benchmark,
             py::arg("batch_size"), py::arg("num_warmup") = 5, 
             py::arg("num_iterations") = 100,
             "Run benchmark and return results dict")
        .def("count_parameters", &PyLLaMAModel<float>::count_parameters,
             "Count total parameters")
        .def("get_gpu_memory_usage", &PyLLaMAModel<float>::get_gpu_memory_usage,
             "Get current GPU memory usage in bytes");
}

// Add to your PYBIND11_MODULE:
// PYBIND11_MODULE(bten, m) {
//     // ... existing bindings ...
//     bind_llama_model(m);
// }
