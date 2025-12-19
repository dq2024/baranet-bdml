#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "llama/llama_config.hpp"
#include "llama/llama_model.hpp"

namespace py = pybind11;

// Helper to copy numpy array to Tensor<uint32_t>
void copy_uint32_from_numpy(py::array_t<uint32_t> arr, Tensor<uint32_t>& tensor) {
    auto buf = arr.request();
    int batch_size = buf.shape[0];
    
    Tensor<uint32_t> host_tensor(batch_size, 1, false);
    uint32_t* ptr = static_cast<uint32_t*>(buf.ptr);
    for (int i = 0; i < batch_size; i++) {
        Index(host_tensor, i, 0) = ptr[i];
    }
    
    if (tensor.on_device) {
        host_tensor.toDevice(tensor);
    } else {
        for (int i = 0; i < batch_size; i++) {
            Index(tensor, i, 0) = Index(host_tensor, i, 0);
        }
    }
}

template <typename T>
class PyLLaMAModel {
public:
    PyLLaMAModel(const llama::LLaMAConfig& config, bool on_device = true)
        : model_(config, on_device), on_device_(on_device) {}
    
    py::array forward(py::array_t<uint32_t> input_ids) {
        auto buf = input_ids.request();
        int batch_size = buf.shape[0];
        
        Tensor<uint32_t> input(batch_size, 1, on_device_);
        copy_uint32_from_numpy(input_ids, input);
        
        int vocab_size = model_.config().vocab_size;
        Tensor<T> logits(batch_size, vocab_size, on_device_);
        
        model_.forward(input, logits);
        
        Tensor<T> logits_host = logits.toHost();
        py::array_t<T> result({batch_size, vocab_size});
        T* result_ptr = result.mutable_data();
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < vocab_size; j++) {
                result_ptr[i * vocab_size + j] = Index(logits_host, i, j);
            }
        }
        
        return result;
    }
    
    float forward_timed(py::array_t<uint32_t> input_ids) {
        auto buf = input_ids.request();
        int batch_size = buf.shape[0];
        
        Tensor<uint32_t> input(batch_size, 1, on_device_);
        copy_uint32_from_numpy(input_ids, input);
        
        int vocab_size = model_.config().vocab_size;
        Tensor<T> logits(batch_size, vocab_size, on_device_);
        
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
    bool on_device_;
};

void bind_llama_model(py::module_ &m) {
    py::class_<llama::LLaMAConfig>(m, "LLaMAConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &llama::LLaMAConfig::vocab_size)
        .def_readwrite("hidden_size", &llama::LLaMAConfig::hidden_size)
        .def_readwrite("intermediate_size", &llama::LLaMAConfig::intermediate_size)
        .def_readwrite("num_hidden_layers", &llama::LLaMAConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &llama::LLaMAConfig::num_attention_heads)
        .def_readwrite("num_key_value_heads", &llama::LLaMAConfig::num_key_value_heads)
        .def("head_dim", &llama::LLaMAConfig::head_dim)
        .def("validate", &llama::LLaMAConfig::validate)
        .def("count_parameters", &llama::LLaMAConfig::count_parameters)
        .def_static("tinyllama_1_1b", &llama::LLaMAConfig::tinyllama_1_1b)
        .def("__repr__", &llama::LLaMAConfig::to_string);
    
    py::class_<PyLLaMAModel<float>>(m, "LLaMAModelCpp")
        .def(py::init<const llama::LLaMAConfig&, bool>(),
             py::arg("config"), py::arg("on_device") = true)
        .def("forward", &PyLLaMAModel<float>::forward)
        .def("forward_timed", &PyLLaMAModel<float>::forward_timed)
        .def("benchmark", &PyLLaMAModel<float>::benchmark,
             py::arg("batch_size"), py::arg("num_warmup") = 5, 
             py::arg("num_iterations") = 100)
        .def("count_parameters", &PyLLaMAModel<float>::count_parameters)
        .def("get_gpu_memory_usage", &PyLLaMAModel<float>::get_gpu_memory_usage);
}