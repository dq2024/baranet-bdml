#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "llama_bindings.hpp"
#include "ops/op_elemwise.cuh"

// Define the extern variable
unsigned long long randgen_seed = 42;

// ... rest of your code ...

PYBIND11_MODULE(bten, m) {
    // Add this line
    m.def("randgen_seed", [](unsigned long long seed) {
        randgen_seed = seed;
    }, "Set random seed");
    
    // ... rest of your existing bindings ...
    bind_tensor_type<float>(m, "Tensor");
    bind_tensor_type<uint32_t>(m, "TensorU32");
    bind_llama_model(m);
}


namespace py = pybind11;

#include "py_tensor_shim.hh"

PYBIND11_MODULE(bten, m) {
  m.doc() = "Python bindings for Barenet (float32 or uint32 only for now)";

  bind_tensor_type<float>(m, "TensorF");
  bind_tensor_type<uint32_t>(m, "TensorU32");
  bind_llama_model(m);
  
}