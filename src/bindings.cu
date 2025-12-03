#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ops/op_elemwise.cuh"
#include "py_tensor_shim.hh"
#include "llama/llama_bindings.hpp"

namespace py = pybind11;

// Define the extern variable declared in op_elemwise.cuh
unsigned long long randgen_seed = 42;

PYBIND11_MODULE(bten, m) {
  m.doc() = "Python bindings for Barenet (float32 or uint32 only for now)";

  // Expose randgen_seed function
  m.def("randgen_seed", [](unsigned long long seed) {
      ::randgen_seed = seed;
  }, "Set random seed");

  // Bind tensor types
  bind_tensor_type<float>(m, "Tensor");
  bind_tensor_type<uint32_t>(m, "TensorU32");
  
  // Bind LLaMA model
  bind_llama_model(m);
}