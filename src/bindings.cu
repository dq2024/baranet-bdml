#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "llama_bindings.hpp"

namespace py = pybind11;

#include "py_tensor_shim.hh"

PYBIND11_MODULE(bten, m) {
  m.doc() = "Python bindings for Barenet (float32 or uint32 only for now)";

  bind_tensor_type<float>(m, "TensorF");
  bind_tensor_type<uint32_t>(m, "TensorU32");
  bind_llama_model(m);
  
}