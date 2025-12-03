#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <type_traits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "utils/tensor.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_cross_entropy.cuh"
#include "ops/op_rmsnorm.cuh"
#include "ops/op_rope.cuh"
#include "ops/op_silu.cuh"
#include "ops/op_softmax.cuh"

namespace py = pybind11;

// map C++ type name to a short string for Python repr, only float32 or uint32 for now
template <typename T> inline const char* dtype_name();
template <> inline const char* dtype_name<float>()  { return "float32"; }
template <> inline const char* dtype_name<uint32_t>()   { return "uint32"; }

// Wrapper from Tensor<T> to a Python object
template <typename T>
class PyTensor {
public:
  Tensor<T> t;

  PyTensor(int h, int w, bool is_cuda)
  : t(h, w, is_cuda) {}  

  
  explicit PyTensor(const Tensor<T>& existing) : t(existing) {}          // copy view
  explicit PyTensor(Tensor<T>&& existing) : t(std::move(existing)) {}     // move view


  std::pair<int,int> shape() const { return {t.h, t.w}; }

  bool is_cuda() const { return t.on_device; }

  void copy_from_numpy(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
    if (arr.ndim() != 2) throw std::runtime_error("Expected 2D NumPy array");
    auto h = static_cast<int>(arr.shape(0));
    auto w = static_cast<int>(arr.shape(1));
    if (h != t.h || w != t.w) throw std::runtime_error("Shape mismatch");

    // allocate a host tensor
    Tensor<T> host_t(h, w, /*on_device=*/false);
    
    // memcpy from NumPy to host tensor's raw pointer
    std::memcpy(host_t.rawp, arr.data(), sizeof(T) * h * w);

    // Move host -> device or host based on whether it's cuda tensor or not
    if (t.on_device) {
      host_t.toDevice(t);
    } else {
      std::memcpy(t.rawp, host_t.rawp, sizeof(T) * h * w);
    }
  }

  // Copy to NumPy (this→host→NumPy)
  py::array to_numpy() const {
    Tensor<T> host_t(t.h, t.w, /*on_device=*/false);
    if (t.on_device) {
      t.toHost(host_t);
    } else {
      std::memcpy(host_t.rawp, t.rawp, sizeof(T) * t.h * t.w);
    }

    // Create NumPy array that owns its own memory (copy)
    py::array_t<T> out({t.h, t.w});
    std::memcpy(out.mutable_data(), host_t.rawp, sizeof(T) * t.h * t.w);
    return out;
  }

  void fill(T v) {
    op_const_fill(t, v);
  }

  PyTensor<T> transpose() const {
    return PyTensor<T>(t.transpose());
  }

  PyTensor<T> slice_view(int start_h, int end_h, int start_w, int end_w) const {
    return PyTensor<T>(t.slice(start_h, end_h, start_w, end_w));
  }
 
  // Interpret Python indexing syntax and return a sliced tensor view.
  PyTensor<T> getitem(py::object index) const {
    auto parse_dim = [&](py::handle obj, int dim_size) -> std::pair<int, int> {
      if (obj.is(py::ellipsis()) || obj.is_none()) {
        return {0, dim_size};
      }

      if (py::isinstance<py::slice>(obj)) {
        Py_ssize_t start = 0;
        Py_ssize_t stop = 0;
        Py_ssize_t step = 0;
        Py_ssize_t slicelength = 0;
        auto slice = obj.cast<py::slice>();
        if (!slice.compute(dim_size, &start, &stop, &step, &slicelength)) {
          throw py::error_already_set();
        }
        if (step != 1) {
          throw py::index_error("Tensor slicing only supports step=1");
        }
        return {static_cast<int>(start), static_cast<int>(stop)};
      }

      if (py::isinstance<py::int_>(obj)) {
        int idx = obj.cast<int>();
        if (idx < 0) {
          idx += dim_size;
        }
        if (idx < 0 || idx >= dim_size) {
          throw py::index_error("Index out of range");
        }
        return {idx, idx + 1};
      }

      throw py::type_error("Tensor indices must be integers, slices, or None");
    };

    py::tuple idx_tuple;
    if (py::isinstance<py::tuple>(index)) {
      idx_tuple = index.cast<py::tuple>();
    } else {
      idx_tuple = py::make_tuple(index);
    }

    if (idx_tuple.size() > 2) {
      throw py::index_error("Tensor only supports 2D indexing");
    }

    py::object first_index = idx_tuple.size() > 0
      ? py::reinterpret_borrow<py::object>(idx_tuple[0])
      : py::ellipsis();
    py::object second_index = idx_tuple.size() > 1
      ? py::reinterpret_borrow<py::object>(idx_tuple[1])
      : py::ellipsis();

    auto [start_h, end_h] = parse_dim(first_index, t.h);
    auto [start_w, end_w] = parse_dim(second_index, t.w);

    return slice_view(start_h, end_h, start_w, end_w);
  }

  std::string repr() const {
    return "dtype=" + std::string(dtype_name<T>()) + " " + t.repr();
  }

};

template <typename T>
void bind_tensor_type(py::module_ &m, const char* pyname) {
  using Self = PyTensor<T>;
  auto cls = py::class_<Self>(m, pyname)
    .def(py::init<int,int,bool>(), py::arg("h"), py::arg("w"), py::arg("is_cuda")=true)
    .def_property_readonly("shape", &Self::shape)
    .def_property_readonly("is_cuda", &Self::is_cuda)
    .def("to_numpy", &Self::to_numpy)
    .def("copy_from_numpy", &Self::copy_from_numpy)
    .def("fill", &Self::fill, py::arg("value"))
    .def("transpose", &Self::transpose)
    .def("slice", &Self::slice_view, py::arg("start_h"), py::arg("end_h"),
         py::arg("start_w"), py::arg("end_w"))
    .def_property_readonly("T", &Self::transpose)
    .def("__getitem__", &Self::getitem)
    .def("__repr__", &Self::repr)
    .def("__add__", [](const Self &me, const Self &other) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      op_add<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("__add__", [](const Self &me, T scalar) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      op_add<T>(me.t, scalar, out.t);
      return out;
    }, py::arg("scalar"))
    .def("__sub__", [](const Self &me, const Self &other) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      op_sub<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("__mul__", [](const Self &me, const Self &other) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      op_multiply<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("__mul__", [](const Self &me, T scalar) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      op_multiply<T>(me.t, scalar, out.t);
      return out;
    }, py::arg("scalar"))
    .def("__eq__", [](const Self &me, const Self &other) {
      PyTensor<uint32_t> out(me.t.h, me.t.w, me.t.on_device);
      op_equal<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("sum", [](const Self &me, int axis=0) {
      int out_h = me.t.h;
      int out_w = me.t.w;
      if (axis == 0) {
        out_h = 1;
      } else if (axis == 1) {
        out_w = 1;
      } else {
        throw std::runtime_error("Invalid axis, must be 0 or 1");
      }
      PyTensor<T> out(out_h, out_w, me.t.on_device);
      op_sum<T>(me.t, out.t);
      return out;
    }, py::arg("axis"))
    .def("argmax", [](const Self &me) {
      PyTensor<uint32_t> out(me.t.h, 1, me.t.on_device);
      op_argmax<T>(me.t, out.t);
      return out;
    })
    .def("__matmul__", [](const Self &me, const Self &other) {
      PyTensor<T> out(me.t.h, other.t.w, me.t.on_device);
      op_mm<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"));

  // FLOAT-ONLY OPERATIONS - add to the SAME class object
  if constexpr (std::is_floating_point_v<T>) {
    cls.def("relu", [](const Self &me) {
        PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
        op_relu<T>(me.t, out.t);
        return out;
      })
      .def("relu_back", [](const Self &me, const Self &dout) {
        PyTensor<T> din(me.t.h, me.t.w, me.t.on_device);
        op_relu_back<T>(me.t, dout.t, din.t);
        return din;
      }, py::arg("DOut"))
      .def("silu", [](const Self &me) {
        PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
        op_silu<T>(me.t, out.t);
        return out;
      })
      .def("silu_back", [](const Self &me, const Self &dout) {
        PyTensor<T> din(me.t.h, me.t.w, me.t.on_device);
        op_silu_backward<T>(me.t, dout.t, din.t);
        return din;
      }, py::arg("DOut"))
      .def("cross_entropy_loss", [](const Self &logits, const PyTensor<uint32_t> &labels, PyTensor<T> &d_logits) {
        return op_cross_entropy_loss<T,uint32_t>(logits.t, labels.t, d_logits.t);
      }, py::arg("labels"), py::arg("d_logits"))
      .def("rmsnorm", [](const Self &me, const Self &weight, T eps) {
        PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
        op_rmsnorm<T>(me.t, weight.t, out.t, eps);
        return out;
      }, py::arg("weight"), py::arg("eps")=1e-6f)
      .def("rmsnorm_back", [](const Self &me, const Self &weight, const Self &grad_out, 
                               PyTensor<T> &grad_weight, T eps) {
        PyTensor<T> grad_in(me.t.h, me.t.w, me.t.on_device);
        op_rmsnorm_backward<T>(me.t, weight.t, grad_out.t, grad_in.t, grad_weight.t, eps);
        return grad_in;
      }, py::arg("weight"), py::arg("grad_out"), py::arg("grad_weight"), py::arg("eps")=1e-6f)
      .def("rope", [](const Self &me, int position_offset, int head_dim, T theta) {
        PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
        op_rope<T>(me.t, out.t, position_offset, head_dim, theta);
        return out;
      }, py::arg("position_offset")=0, py::arg("head_dim")=64, py::arg("theta")=10000.0f)
      .def("rope_back", [](const Self &grad_out, int position_offset, int head_dim, T theta) {
        PyTensor<T> grad_in(grad_out.t.h, grad_out.t.w, grad_out.t.on_device);
        op_rope_backward<T>(grad_out.t, grad_in.t, position_offset, head_dim, theta);
        return grad_in;
      }, py::arg("position_offset")=0, py::arg("head_dim")=64, py::arg("theta")=10000.0f)
      .def("softmax", [](const Self &me, bool causal, int seq_offset) {
        PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
        op_softmax<T>(me.t, out.t, causal, seq_offset);
        return out;
      }, py::arg("causal")=false, py::arg("seq_offset")=0)
      .def("softmax_back", [](const Self &output, const Self &grad_out) {
        PyTensor<T> grad_in(output.t.h, output.t.w, output.t.on_device);
        op_softmax_backward<T>(output.t, grad_out.t, grad_in.t);
        return grad_in;
      }, py::arg("grad_out"));
  }
}