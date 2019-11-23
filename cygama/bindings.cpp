#include <pybind11/pybind11.h>
#include "test.h"

namespace py = pybind11;

PYBIND11_PLUGIN(cygama) {
  py::module m("cygama", "docstring 1");

  m.def("add", &add, "docstring 2");

  m.def("subtract", &subtract, "docstring 3");
  
  m.def("compress_signal", &compress_signal, "docstring 4");

  return m.ptr();
}