#include <pybind11/pybind11.h>
// #include "test.h"
#include "sigcompress.h"

namespace py = pybind11;

PYBIND11_PLUGIN(cygama) {
  /*
  structure taken from here:
  https://www.benjack.io/2018/02/02/python-cpp-revisited.html
  
  TODO: can include sphinx-style docstrings for the c++ functions here, check the example above
  */
  
  py::module m("cygama", "docstring 1");

  // m.def("add", &add, "docstring 2");

  // m.def("subtract", &subtract, "docstring 3");
  
  // m.def("compress_signal", &compress_signal, "docstring 4");
  
  // m.def("decompress_signal", &decompress_signal, "docstring 5");
  
  // m.def("multiply", &multiply, "docstring 6");
  
  // m.def("py_multiply", &py_multiply, "docstring 6");

  return m.ptr();
}