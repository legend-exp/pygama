// sigcompress.cc
// pybind11 adaptation by Clint Wiseman
// 
// based on radware-sigcompress, v1.0
// This code is licensed under the MIT License (MIT).
// Copyright (c) 2018, David C. Radford <radforddc@ornl.gov>

#include <iostream>

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

std::vector<unsigned short> compress_signal(std::vector<short>& sig_in);

// int compress_signal(short *sig_in, unsigned short *sig_out, int sig_len_in);

int decompress_signal(unsigned short *sig_in, short *sig_out, int sig_len_in);

std::vector<int> multiply(const std::vector<double>& input);

py::array_t<int> py_multiply(py::array_t<double, py::array::c_style | py::array::forcecast> array);

py::array_t<int> py_compress(py::array_t<int, py::array::c_style | py::array::forcecast> array);