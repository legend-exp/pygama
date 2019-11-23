cmake_minimum_required(VERSION 2.8.12)
project(cygama)

SET(SOURCE_DIR "cygama")

# Tell cmake that headers are in source_dir
include_directories(${SOURCE_DIR})

# Specify C++ source
SET(SOURCES "${SOURCE_DIR}/test.cpp")

# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(cygama ${SOURCES} "${SOURCE_DIR}/bindings.cpp")