cmake_minimum_required(VERSION 2.8.12)

# set the name of the extension
project(cygama)
set(SOURCE_DIR "cygama")

# Tell cmake that headers are in source_dir
include_directories(${SOURCE_DIR})

# Specify C++ sources
set(GLOB SOURCES "${SOURCE_DIR}/*.cc")

# point to the pybind11 installation
add_subdirectory(lib/pybind11)

# Generate the python extension module (used by setup.py)
pybind11_add_module(cygama ${SOURCES} "${SOURCE_DIR}/bindings.cpp")