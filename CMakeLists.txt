cmake_minimum_required(VERSION 2.8.12)

# set the name of the extension
project(cygama)
set(SDIR "cygama")

# Tell cmake that headers are in source_dir
include_directories(${SDIR})

# Specify C++ sources
set(SOURCES "${SDIR}/test.cc" "${SDIR}/sigcompress.cc")
# set(GLOB SOURCES "${SDIR}/*.cc") # this doesn't work like it should

# point to the pybind11 installation
add_subdirectory(cygama/pybind11)

# Generate the python extension module (used by setup.py)
pybind11_add_module(cygama ${SOURCES} "${SDIR}/bindings.cpp")