cmake_minimum_required(VERSION 3.18)
project(dex_algorithm)

find_package(pybind11 REQUIRED)

pybind11_add_module(dex_algorithm example_greedy.cpp)

target_compile_options(dex_algorithm PRIVATE -O3)

option(ENABLE_NATIVE_ARCH "Enable -march=native for current CPU" OFF)
if(ENABLE_NATIVE_ARCH)
    target_compile_options(dex_algorithm PRIVATE -march=native)
endif()
