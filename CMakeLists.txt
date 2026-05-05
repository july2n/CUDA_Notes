cmake_minimum_required(VERSION 3.18)
project(CUDA_PROJECT VERSION 0.1.0 LANGUAGES CUDA CXX C)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDAToolkit_ROOT /usr/local/cuda)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(0_elementwise)
add_subdirectory(1_reduce/sum)
add_subdirectory(1_reduce/max)
add_subdirectory(2_sgemm)
add_subdirectory(3_transpose)