cmake_minimum_required(VERSION 3.5)
project(tvm_cpp_benchmark)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR})

SET(HOME_TVM /root/aunjgr/tvm)
include(util/Util.cmake)
include(util/FindCUDA.cmake)
set(USE_CUDA ON)                                   
find_cuda(${USE_CUDA})


INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${HOME_TVM}/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dlpack/include)
INCLUDE_DIRECTORIES(/usr/local/cuda/include)

add_executable(tvm_cpp_benchmark   cpp_benchmark.cpp tvm_runtime_pack.cc)
target_link_libraries(tvm_cpp_benchmark  dl pthread ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})


