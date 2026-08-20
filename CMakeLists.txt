cmake_minimum_required(VERSION 3.20)
project(lyapunov_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr")

include_directories(
    csrc/include
)

set(CUDA_SOURCES
    csrc/kernels/fused_rmsnorm_swiglu.cu
    csrc/kernels/flash_attn_v2.cu
    csrc/kernels/flash_decoding.cu
    csrc/kernels/paged_attention.cu
)

set(CPP_SOURCES
    csrc/engine/block_manager.cpp
    csrc/engine/scheduler.cpp
    csrc/engine/cuda_graph.cpp
)

add_library(lyapunov_engine_core STATIC ${CUDA_SOURCES} ${CPP_SOURCES})

set_target_properties(lyapunov_engine_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
