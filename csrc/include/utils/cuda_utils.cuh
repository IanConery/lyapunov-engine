#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdint>

#define WARP_SIZE 32
#define FULL_WARP_MASK 0xFFFFFFFF

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err = (expr);                                              \
        if (err != cudaSuccess) {                                              \
            std::stringstream ss;                                              \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__              \
               << " code=" << err << " (" << cudaGetErrorString(err)          \
               << ") \"" #expr "\"";                                           \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::stringstream ss;                                              \
            ss << "CUDA kernel launch error at " << __FILE__ << ":"            \
               << __LINE__ << " code=" << err << " ("                          \
               << cudaGetErrorString(err) << ")";                              \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (0)

namespace lyapunov {
namespace cuda {

// Vectorized memory access helpers (128-bit transactions)
template <typename T, int N>
struct alignas(sizeof(T) * N) VecType {
    T data[N];
};

using float4_t = VecType<float, 4>;
using float2_t = VecType<float, 2>;

#ifdef __CUDACC__

// Type conversion helpers
template <typename T>
__inline__ __device__ float to_float(T val) {
    return static_cast<float>(val);
}

template <>
__inline__ __device__ float to_float<half>(half val) {
    return __half2float(val);
}

template <typename T>
__inline__ __device__ T from_float(float val) {
    return static_cast<T>(val);
}

template <>
__inline__ __device__ half from_float<half>(float val) {
    return __float2half(val);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
__inline__ __device__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template <>
__inline__ __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}
#endif

// Warp-level sum reduction
template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

// Warp-level max reduction
template <typename T>
__inline__ __device__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(FULL_WARP_MASK, val, offset);
        val = (val > other) ? val : other;
    }
    return val;
}

// Block-level sum reduction using shared memory across warps
template <typename T, int NUM_THREADS>
__inline__ __device__ T block_reduce_sum(T val, T* shared_scratch) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

    val = warp_reduce_sum<T>(val);

    if (lane_id == 0) {
        shared_scratch[warp_id] = val;
    }
    __syncthreads();

    T warp_sum = (threadIdx.x < NUM_WARPS) ? shared_scratch[lane_id] : T(0);
    if (warp_id == 0) {
        warp_sum = warp_reduce_sum<T>(warp_sum);
    }
    return warp_sum;
}

// Block-level max reduction using shared memory across warps
template <typename T, int NUM_THREADS>
__inline__ __device__ T block_reduce_max(T val, T* shared_scratch) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

    val = warp_reduce_max<T>(val);

    if (lane_id == 0) {
        shared_scratch[warp_id] = val;
    }
    __syncthreads();

    T warp_max = (threadIdx.x < NUM_WARPS) ? shared_scratch[lane_id] : T(-1e20);
    if (warp_id == 0) {
        warp_max = warp_reduce_max<T>(warp_max);
    }
    return warp_max;
}

// Fast activation functions
__inline__ __device__ float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__inline__ __device__ half fast_silu_half(half x) {
    float x_f = __half2float(x);
    return __float2half(fast_silu(x_f));
}

#endif // __CUDACC__

} // namespace cuda
} // namespace lyapunov
