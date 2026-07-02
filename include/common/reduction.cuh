#pragma once

#include <cuda_runtime.h>

template <typename T>
struct AddOp
{
    static __host__ __device__ T invoke(T a, T b) { return a + b; }
};

template <typename T>
struct MaxOp
{
    static __host__ __device__ T invoke(T a, T b) { return a > b ? a : b; }
};

template <int warp_size, typename T, template <typename> class Op>
__device__ __forceinline__ T shuffle_warp_reduce(T x)
{
    constexpr unsigned int mask = (1ULL << warp_size) - 1;
    if (32 <= warp_size)
        x = Op<T>::invoke(x, __shfl_down_sync(mask, x, 16));
    if (16 <= warp_size)
        x = Op<T>::invoke(x, __shfl_down_sync(mask, x, 8));
    if (8 <= warp_size)
        x = Op<T>::invoke(x, __shfl_down_sync(mask, x, 4));
    if (4 <= warp_size)
        x = Op<T>::invoke(x, __shfl_down_sync(mask, x, 2));
    if (2 <= warp_size)
        x = Op<T>::invoke(x, __shfl_down_sync(mask, x, 1));
    return x;
}
