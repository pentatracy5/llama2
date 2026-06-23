#pragma once

#include <cuda_fp16.h>

struct alignas(16) __half8
{
    __half val[8];
    __host__ __device__ inline const __half &operator[](int i) const { return val[i]; }
    __host__ __device__ inline __half &operator[](int i) { return val[i]; }
};

template <typename T>
struct VecType
{
    using Type = void;
    static const unsigned int vec_len = 0;
    static_assert(false, "Unsupported dtype");
};

template <>
struct VecType<float>
{
    using Type = float4;
    static const unsigned int vec_len = 4;
};

template <>
struct VecType<__half>
{
    using Type = __half8;
    static const unsigned int vec_len = 8;
};

enum DeviceType
{
    CPU_PINNED,
    CPU,
    GPU
};

enum DataType
{
    UNSUPPORTED,
    FP32,
    FP16,
    INT8,
    INT32,
    BOOL,
    BYTES
};

template <typename T>
struct RealTypeToDataType
{
    static const DataType dtype = UNSUPPORTED;
    static_assert(false, "Unsupported dtype");
};

template <>
struct RealTypeToDataType<float>
{
    static const DataType dtype = FP32;
};

template <>
struct RealTypeToDataType<half>
{
    static const DataType dtype = FP16;
};

template <>
struct RealTypeToDataType<int8_t>
{
    static const DataType dtype = INT8;
};

template <>
struct RealTypeToDataType<int32_t>
{
    static const DataType dtype = INT32;
};

template <>
struct RealTypeToDataType<bool>
{
    static const DataType dtype = BOOL;
};

template <>
struct RealTypeToDataType<unsigned char>
{
    static const DataType dtype = BYTES;
};
