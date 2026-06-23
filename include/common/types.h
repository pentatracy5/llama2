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
    static constexpr unsigned int vec_len = 0;
    static_assert(false, "Unsupported dtype");
};

template <>
struct VecType<float>
{
    using Type = float4;
    static constexpr unsigned int vec_len = 4;
};

template <>
struct VecType<__half>
{
    using Type = __half8;
    static constexpr unsigned int vec_len = 8;
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
    static constexpr DataType dtype = UNSUPPORTED;
    static_assert(false, "Unsupported dtype");
};

template <>
struct RealTypeToDataType<float>
{
    static constexpr DataType dtype = FP32;
};

template <>
struct RealTypeToDataType<half>
{
    static constexpr DataType dtype = FP16;
};

template <>
struct RealTypeToDataType<int8_t>
{
    static constexpr DataType dtype = INT8;
};

template <>
struct RealTypeToDataType<int32_t>
{
    static constexpr DataType dtype = INT32;
};

template <>
struct RealTypeToDataType<bool>
{
    static constexpr DataType dtype = BOOL;
};

template <>
struct RealTypeToDataType<unsigned char>
{
    static constexpr DataType dtype = BYTES;
};
