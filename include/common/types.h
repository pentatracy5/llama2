#pragma once

#include <cuda_fp16.h>

template <typename T>
struct alignas(2 * sizeof(T)) Vec2
{
    T x[2];
};

template <typename T, unsigned int BYTE_SIZE>
struct alignas(BYTE_SIZE) Vec
{
    static constexpr unsigned int vec_len = BYTE_SIZE / sizeof(T);
    T x[vec_len];
    __host__ __device__ inline const T &operator[](unsigned int i) const { return x[i]; }
    __host__ __device__ inline T &operator[](unsigned int i) { return x[i]; }
};

template <typename T, unsigned int BYTE_SIZE>
struct VecType
{
    using Type = Vec<T, BYTE_SIZE>;
    static_assert(BYTE_SIZE % sizeof(T) == 0, "Unsupported vectype");
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
    UINT32,
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
struct RealTypeToDataType<uint32_t>
{
    static constexpr DataType dtype = UINT32;
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
