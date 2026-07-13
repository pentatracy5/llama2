#include <kernel/swiglu.h>
#include <core/Tensor.cuh>
#include <common/config.h>
#include <common/macro.h>
#include <common/types.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T>
__device__ __forceinline__ T silu(const T in)
{
    return T(float(in) / (1.f + expf(-float(in))));
}

template <typename T, unsigned int BYTE_SIZE>
__device__ __forceinline__ Vec<T, BYTE_SIZE> silu(const Vec<T, BYTE_SIZE> &in)
{
    Vec<T, BYTE_SIZE> out;
#pragma unroll Vec < T, BYTE_SIZE> ::vec_len
    for (unsigned int i = 0; i < Vec<T, BYTE_SIZE>::vec_len; i++)
        out[i] = silu<T>(in[i]);
    return out;
}

template <typename T>
__global__ void swiglu_kernel(const unsigned int num_input_tokens,
                              const unsigned int intermediate_size,
                              const T *input,
                              T *output)
{
    using VECTYPE = typename Vec<T, CUDA_VEC_LS_BYTE_SIZE>;
    constexpr unsigned int vlen = VECTYPE::vec_len;

    const VECTYPE *vec_in = reinterpret_cast<const VECTYPE *>(input);
    VECTYPE *vec_out = reinterpret_cast<VECTYPE *>(output);

    const unsigned int inter_vec_size = intermediate_size / vlen;

    unsigned int token_id = blockIdx.x;
    while (token_id < num_input_tokens)
    {
        unsigned int element_idx = threadIdx.x;
        while (element_idx < inter_vec_size)
        {
            const VECTYPE x = vec_in[token_id * inter_vec_size * 2 + element_idx];
            const VECTYPE y = vec_in[token_id * inter_vec_size * 2 + inter_vec_size + element_idx];
            vec_out[token_id * inter_vec_size + element_idx] = silu(x) * y;
            element_idx += blockDim.x;
        }
        token_id += gridDim.x;
    }
}

template <typename T>
void launch_swiglu(const Tensor<T> &input,
                   Tensor<T> &output)
{
    constexpr unsigned int vlen = Vec<T, CUDA_VEC_LS_BYTE_SIZE>::vec_len;
    const unsigned int num_input_tokens = output.shape()[0];
    const unsigned int intermediate_size = output.shape()[1];
    assert(input.shape()[0] == num_input_tokens && "Incorrect input size");
    assert(input.shape()[1] == 2 * intermediate_size && "Incorrect input size");
    assert(intermediate_size % vlen == 0 && "Intermediate size is not divisible by vector length");
    const dim3 threads_per_block{THREADS_PER_BLOCK};
    const dim3 n_threads{std::min(num_input_tokens, NUM_BLOCKS_X) * threads_per_block.x};
    CUDA_LAUNCH(swiglu_kernel, n_threads, threads_per_block)(num_input_tokens,
                                                             intermediate_size,
                                                             input.data(),
                                                             output.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_swiglu<float>(const Tensor<float> &input,
                                   Tensor<float> &output);

template void launch_swiglu<half>(const Tensor<half> &input,
                                  Tensor<half> &output);
