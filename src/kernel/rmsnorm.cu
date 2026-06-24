#include <kernel/rmsnorm.h>
#include <core/Tensor.cuh>
#include <common/types.h>
#include <common/config.h>
#include <common/reduction.cuh>
#include <cuda_fp16.h>

template <typename T>
__global__ void rmsnorm_kernel(const unsigned int num_input_tokens,
                               const unsigned int embed_dim,
                               const T *input_tokens,
                               const T *weights,
                               T *output_tokens)
{
    using VECTYPE = typename VecType<T>::Type;
    constexpr unsigned int vlen = VecType<T>::vec_len;
    
    extern __shared__ char temp[];
    float *reduce_buf = reinterpret_cast<float *>(temp);
    T *shared_weights = reinterpret_cast<T *>(reduce_buf + WARPS_PER_BLOCK);

    unsigned int tid = threadIdx.x * vlen;
    const unsigned int tid_stride = blockDim.x * vlen;
    while (tid < embed_dim + 1 - vlen)
    {
        FETCH_VEC(VECTYPE, shared_weights[tid]) = FETCH_VEC(const VECTYPE, weights[tid]);
        tid += tid_stride;
    }

    float x;
    unsigned int token_idx = blockIdx.x;
    const unsigned int token_idx_stride = gridDim.x;
    while (token_idx < num_input_tokens)
    {
        x = 0.f;
        tid = threadIdx.x * vlen;
        while (tid < embed_dim + 1 - vlen)
        {
            const VECTYPE in = FETCH_VEC(const VECTYPE, input_tokens[token_idx * embed_dim + tid]);
            const T *in_ptr = reinterpret_cast<const T *>(&in);
#pragma unroll vlen
            for (unsigned int i = 0; i < vlen; i++)
                x += float(in_ptr[i] * in_ptr[i]);
            tid += tid_stride;
        }

        tid = threadIdx.x;
        x = shuffle_warp_reduce<WARP_SIZE, float, AddOp>(x);
        if (0 == (tid & (WARP_SIZE - 1)))
            reduce_buf[tid >> 5] = x;
        __syncthreads();

        if (tid < WARPS_PER_BLOCK)
            x = shuffle_warp_reduce<WARPS_PER_BLOCK, float, AddOp>(reduce_buf[tid]);

        if (0 == tid)
            reduce_buf[0] = rsqrtf(x / embed_dim + EPS);
        __syncthreads();
        T inv_var = reduce_buf[0];
        __syncthreads();

        tid = threadIdx.x * vlen;
        while (tid < embed_dim + 1 - vlen)
        {
            const VECTYPE in = FETCH_VEC(const VECTYPE, input_tokens[token_idx * embed_dim + tid]);
            const VECTYPE weight = FETCH_VEC(const VECTYPE, shared_weights[tid]);
            VECTYPE out;
            const T *in_ptr = reinterpret_cast<const T *>(&in);
            const T *weight_ptr = reinterpret_cast<const T *>(&weight);
            T *out_ptr = reinterpret_cast<T *>(&out);
#pragma unroll vlen
            for (unsigned int i = 0; i < vlen; i++)
                out_ptr[i] = in_ptr[i] * weight_ptr[i] * inv_var;
            FETCH_VEC(VECTYPE, output_tokens[token_idx * embed_dim + tid]) = out;
            tid += tid_stride;
        }

        token_idx += token_idx_stride;
    }
}

template <typename T>
void launch_rmsnorm(const Tensor<T> &input_tokens,
                    const Tensor<T> &weights,
                    Tensor<T> &output_tokens)
{
    const dim3 threads_per_block{THREADS_PER_BLOCK};
    const dim3 nthreads{std::min(NUM_BLOCKS, input_tokens.shape()[0]) * threads_per_block.x};
    unsigned int shared_mem_bytes = WARPS_PER_BLOCK * sizeof(float) + input_tokens.shape()[1] * sizeof(T);
    CUDA_LAUNCH_SHAREDMEM(rmsnorm_kernel, nthreads, threads_per_block, shared_mem_bytes)(input_tokens.shape()[0],
                                                                                         input_tokens.shape()[1],
                                                                                         input_tokens.data(),
                                                                                         weights.data(),
                                                                                         output_tokens.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_rmsnorm<float>(const Tensor<float> &, const Tensor<float> &, Tensor<float> &);
template void launch_rmsnorm<__half>(const Tensor<__half> &, const Tensor<__half> &, Tensor<__half> &);
