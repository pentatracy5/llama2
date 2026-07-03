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
    using SHARED_VECTYPE = typename VecType<T, SHARED_MEM_BANK_BYTE_SIZE>::Type;
    using VECTYPE = typename VecType<SHARED_VECTYPE, CUDA_VEC_LS_BYTE_SIZE>::Type;
    constexpr unsigned int shared_vlen = SHARED_VECTYPE::vec_len;
    constexpr unsigned int vlen = VECTYPE::vec_len;

    extern __shared__ char temp[];
    float *reduce_buf = reinterpret_cast<float *>(temp);
    SHARED_VECTYPE *shared_vec_weights = reinterpret_cast<SHARED_VECTYPE *>(reduce_buf + WARPS_PER_BLOCK);

    const VECTYPE *vec_input_tokens = reinterpret_cast<const VECTYPE *>(input_tokens);
    const VECTYPE *vec_weights = reinterpret_cast<const VECTYPE *>(weights);
    VECTYPE *vec_output_tokens = reinterpret_cast<VECTYPE *>(output_tokens);

    unsigned int idx = threadIdx.x;
    const unsigned int lane_id = idx % WARP_SIZE;
    const unsigned int idx_stride = blockDim.x;
    const unsigned int vec_embed_dim = embed_dim / (vlen * shared_vlen);
    while (idx < vec_embed_dim)
    {
        const VECTYPE temp = vec_weights[idx];
        const SHARED_VECTYPE *temp_ptr = reinterpret_cast<const SHARED_VECTYPE *>(&temp);
        const unsigned int offset = idx / WARP_SIZE * WARP_SIZE * vlen;
#pragma unroll vlen
        for (unsigned int i = 0; i < vlen; i++)
            shared_vec_weights[offset + i * WARP_SIZE + lane_id] = temp_ptr[i];
        idx += idx_stride;
    }

    float x;
    unsigned int token_idx = blockIdx.x;
    const unsigned int token_idx_stride = gridDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int laneid = tid & (WARP_SIZE - 1);
    const unsigned int warpid = tid >> 5;
    while (token_idx < num_input_tokens)
    {
        x = 0.f;
        idx = threadIdx.x;
        while (idx < vec_embed_dim)
        {
            const VECTYPE in = vec_input_tokens[token_idx * vec_embed_dim + idx];
#pragma unroll vlen
            for (unsigned int i = 0; i < vlen; i++)
#pragma unroll shared_vlen
                for (unsigned int j = 0; j < shared_vlen; j++)
                    x += float(in[i][j] * in[i][j]);
            idx += idx_stride;
        }

        x = shuffle_warp_reduce<WARP_SIZE, float, AddOp>(x);
        if (0 == laneid)
            reduce_buf[warpid] = x;
        __syncthreads();
        if (tid < WARPS_PER_BLOCK)
            x = shuffle_warp_reduce<WARPS_PER_BLOCK, float, AddOp>(reduce_buf[tid]);
        if (0 == tid)
            reduce_buf[0] = rsqrtf(x / embed_dim + EPS);
        __syncthreads();
        T inv_var = reduce_buf[0];
        __syncthreads();

        idx = threadIdx.x;
        while (idx < vec_embed_dim)
        {
            const unsigned int offset = idx / WARP_SIZE * WARP_SIZE * vlen;
            const VECTYPE in = vec_input_tokens[token_idx * vec_embed_dim + idx];
            VECTYPE out;
#pragma unroll vlen
            for (unsigned int i = 0; i < vlen; i++)
#pragma unroll shared_vlen
                for (unsigned int j = 0; j < shared_vlen; j++)
                    out[i][j] = in[i][j] * shared_vec_weights[offset + i * WARP_SIZE + lane_id][j] * inv_var;
            vec_output_tokens[token_idx * vec_embed_dim + idx] = out;
            idx += idx_stride;
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
    const dim3 nthreads{std::min(NUM_BLOCKS_X, input_tokens.shape()[0]) * threads_per_block.x};
    constexpr unsigned int WARP_GROUP_BYTES = WARP_SIZE * CUDA_VEC_LS_BYTE_SIZE;
    const unsigned int num_input_tokens = input_tokens.shape()[0];
    const unsigned int embed_dim = input_tokens.shape()[1];
    const unsigned int weight_bytes = ((embed_dim * sizeof(T) + WARP_GROUP_BYTES - 1) / WARP_GROUP_BYTES) * WARP_GROUP_BYTES;
    const unsigned int shared_mem_bytes = WARPS_PER_BLOCK * sizeof(float) + weight_bytes;
    CUDA_LAUNCH_SHAREDMEM(rmsnorm_kernel, nthreads, threads_per_block, shared_mem_bytes)(num_input_tokens,
                                                                                         embed_dim,
                                                                                         input_tokens.data(),
                                                                                         weights.data(),
                                                                                         output_tokens.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_rmsnorm<float>(const Tensor<float> &input_tokens,
                                    const Tensor<float> &weights,
                                    Tensor<float> &output_tokens);
template void launch_rmsnorm<half>(const Tensor<half> &input_tokens,
                                   const Tensor<half> &weights,
                                   Tensor<half> &output_tokens);
