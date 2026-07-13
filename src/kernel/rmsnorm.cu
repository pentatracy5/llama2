#include <kernel/rmsnorm.h>
#include <kernel/reduction.cuh>
#include <core/Tensor.cuh>
#include <common/types.h>
#include <common/macro.h>
#include <common/config.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T, bool add_residual>
__global__ void fused_add_residual_rmsnorm_kernel(const unsigned int num_input_tokens,
                                                  const unsigned int embed_dim,
                                                  const T *weights,
                                                  T *x,
                                                  T *residual)
{
    using SHARED_VECTYPE = typename Vec<T, SHARED_MEM_BANK_BYTE_SIZE>;
    using VECTYPE = typename Vec<SHARED_VECTYPE, CUDA_VEC_LS_BYTE_SIZE>;
    constexpr unsigned int shared_vlen = SHARED_VECTYPE::vec_len;
    constexpr unsigned int vlen = VECTYPE::vec_len;

    extern __shared__ char temp[];
    float *reduce_buf = reinterpret_cast<float *>(temp);
    SHARED_VECTYPE *shared_vec_weights = reinterpret_cast<SHARED_VECTYPE *>(reduce_buf + WARPS_PER_BLOCK);

    const VECTYPE *vec_weights = reinterpret_cast<const VECTYPE *>(weights);
    VECTYPE *vec_x = reinterpret_cast<VECTYPE *>(x);
    VECTYPE *vec_residual = reinterpret_cast<VECTYPE *>(residual);

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

    float variance;
    unsigned int token_idx = blockIdx.x;
    const unsigned int token_idx_stride = gridDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int laneid = tid & (WARP_SIZE - 1);
    const unsigned int warpid = tid >> 5;
    while (token_idx < num_input_tokens)
    {
        variance = 0.f;
        idx = threadIdx.x;
        while (idx < vec_embed_dim)
        {
            VECTYPE in = vec_x[token_idx * vec_embed_dim + idx];
            if constexpr (add_residual)
            {
                in += vec_residual[token_idx * vec_embed_dim + idx];
                vec_x[token_idx * vec_embed_dim + idx] = in;
            }
            in *= in;
#pragma unroll vlen
            for (unsigned int i = 0; i < vlen; i++)
#pragma unroll shared_vlen
                for (unsigned int j = 0; j < shared_vlen; j++)
                    variance += float(in[i][j]);
            idx += idx_stride;
        }

        variance = shuffle_warp_reduce<WARP_SIZE, float, AddOp>(variance);
        if (0 == laneid)
            reduce_buf[warpid] = variance;
        __syncthreads();
        if (tid < WARPS_PER_BLOCK)
            variance = shuffle_warp_reduce<WARPS_PER_BLOCK, float, AddOp>(reduce_buf[tid]);
        if (0 == tid)
            reduce_buf[0] = rsqrtf(variance / embed_dim + RMSNORM_EPS);
        __syncthreads();
        T inv_var = reduce_buf[0];
        __syncthreads();

        idx = threadIdx.x;
        while (idx < vec_embed_dim)
        {
            const unsigned int offset = idx / WARP_SIZE * WARP_SIZE * vlen;
            const VECTYPE in = vec_x[token_idx * vec_embed_dim + idx];
            VECTYPE out;
#pragma unroll vlen
            for (unsigned int i = 0; i < vlen; i++)
#pragma unroll shared_vlen
                for (unsigned int j = 0; j < shared_vlen; j++)
                    out[i][j] = in[i][j] * shared_vec_weights[offset + i * WARP_SIZE + lane_id][j] * inv_var;
            vec_residual[token_idx * vec_embed_dim + idx] = out;
            idx += idx_stride;
        }

        token_idx += token_idx_stride;
    }
}

template <typename T, bool add_residual>
void launch_fused_add_residual_rmsnorm(const Tensor<T> &weights,
                                       Tensor<T> &x,
                                       Tensor<T> &residual)
{
    const unsigned int num_input_tokens = x.shape()[0];
    const unsigned int embed_dim = x.shape()[1];
    assert(residual.shape()[0] == num_input_tokens && "Incorrect residual size");
    assert(residual.shape()[1] == embed_dim && "Incorrect residual size");
    assert(weights.shape()[0] == embed_dim && "Incorrect rmsnorm weight size");

    const dim3 threads_per_block{THREADS_PER_BLOCK};
    const dim3 nthreads{std::min(NUM_BLOCKS_X, x.shape()[0]) * threads_per_block.x};
    constexpr unsigned int WARP_GROUP_BYTES = WARP_SIZE * CUDA_VEC_LS_BYTE_SIZE;
    const unsigned int weight_bytes = ((embed_dim * sizeof(T) + WARP_GROUP_BYTES - 1) / WARP_GROUP_BYTES) * WARP_GROUP_BYTES;
    const unsigned int shared_mem_bytes = WARPS_PER_BLOCK * sizeof(float) + weight_bytes;
    CUDA_LAUNCH_SHAREDMEM((fused_add_residual_rmsnorm_kernel<T, add_residual>), nthreads, threads_per_block, shared_mem_bytes)(num_input_tokens,
                                                                                                                               embed_dim,
                                                                                                                               weights.data(),
                                                                                                                               x.data(),
                                                                                                                               residual.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_fused_add_residual_rmsnorm<float, false>(const Tensor<float> &weights,
                                                              Tensor<float> &x,
                                                              Tensor<float> &residual);

template void launch_fused_add_residual_rmsnorm<half, false>(const Tensor<half> &weights,
                                                             Tensor<half> &x,
                                                             Tensor<half> &residual);

template void launch_fused_add_residual_rmsnorm<float, true>(const Tensor<float> &weights,
                                                             Tensor<float> &x,
                                                             Tensor<float> &residual);

template void launch_fused_add_residual_rmsnorm<half, true>(const Tensor<half> &weights,
                                                            Tensor<half> &x,
                                                            Tensor<half> &residual);
