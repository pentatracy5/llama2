#include <kernel/compute_masked_attention_score.h>
#include <kernel/reduction.cuh>
#include <core/Tensor.cuh>
#include <common/macro.h>
#include <common/config.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>

template <typename T, typename MASK_T>
__global__ void compute_masked_attention_score_kernel(const float scale,
                                                      const unsigned int batch_size,
                                                      const unsigned int q_head_num,
                                                      const unsigned int q_cache_len,
                                                      const unsigned int kv_cache_len,
                                                      const unsigned int max_kv_len,
                                                      const unsigned int *q_lens,
                                                      const MASK_T *mask,
                                                      T *qk)
{
    extern __shared__ float reduce_buf[];
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid - warp_id * WARP_SIZE;
    unsigned int seq_id = blockIdx.z;
    while (seq_id < batch_size)
    {
        const unsigned int q_len = q_lens[seq_id];
        const unsigned int qk_seq_offset = seq_id * q_head_num;
        const unsigned int mask_seq_offset = seq_id * q_cache_len;

        unsigned int head_id = blockIdx.y;
        while (head_id < q_head_num)
        {
            const unsigned int qk_h_offset = (qk_seq_offset + head_id) * q_cache_len;

            unsigned int q_id = blockIdx.x;
            while (q_id < q_len)
            {
                const unsigned int qk_q_offset = (qk_h_offset + q_id) * kv_cache_len;
                const unsigned int mask_q_offset = (mask_seq_offset + q_id) * kv_cache_len;

                float max_val = -FLT_MAX;
                unsigned int k_id = threadIdx.x;
                while (k_id < max_kv_len)
                {
                    const unsigned int qk_k_offset = qk_q_offset + k_id;
                    const unsigned int mask_k_offset = mask_q_offset + k_id;
                    max_val = bool(mask[mask_k_offset]) ? MaxOp<float>::invoke(float(qk[qk_k_offset]) * scale, max_val) : max_val;
                    k_id += blockDim.x;
                }

                max_val = shuffle_warp_reduce<WARP_SIZE, float, MaxOp>(max_val);
                if (0 == lane_id)
                    reduce_buf[warp_id] = max_val;
                __syncthreads();
                if (tid < WARPS_PER_BLOCK)
                    max_val = shuffle_warp_reduce<WARPS_PER_BLOCK, float, MaxOp>(reduce_buf[tid]);
                if (0 == tid)
                    reduce_buf[0] = max_val;
                __syncthreads();
                max_val = reduce_buf[0];
                __syncthreads();

                float x = 0.f;
                k_id = threadIdx.x;
                while (k_id < max_kv_len)
                {
                    const unsigned int qk_k_offset = qk_q_offset + k_id;
                    const unsigned int mask_k_offset = mask_q_offset + k_id;
                    x += bool(mask[mask_k_offset]) ? expf(float(qk[qk_k_offset]) * scale - max_val) : 0.f;
                    k_id += blockDim.x;
                }

                x = shuffle_warp_reduce<WARP_SIZE, float, AddOp>(x);
                if (0 == lane_id)
                    reduce_buf[warp_id] = x;
                __syncthreads();
                if (tid < WARPS_PER_BLOCK)
                    x = shuffle_warp_reduce<WARPS_PER_BLOCK, float, AddOp>(reduce_buf[tid]);
                if (0 == tid)
                    reduce_buf[0] = 1.f / (x + SOFTMAX_EPS);
                __syncthreads();

                k_id = threadIdx.x;
                while (k_id < max_kv_len)
                {
                    const unsigned int qk_k_offset = qk_q_offset + k_id;
                    const unsigned int mask_k_offset = mask_q_offset + k_id;
                    qk[qk_k_offset] = bool(mask[mask_k_offset]) ? expf(float(qk[qk_k_offset]) * scale - max_val) * reduce_buf[0] : 0.f;
                    k_id += blockDim.x;
                }
                __syncthreads();

                q_id += gridDim.x;
            }
            head_id += gridDim.y;
        }
        seq_id += gridDim.z;
    }
}

template <typename T, typename MASK_T>
void launch_compute_masked_attention_score(const unsigned int head_dim,
                                           const unsigned int max_q_len,
                                           const unsigned int max_kv_len,
                                           const Tensor<unsigned int> &q_lens,
                                           const Tensor<MASK_T> &mask,
                                           Tensor<T> &qk)
{
    const float inv_sqrt_head_dim = 1.f / sqrtf(float(head_dim));
    const unsigned int batch_size = qk.shape()[0];
    const unsigned int q_head_num = qk.shape()[1];
    const unsigned int q_cache_len = qk.shape()[2];
    const unsigned int kv_cache_len = qk.shape()[3];
    assert(batch_size == q_lens.shape()[0] && "Incorrect q_lens size");
    assert(batch_size == mask.shape()[0] && "Incorrect mask size");
    assert(q_cache_len == mask.shape()[1] && "Incorrect mask size");
    assert(kv_cache_len == mask.shape()[2] && "Incorrect mask size");
    assert(max_q_len <= q_cache_len && "Incorrect max_q_len");
    assert(max_kv_len <= kv_cache_len && "Incorrect max_kv_len");

    const dim3 threads_per_block{THREADS_PER_BLOCK};
    const dim3 n_threads{std::min(max_q_len, NUM_BLOCKS_X) * threads_per_block.x,
                         std::min(q_head_num, NUM_BLOCKS_Y) * threads_per_block.y,
                         std::min(batch_size, NUM_BLOCKS_Z) * threads_per_block.z};
    const unsigned int shared_mem_bytes = WARPS_PER_BLOCK * sizeof(float);
    CUDA_LAUNCH_SHAREDMEM(compute_masked_attention_score_kernel, n_threads, threads_per_block, shared_mem_bytes)(inv_sqrt_head_dim,
                                                                                                                 batch_size,
                                                                                                                 q_head_num,
                                                                                                                 q_cache_len,
                                                                                                                 kv_cache_len,
                                                                                                                 max_kv_len,
                                                                                                                 q_lens.data(),
                                                                                                                 mask.data(),
                                                                                                                 qk.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_compute_masked_attention_score<float, unsigned char>(const unsigned int head_dim,
                                                                          const unsigned int max_q_len,
                                                                          const unsigned int max_kv_len,
                                                                          const Tensor<unsigned int> &q_lens,
                                                                          const Tensor<unsigned char> &mask,
                                                                          Tensor<float> &qk);

template void launch_compute_masked_attention_score<half, unsigned char>(const unsigned int head_dim,
                                                                         const unsigned int max_q_len,
                                                                         const unsigned int max_kv_len,
                                                                         const Tensor<unsigned int> &q_lens,
                                                                         const Tensor<unsigned char> &mask,
                                                                         Tensor<half> &qk);
