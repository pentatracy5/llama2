#include <kernel/build_causal_mask.h>
#include <core/Tensor.cuh>
#include <common/macro.h>
#include <common/config.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T>
__global__ void build_causal_mask_kernel(const unsigned int batch_size,
                                         const unsigned int q_cache_len,
                                         const unsigned int kv_cache_len,
                                         const unsigned int max_q_len,
                                         const unsigned int max_kv_len,
                                         const unsigned int *q_lens,
                                         const unsigned int *kv_lens,
                                         T *mask)
{
    const unsigned int seq_stride = gridDim.y;
    const unsigned int q_stride = gridDim.x;
    const unsigned int kv_stride = blockDim.x;
    unsigned int seq_id = blockIdx.y;
    while (seq_id < batch_size)
    {
        unsigned int seq_offset = seq_id * q_cache_len * kv_cache_len;
        unsigned int q_len = q_lens[seq_id];
        unsigned int kv_len = kv_lens[seq_id];
        unsigned int q_id = blockIdx.x;
        while (q_id < max_q_len)
        {
            unsigned int q_offset = q_id * kv_cache_len + seq_offset;
            unsigned int kv_id = threadIdx.x;
            while (kv_id < max_kv_len)
            {
                unsigned int kv_offset = q_offset + kv_id;
                bool is_one = q_id < q_len && kv_id < kv_len && q_id >= kv_id - (kv_len - q_len);
                mask[kv_offset] = static_cast<T>(is_one);
                kv_id += kv_stride;
            }
            q_id += q_stride;
        }
        seq_id += seq_stride;
    }
}

template <typename T>
void launch_build_causal_mask(const unsigned int max_q_len,
                              const unsigned int max_kv_len,
                              const Tensor<unsigned int> &q_lens,
                              const Tensor<unsigned int> &kv_lens,
                              Tensor<T> &mask)
{
    const unsigned int batch_size = mask.shape()[0];
    const unsigned int q_cache_len = mask.shape()[1];
    const unsigned int kv_cache_len = mask.shape()[2];
    const dim3 threads_per_block{std::min(THREADS_PER_BLOCK, (max_kv_len + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE)};
    const dim3 nthreads{std::min(NUM_BLOCKS_X, max_q_len) * threads_per_block.x,
                        std::min(NUM_BLOCKS_Y, batch_size) * threads_per_block.y};
    CUDA_LAUNCH(build_causal_mask_kernel, nthreads, threads_per_block)(batch_size,
                                                                       q_cache_len,
                                                                       kv_cache_len,
                                                                       max_q_len,
                                                                       max_kv_len,
                                                                       q_lens.data(),
                                                                       kv_lens.data(),
                                                                       mask.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_build_causal_mask<unsigned char>(const unsigned int max_q_len,
                                                      const unsigned int max_kv_len,
                                                      const Tensor<unsigned int> &q_lens,
                                                      const Tensor<unsigned int> &kv_lens,
                                                      Tensor<unsigned char> &mask);
