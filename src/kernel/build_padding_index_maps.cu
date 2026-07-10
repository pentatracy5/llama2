#include <kernel/build_padding_index_maps.h>
#include <core/Tensor.cuh>
#include <common/macro.h>
#include <common/config.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void build_padding_index_maps_kernel(const unsigned int num_input_tokens,
                                                const unsigned int q_cache_len,
                                                const unsigned int *seq_lens,
                                                unsigned int *unpad_to_pad_idx,
                                                unsigned int *seq_offsets)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (idx < num_input_tokens)
    {
        unsigned int accumulate_len = 0;
        unsigned int seq_id = 0;
        while (accumulate_len + seq_lens[seq_id] <= idx)
            accumulate_len += seq_lens[seq_id++];
        unsigned int idx_in_seq = idx - accumulate_len;
        unpad_to_pad_idx[idx] = seq_id * q_cache_len + idx_in_seq;
        if (0 == idx_in_seq)
            seq_offsets[seq_id] = accumulate_len;
        idx += stride;
    }
}

void launch_build_padding_index_maps(const unsigned int num_input_tokens,
                                     const Tensor<unsigned int> &seq_lens,
                                     Tensor<unsigned int> &unpad_to_pad_idx,
                                     Tensor<unsigned int> &seq_offsets)
{
    const dim3 threads_per_block{THREADS_PER_BLOCK};
    const dim3 nthreads{std::min(NUM_BLOCKS_X * threads_per_block.x, num_input_tokens)};
    CUDA_LAUNCH(build_padding_index_maps_kernel, nthreads, threads_per_block)(num_input_tokens,
                                                                              unpad_to_pad_idx.shape()[1],
                                                                              seq_lens.data(),
                                                                              unpad_to_pad_idx.data(),
                                                                              seq_offsets.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}
