#include <kernel/build_padding_index_maps.h>
#include <core/Tensor.cuh>

__global__ void build_padding_index_maps_kernel(const unsigned int num_actual_tokens,
                                                const unsigned int max_seq_len,
                                                const int *seq_lens,
                                                int *unpad_to_padded_idx,
                                                int *seq_offsets)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (idx < num_actual_tokens)
    {
        unsigned int accumulate_len = 0;
        unsigned int seq_id = 0;
        while (accumulate_len + seq_lens[seq_id] <= idx)
            accumulate_len += seq_lens[seq_id++];
        unsigned int idx_in_seq = idx - accumulate_len;
        unpad_to_padded_idx[idx] = seq_id * max_seq_len + idx_in_seq;
        if (0 == idx_in_seq)
            seq_offsets[seq_id] = accumulate_len;
        idx += stride;
    }
}

void launch_build_padding_index_maps(const unsigned int num_actual_tokens,
                                     const unsigned int max_seq_len,
                                     const Tensor<int> &seq_lens,
                                     Tensor<int> &unpad_to_padded_idx,
                                     Tensor<int> &seq_offsets)
{
    const dim3 nthreads{256 * 512};
    const dim3 threads_per_block{512};
    CUDA_LAUNCH(build_padding_index_maps_kernel, nthreads, threads_per_block)(num_actual_tokens,
                                                                              max_seq_len,
                                                                              seq_lens.data(),
                                                                              unpad_to_padded_idx.data(),
                                                                              seq_offsets.data());
}
