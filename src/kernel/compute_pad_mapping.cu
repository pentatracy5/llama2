#include <kernel/compute_pad_mapping.h>
#include <core/Tensor.cuh>

__global__ void ComputePadMapping(const unsigned int num_input_ids,
                                  const unsigned int max_seq_len,
                                  const int *seq_lens,
                                  int *unpad2pad_map,
                                  int *pad2unpad_map)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (idx < num_input_ids)
    {
        unsigned int accumulate_len = 0;
        unsigned int seq_id = 0;
        while (accumulate_len + seq_lens[seq_id] <= idx)
            accumulate_len += seq_lens[seq_id++];
        unsigned int idx_in_seq = idx - accumulate_len;
        unpad2pad_map[idx] = seq_id * max_seq_len + idx_in_seq;
        if (0 == idx_in_seq)
            pad2unpad_map[seq_id] = accumulate_len;
        idx += stride;
    }
}

void ComputePadMappingLauncher(const unsigned int num_input_ids,
                               const unsigned int max_seq_len,
                               const Tensor<int> &seq_lens,
                               Tensor<int> &pad2unpad_map,
                               Tensor<int> &unpad2pad_map)
{
    const dim3 nthreads{256 * 512};
    const dim3 threads_per_block{512};
    CUDA_LAUNCH(ComputePadMapping, nthreads, threads_per_block)(num_input_ids,
                                                                max_seq_len,
                                                                seq_lens.data(),
                                                                unpad2pad_map.data(),
                                                                pad2unpad_map.data());
}