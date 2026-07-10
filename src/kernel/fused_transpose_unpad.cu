#include <kernel/fused_transpose_unpad.h>
#include <core/Tensor.cuh>
#include <common/macro.h>
#include <common/config.h>
#include <common/types.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

template <typename T>
__global__ void fused_transpose_unpad(const unsigned int num_input_tokens,
                                      const unsigned int q_head_num,
                                      const unsigned int q_cache_len,
                                      const unsigned int head_size,
                                      const unsigned int *unpad_to_pad_idx,
                                      const T *qkv,
                                      T *unpad_qkv)
{
    using VECTYPE = typename Vec<T, CUDA_VEC_LS_BYTE_SIZE>;
    constexpr unsigned int vlen = VECTYPE::vec_len;

    const VECTYPE *vec_qkv = reinterpret_cast<const VECTYPE *>(qkv);
    VECTYPE *vec_unpad_qkv = reinterpret_cast<VECTYPE *>(unpad_qkv);
    const unsigned int h_d = head_size / vlen;

    unsigned int global_token_id = blockIdx.y;
    while (global_token_id < num_input_tokens)
    {
        const unsigned int seq_id = unpad_to_pad_idx[global_token_id] / q_cache_len;
        const unsigned int token_id = unpad_to_pad_idx[global_token_id] - seq_id * q_cache_len;

        unsigned int head_id = blockIdx.x;
        while (head_id < q_head_num)
        {
            unsigned int idx_in_head = threadIdx.x;
            while (idx_in_head < h_d)
            {
                const unsigned int vec_unpad_qkv_offset = (global_token_id * q_head_num + head_id) * h_d + idx_in_head;
                const unsigned int vec_qkv_offset = ((seq_id * q_head_num + head_id) * q_cache_len + token_id) * h_d + idx_in_head;
                vec_unpad_qkv[vec_unpad_qkv_offset] = vec_qkv[vec_qkv_offset];
                idx_in_head += blockDim.x;
            }
            head_id += gridDim.x;
        }
        global_token_id += gridDim.y;
    }
}

template <typename T>
void launch_fused_transpose_unpad(const unsigned int num_input_tokens,
                                  const Tensor<T> &qkv,
                                  const Tensor<unsigned int> &unpad_to_pad_idx,
                                  Tensor<T> &unpad_qkv)
{
    constexpr unsigned int vlen = Vec<T, CUDA_VEC_LS_BYTE_SIZE>::vec_len;
    const unsigned int q_head_num = qkv.shape()[1];
    const unsigned int q_cache_len = qkv.shape()[2];
    const unsigned int head_size = qkv.shape()[3];
    assert(unpad_qkv.shape()[1] == q_head_num * head_size && "Incorrect embed dim");
    assert(head_size % vlen == 0 && "head_size must be divisible by vector length");
    const dim3 threads_per_block{std::min(THREADS_PER_BLOCK, (head_size / vlen + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE)};
    const dim3 n_threads{std::min(NUM_BLOCKS_X, q_head_num) * threads_per_block.x,
                         std::min(NUM_BLOCKS_Y, num_input_tokens) * threads_per_block.y};
    CUDA_LAUNCH(fused_transpose_unpad, n_threads, threads_per_block)(num_input_tokens,
                                                                     q_head_num,
                                                                     q_cache_len,
                                                                     head_size,
                                                                     unpad_to_pad_idx.data(),
                                                                     qkv.data(),
                                                                     unpad_qkv.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_fused_transpose_unpad<float>(const unsigned int num_input_tokens,
                                                  const Tensor<float> &qkv,
                                                  const Tensor<unsigned int> &unpad_to_pad_idx,
                                                  Tensor<float> &unpad_qkv);

template void launch_fused_transpose_unpad<half>(const unsigned int num_input_tokens,
                                                 const Tensor<half> &qkv,
                                                 const Tensor<unsigned int> &unpad_to_pad_idx,
                                                 Tensor<half> &unpad_qkv);
