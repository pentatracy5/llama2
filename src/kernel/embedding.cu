#include <kernel/embedding.h>
#include <core/Tensor.cuh>
#include <common/types.h>
#include <common/macro.h>
#include <common/config.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T>
__global__ void embedding_kernel(const unsigned int num_input_ids,
                                 const unsigned int embed_dim,
                                 const unsigned int *input_ids,
                                 const T *embed_table,
                                 T *output)
{
    using VECTYPE = typename Vec<T, CUDA_VEC_LS_BYTE_SIZE>;
    constexpr unsigned int vlen = VECTYPE::vec_len;

    const VECTYPE *vec_embed_table = reinterpret_cast<const VECTYPE *>(embed_table);
    VECTYPE *vec_output = reinterpret_cast<VECTYPE *>(output);

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const unsigned int vec_embed_dim = embed_dim / vlen;
    const unsigned int total_vec_nums = num_input_ids * vec_embed_dim;
    while (idx < total_vec_nums)
    {
        unsigned int token_id = idx / vec_embed_dim;
        vec_output[idx] = vec_embed_table[input_ids[token_id] * vec_embed_dim + (idx - token_id * vec_embed_dim)];
        idx += stride;
    }
}

template <typename T>
void launch_embedding(const Tensor<unsigned int> &input_ids,
                      const Tensor<T> &embed_table,
                      Tensor<T> &output)
{
    constexpr unsigned int vlen = Vec<T, CUDA_VEC_LS_BYTE_SIZE>::vec_len;
    const dim3 threads_per_block{THREADS_PER_BLOCK};
    const dim3 nthreads{std::min(NUM_BLOCKS_X * threads_per_block.x, input_ids.shape()[0] * embed_table.shape()[1]) / vlen};
    CUDA_LAUNCH(embedding_kernel<T>, nthreads, threads_per_block)(input_ids.shape()[0],
                                                                  embed_table.shape()[1],
                                                                  input_ids.data(),
                                                                  embed_table.data(),
                                                                  output.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_embedding<float>(const Tensor<unsigned int> &input_ids,
                                      const Tensor<float> &embed_table,
                                      Tensor<float> &output);
template void launch_embedding<half>(const Tensor<unsigned int> &input_ids,
                                     const Tensor<half> &embed_table,
                                     Tensor<half> &output);
