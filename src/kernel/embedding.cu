#include <kernel/embedding.h>
#include <core/Tensor.cuh>

template <typename T>
__global__ void Embedding(const int *input_ids,
                          T *output,
                          const T *embed_table,
                          const unsigned int max_context_token_num,
                          const unsigned int embed_dim)
{
    int element_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (element_id < max_context_token_num * embed_dim)
    {
        int token_id = element_id / embed_dim;
        output[element_id] = embed_table[input_ids[token_id] * embed_dim + (element_id - token_id * embed_dim)];
        element_id += stride;
    }
}

template <typename T>
void EmbeddingLauncher(const Tensor<int> &input_ids,
                       Tensor<T> &output,
                       const Tensor<T> &embed_table)
{
    const dim3 nthreads{256 * 512};
    const dim3 threads_per_block{512};
    CUDA_LAUNCH(Embedding, nthreads, threads_per_block)(input_ids.data(),
                                                        output.data(),
                                                        embed_table.data(),
                                                        input_ids.shape()[0],
                                                        embed_table.shape()[1]);
}

template void EmbeddingLauncher<float>(const Tensor<int>&, Tensor<float>&, const Tensor<float>&);
