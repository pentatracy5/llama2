#include <kernel/embedding.h>
#include <core/Tensor.cuh>

template <typename T>
__global__ void Embedding(const unsigned int num_input_ids,
                          const unsigned int embed_dim,
                          const int *input_ids,
                          const T *embed_table,
                          T *output)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (idx < num_input_ids * embed_dim)
    {
        unsigned int token_id = idx / embed_dim;
        output[idx] = embed_table[input_ids[token_id] * embed_dim + (idx - token_id * embed_dim)];
        idx += stride;
    }
}

template <typename T>
void EmbeddingLauncher(const Tensor<int> &input_ids,
                       Tensor<T> &output,
                       const Tensor<T> &embed_table)
{
    const dim3 nthreads{256 * 512};
    const dim3 threads_per_block{512};
    CUDA_LAUNCH(Embedding, nthreads, threads_per_block)(input_ids.shape()[0],
                                                        embed_table.shape()[1],
                                                        input_ids.data(),
                                                        embed_table.data(),
                                                        output.data());
}

template void EmbeddingLauncher<float>(const Tensor<int> &, Tensor<float> &, const Tensor<float> &);
