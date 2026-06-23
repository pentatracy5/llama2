#include <kernel/embedding.h>
#include <core/Tensor.cuh>
#include <common/types.h>

template <typename T>
__global__ void embedding_kernel(const unsigned int num_input_ids,
                                 const unsigned int embed_dim,
                                 const int *input_ids,
                                 const T *embed_table,
                                 T *output)
{
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecType<T>::vec_len;
    unsigned int stride = blockDim.x * gridDim.x * VecType<T>::vec_len;
    while (idx < num_input_ids * embed_dim + 1 - VecType<T>::vec_len)
    {
        unsigned int token_id = idx / embed_dim;
        FETCH_VEC(typename VecType<T>::Type, output[idx]) =
            FETCH_VEC(const typename VecType<T>::Type, embed_table[input_ids[token_id] * embed_dim + (idx - token_id * embed_dim)]);
        idx += stride;
    }
}

template <typename T>
void launch_embedding(const Tensor<int> &input_ids,
                      const Tensor<T> &embed_table,
                      Tensor<T> &output)
{
    const dim3 nthreads{256 * 512};
    const dim3 threads_per_block{512};
    CUDA_LAUNCH(embedding_kernel, nthreads, threads_per_block)(input_ids.shape()[0],
                                                               embed_table.shape()[1],
                                                               input_ids.data(),
                                                               embed_table.data(),
                                                               output.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_embedding<float>(const Tensor<int> &, const Tensor<float> &, Tensor<float> &);
template void launch_embedding<__half>(const Tensor<int> &, const Tensor<__half> &, Tensor<__half> &);
