#include <kernel/fused_reshape_rope_kvcache_decode.h>
#include <kernel/rope.cuh>
#include <core/Tensor.cuh>
#include <common/types.h>
#include <common/macro.h>
#include <common/config.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T>
__global__ void fused_reshape_rope_kvcache_decode(const unsigned int batch_size,
                                                  const unsigned int kv_cache_len,
                                                  const unsigned int q_head_num,
                                                  const unsigned int kv_head_num,
                                                  const unsigned int head_dim,
                                                  const unsigned char *is_done,
                                                  const unsigned int *kv_lens,
                                                  const T *input,
                                                  T *output_q,
                                                  T *output_k,
                                                  T *output_v)
{
    constexpr unsigned int VEC2_SIZE = 2;
    using VEC2 = typename VecN<T, VEC2_SIZE>;

    const VEC2 *embed = reinterpret_cast<const VEC2 *>(input);
    VEC2 *q = reinterpret_cast<VEC2 *>(output_q);
    VEC2 *k = reinterpret_cast<VEC2 *>(output_k);
    VEC2 *v = reinterpret_cast<VEC2 *>(output_v);
    const unsigned int h_d = head_dim / VEC2_SIZE;
    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;
    const unsigned int seq_idx_stride = gridDim.x;
    const unsigned int head_idx_stride = blockDim.y;
    const unsigned int idx_in_head_stride = blockDim.x;

    unsigned int seq_idx = blockIdx.x;
    while (seq_idx < batch_size)
    {
        const bool done = is_done[seq_idx];
        if (!done)
        {
            const unsigned int kv_len = kv_lens[seq_idx];
            const unsigned int history_len = kv_len - 1;

            unsigned int head_idx = threadIdx.y;
            while (head_idx < total_head_num)
            {
                unsigned int idx_in_head = threadIdx.x;
                if (head_idx < q_head_num) // q
                {
                    while (idx_in_head < h_d)
                    {
                        VEC2 x2 = embed[(seq_idx * total_head_num + head_idx) * h_d + idx_in_head];
                        x2 = vec2_rope<T>(history_len, idx_in_head, head_dim, kv_len, x2);
                        q[((seq_idx * q_head_num + head_idx) * 1) * h_d + idx_in_head] = x2;
                        idx_in_head += idx_in_head_stride;
                    }
                }
                else if (head_idx >= q_head_num && head_idx < q_head_num + kv_head_num) // k
                {
                    while (idx_in_head < h_d)
                    {
                        VEC2 x2 = embed[(seq_idx * total_head_num + head_idx) * h_d + idx_in_head];
                        x2 = vec2_rope<T>(history_len, idx_in_head, head_dim, kv_len, x2);
                        k[((seq_idx * kv_head_num + head_idx - q_head_num) * kv_cache_len + history_len) * h_d + idx_in_head] = x2;
                        idx_in_head += idx_in_head_stride;
                    }
                }
                else // v
                {
                    while (idx_in_head < h_d)
                    {
                        VEC2 x2 = embed[(seq_idx * total_head_num + head_idx) * h_d + idx_in_head];
                        v[((seq_idx * kv_head_num + head_idx - q_head_num - kv_head_num) * kv_cache_len + history_len) * h_d + idx_in_head] = x2;
                        idx_in_head += idx_in_head_stride;
                    }
                }
                head_idx += head_idx_stride;
            }
        }
        seq_idx += seq_idx_stride;
    }
}

template <typename T>
void launch_fused_reshape_rope_kvcache_decode(const Tensor<T> &input,
                                              const Tensor<unsigned char> &is_done,
                                              const Tensor<unsigned int> &kv_lens,
                                              Tensor<T> &q,
                                              Tensor<T> &k,
                                              Tensor<T> &v)
{
    const unsigned int batch_size = input.shape()[0];
    const unsigned int head_dim = q.shape()[3];
    assert(head_dim % 2 == 0 && "Head dim should be even");
    assert(head_dim == k.shape()[3] && "Head dim of q k do not match");
    assert(head_dim == v.shape()[3] && "Head dim of q v do not match");
    const unsigned int q_head_num = q.shape()[1];
    const unsigned int kv_head_num = k.shape()[1];
    assert(kv_head_num == v.shape()[1] && "Head num of k v do not match");
    assert(input.shape()[1] == (q_head_num + 2 * kv_head_num) * head_dim && "Incorrect embedding dim");
    assert(batch_size == is_done.shape()[0] && "Incorrect batch size of done flag");
    assert(batch_size == kv_lens.shape()[0] && "Incorrect batch size of sequence length");
    assert(batch_size == q.shape()[0] && "Batch size of q do not match");
    assert(batch_size == k.shape()[0] && "Batch size of k do not match");
    assert(batch_size == v.shape()[0] && "Batch size of v do not match");
    assert(1 == q.shape()[2] && "Incorrect q cache lenght");
    const unsigned int kv_cache_len = k.shape()[2];
    assert(kv_cache_len == v.shape()[2] && "k v cache length do not match");
    const dim3 threads_per_block{WARP_SIZE, THREADS_PER_BLOCK / WARP_SIZE};
    const dim3 n_threads{std::min(NUM_BLOCKS_X, batch_size) * threads_per_block.x, threads_per_block.y};
    CUDA_LAUNCH(fused_reshape_rope_kvcache_decode<T>, n_threads, threads_per_block)(batch_size,
                                                                                    kv_cache_len,
                                                                                    q_head_num,
                                                                                    kv_head_num,
                                                                                    head_dim,
                                                                                    is_done.data(),
                                                                                    kv_lens.data(),
                                                                                    input.data(),
                                                                                    q.data(),
                                                                                    k.data(),
                                                                                    v.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_fused_reshape_rope_kvcache_decode<float>(const Tensor<float> &input,
                                                              const Tensor<unsigned char> &is_done,
                                                              const Tensor<unsigned int> &kv_lens,
                                                              Tensor<float> &q,
                                                              Tensor<float> &k,
                                                              Tensor<float> &v);

template void launch_fused_reshape_rope_kvcache_decode<half>(const Tensor<half> &input,
                                                             const Tensor<unsigned char> &is_done,
                                                             const Tensor<unsigned int> &kv_lens,
                                                             Tensor<half> &q,
                                                             Tensor<half> &k,
                                                             Tensor<half> &v);
