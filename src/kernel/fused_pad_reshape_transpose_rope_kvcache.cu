#include <kernel/fused_pad_reshape_transpose_rope_kvcache.h>
#include <core/Tensor.cuh>
#include <common/config.h>
#include <common/macro.h>
#include <common/types.h>

template <typename T>
__device__ VecN<T, 2> rotation(const unsigned int token_idx,
                               const unsigned int theta_idx,
                               const unsigned int head_dim,
                               const unsigned int seq_len,
                               const VecN<T, 2> &input)
{
    float theta = 1.f / powf(ROPE_BASE, 2.f * theta_idx / head_dim);
    float alpha = 1.f / powf(fmaxf(1.f, float(seq_len) / TRAIN_SEQ_LEN), 2.f * theta_idx / (head_dim - 2));
    float cos_value = cosf(token_idx * alpha * theta);
    float sin_value = sinf(token_idx * alpha * theta);
    VecN<T, 2> output;
    output[0] = T(cos_value * float(input[0]) - sin_value * float(input[1]));
    output[1] = T(sin_value * float(input[0]) + cos_value * float(input[1]));
    return output;
}

template <typename T>
__global__ void fused_pad_reshape_transpose_rope_kvcache(const unsigned int num_actual_tokens,
                                                         const unsigned int q_cache_len,
                                                         const unsigned int kv_cache_len,
                                                         const unsigned int q_head_num,
                                                         const unsigned int kv_head_num,
                                                         const unsigned int head_dim,
                                                         const unsigned int *q_lens,
                                                         const unsigned int *kv_lens,
                                                         const unsigned int *unpad_to_padded_idx,
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
    const unsigned int idx_stride = gridDim.x;
    const unsigned int head_idx_stride = blockDim.y;
    const unsigned int idx_in_head_stride = blockDim.x;
    unsigned int idx = blockIdx.x;
    while (idx < num_actual_tokens)
    {
        const unsigned int padded_idx = unpad_to_padded_idx[idx];
        const unsigned int seq_idx = padded_idx / q_cache_len;
        const unsigned int token_idx = padded_idx - q_cache_len * seq_idx;
        const unsigned int kv_len = kv_lens[seq_idx];
        const unsigned int history_len = kv_len - q_lens[seq_idx];
        unsigned int head_idx = threadIdx.y;
        while (head_idx < total_head_num)
        {
            unsigned int idx_in_head = threadIdx.x;
            if (head_idx < q_head_num) // q
            {
                while (idx_in_head < h_d)
                {
                    VEC2 x2 = embed[(idx * total_head_num + head_idx) * h_d + idx_in_head];
                    x2 = rotation<T>(token_idx + history_len, idx_in_head, head_dim, kv_len, x2);
                    q[((seq_idx * q_head_num + head_idx) * q_cache_len + token_idx) * h_d + idx_in_head] = x2;
                    idx_in_head += idx_in_head_stride;
                }
            }
            else if (head_idx >= q_head_num && head_idx < q_head_num + kv_head_num) // k
            {
                while (idx_in_head < h_d)
                {
                    VEC2 x2 = embed[(idx * total_head_num + head_idx) * h_d + idx_in_head];
                    x2 = rotation<T>(token_idx + history_len, idx_in_head, head_dim, kv_len, x2);
                    k[((seq_idx * kv_head_num + head_idx - q_head_num) * kv_cache_len + token_idx + history_len) * h_d + idx_in_head] = x2;
                    idx_in_head += idx_in_head_stride;
                }
            }
            else // v
            {
                while (idx_in_head < h_d)
                {
                    VEC2 x2 = embed[(idx * total_head_num + head_idx) * h_d + idx_in_head];
                    v[((seq_idx * kv_head_num + head_idx - q_head_num - kv_head_num) * kv_cache_len + token_idx + history_len) * h_d + idx_in_head] = x2;
                    idx_in_head += idx_in_head_stride;
                }
            }
            head_idx += head_idx_stride;
        }
        idx += idx_stride;
    }
}

template <typename T>
void launch_fused_pad_reshape_transpose_rope_kvcache(const Tensor<T> &input,
                                                     const Tensor<unsigned int> &q_lens,
                                                     const Tensor<unsigned int> &kv_lens,
                                                     const Tensor<unsigned int> &unpad_to_padded_idx,
                                                     Tensor<T> &q,
                                                     Tensor<T> &k,
                                                     Tensor<T> &v)
{
    const unsigned int num_actual_tokens = input.shape()[0];
    const unsigned int head_dim = q.shape()[3];
    assert(head_dim % 2 == 0 && "Head dim should be even");
    assert(head_dim == k.shape()[3] && "Head dim of q k do not match");
    assert(head_dim == v.shape()[3] && "Head dim of q v do not match");
    const unsigned int q_head_num = q.shape()[1];
    const unsigned int kv_head_num = k.shape()[1];
    assert(kv_head_num == v.shape()[1] && "Head num of k v do not match");
    assert(input.shape()[1] == (q_head_num + 2 * kv_head_num) * head_dim && "Incorrect embedding dim");
    assert(q_lens.shape()[0] == kv_lens.shape()[0] && "Incorrect batch size of sequence length");
    assert(q_lens.shape()[0] == q.shape()[0] && "Batch size of q do not match");
    assert(q_lens.shape()[0] == k.shape()[0] && "Batch size of k do not match");
    assert(q_lens.shape()[0] == v.shape()[0] && "Batch size of v do not match");
    const unsigned int q_cache_len = q.shape()[2];
    const unsigned int kv_cache_len = k.shape()[2];
    assert(kv_cache_len == v.shape()[2] && "k v cache length do not match");
    const dim3 threads_per_block{WARP_SIZE, THREADS_PER_BLOCK / WARP_SIZE};
    const dim3 n_threads{std::min(NUM_BLOCKS_X, num_actual_tokens) * threads_per_block.x, threads_per_block.y};
    CUDA_LAUNCH(fused_pad_reshape_transpose_rope_kvcache, n_threads, threads_per_block)(num_actual_tokens,
                                                                                        q_cache_len,
                                                                                        kv_cache_len,
                                                                                        q_head_num,
                                                                                        kv_head_num,
                                                                                        head_dim,
                                                                                        q_lens.data(),
                                                                                        kv_lens.data(),
                                                                                        unpad_to_padded_idx.data(),
                                                                                        input.data(),
                                                                                        q.data(),
                                                                                        k.data(),
                                                                                        v.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_fused_pad_reshape_transpose_rope_kvcache<float>(const Tensor<float> &input,
                                                                     const Tensor<unsigned int> &q_lens,
                                                                     const Tensor<unsigned int> &kv_lens,
                                                                     const Tensor<unsigned int> &unpad_to_padded_idx,
                                                                     Tensor<float> &q,
                                                                     Tensor<float> &k,
                                                                     Tensor<float> &v);

template void launch_fused_pad_reshape_transpose_rope_kvcache<half>(const Tensor<half> &input,
                                                                    const Tensor<unsigned int> &q_lens,
                                                                    const Tensor<unsigned int> &kv_lens,
                                                                    const Tensor<unsigned int> &unpad_to_padded_idx,
                                                                    Tensor<half> &q,
                                                                    Tensor<half> &k,
                                                                    Tensor<half> &v);