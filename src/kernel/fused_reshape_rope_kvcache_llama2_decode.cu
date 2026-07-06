#include <kernel/fused_reshape_rope_kvcache_llama2_decode.h>
#include <core/Tensor.cuh>
#include <common/config.h>
#include <common/macro.h>
#include <common/types.h>
#include <kernel/rope.cuh>

template <typename T>
__global__ void fused_reshape_rope_kvcache_llama2_decode(const unsigned int batch_size,
                                                         const unsigned int kv_cache_len,
                                                         const unsigned int q_head_num,
                                                         const unsigned int kv_head_num,
                                                         const unsigned char *is_done,
                                                         const unsigned int *kv_lens,
                                                         const T *input,
                                                         T *output_q,
                                                         T *output_k,
                                                         T *output_v)
{
    constexpr unsigned int VEC4_SIZE = 4;
    using VEC4 = typename VecN<T, VEC4_SIZE>;
    constexpr unsigned int HEAD_DIM = WARP_SIZE * VEC4_SIZE; // 128
    constexpr unsigned int H_D = WARP_SIZE;                  // 32
    constexpr unsigned int LANE_GROUP_SIZE = WARP_SIZE / 2;  // 16
    if (H_D != blockDim.x)
        return;

    const VEC4 *embed = reinterpret_cast<const VEC4 *>(input);
    VEC4 *q = reinterpret_cast<VEC4 *>(output_q);
    VEC4 *k = reinterpret_cast<VEC4 *>(output_k);
    VEC4 *v = reinterpret_cast<VEC4 *>(output_v);
    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;
    const unsigned int seq_idx_stride = gridDim.x;
    const unsigned int head_idx_stride = blockDim.y;
    const unsigned int lane_id = threadIdx.x;                                        // 0 - 31
    const unsigned int lane_group_id = lane_id / LANE_GROUP_SIZE;                    // 0 or 1
    const unsigned int lane_id_in_group = lane_id - lane_group_id * LANE_GROUP_SIZE; // 0 - 15

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
                VEC4 x4 = embed[(seq_idx * total_head_num + head_idx) * H_D + lane_id];
                if (head_idx < q_head_num) // q
                {
                    head_dim_128_warp_shuffle_rope<T>(lane_group_id, lane_id_in_group, history_len, HEAD_DIM, kv_len, x4);
                    q[(seq_idx * q_head_num + head_idx) * H_D + lane_id] = x4;
                }
                else if (head_idx >= q_head_num && head_idx < q_head_num + kv_head_num) // k
                {
                    head_dim_128_warp_shuffle_rope<T>(lane_group_id, lane_id_in_group, history_len, HEAD_DIM, kv_len, x4);
                    k[((seq_idx * kv_head_num + head_idx - q_head_num) * kv_cache_len + history_len) * H_D + lane_id] = x4;
                }
                else // v
                {
                    v[((seq_idx * kv_head_num + head_idx - q_head_num - kv_head_num) * kv_cache_len + history_len) * H_D + lane_id] = x4;
                }
                head_idx += head_idx_stride;
            }
        }
        seq_idx += seq_idx_stride;
    }
}

template <typename T>
void launch_fused_reshape_rope_kvcache_llama2_decode(const Tensor<T> &input,
                                                     const Tensor<unsigned char> &is_done,
                                                     const Tensor<unsigned int> &kv_lens,
                                                     Tensor<T> &q,
                                                     Tensor<T> &k,
                                                     Tensor<T> &v)
{
    const unsigned int batch_size = input.shape()[0];
    const unsigned int head_dim = q.shape()[3];
    assert(head_dim == 128 && "Head dim should be 128");
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
    CUDA_LAUNCH(fused_reshape_rope_kvcache_llama2_decode, n_threads, threads_per_block)(batch_size,
                                                                                        kv_cache_len,
                                                                                        q_head_num,
                                                                                        kv_head_num,
                                                                                        is_done.data(),
                                                                                        kv_lens.data(),
                                                                                        input.data(),
                                                                                        q.data(),
                                                                                        k.data(),
                                                                                        v.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_fused_reshape_rope_kvcache_llama2_decode<float>(const Tensor<float> &input,
                                                                     const Tensor<unsigned char> &is_done,
                                                                     const Tensor<unsigned int> &kv_lens,
                                                                     Tensor<float> &q,
                                                                     Tensor<float> &k,
                                                                     Tensor<float> &v);

template void launch_fused_reshape_rope_kvcache_llama2_decode<half>(const Tensor<half> &input,
                                                                    const Tensor<unsigned char> &is_done,
                                                                    const Tensor<unsigned int> &kv_lens,
                                                                    Tensor<half> &q,
                                                                    Tensor<half> &k,
                                                                    Tensor<half> &v);
