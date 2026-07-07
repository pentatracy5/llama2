#include <kernel/build_q_k_v_qk_qkv_array.h>
#include <core/Tensor.cuh>
#include <common/config.h>

template <typename T>
__global__ void build_q_k_v_qk_qkv_array(const unsigned int batch_size,
                                         const unsigned int kv_head_num,
                                         const unsigned int q_heads_per_kv_head,
                                         const unsigned int stride_q,
                                         const unsigned int stride_k,
                                         const unsigned int stride_v,
                                         const unsigned int stride_qk,
                                         const unsigned int stride_qkv,
                                         T *q,
                                         T *k,
                                         T *v,
                                         T *qk,
                                         T *qkv,
                                         void **q_array,
                                         void **k_array,
                                         void **v_array,
                                         void **qk_array,
                                         void **qkv_array)
{
    unsigned int batch_id = blockIdx.x;
    while (batch_id < batch_size)
    {
        unsigned int kv_head_id = threadIdx.y;
        while (kv_head_id < kv_head_num)
        {
            void *sub_k = static_cast<void *>(k + (batch_id * kv_head_num + kv_head_id) * stride_k);
            void *sub_v = static_cast<void *>(v + (batch_id * kv_head_num + kv_head_id) * stride_v);
            unsigned int q_head_id_in_kv_head = threadIdx.x;
            while (q_head_id_in_kv_head < q_heads_per_kv_head)
            {
                unsigned int idx = (batch_id * kv_head_num + kv_head_id) * q_heads_per_kv_head + q_head_id_in_kv_head;
                q_array[idx] = static_cast<void *>(q + idx * stride_q);
                k_array[idx] = sub_k;
                v_array[idx] = sub_v;
                qk_array[idx] = static_cast<void *>(qk + idx * stride_qk);
                qkv_array[idx] = static_cast<void *>(qkv + idx * stride_qkv);
                q_head_id_in_kv_head += blockDim.x;
            }
            kv_head_id += blockDim.y;
        }
        batch_id += gridDim.x;
    }
}

template <typename T>
void launch_build_q_k_v_qk_qkv_array(Tensor<T> &q,
                                     Tensor<T> &k,
                                     Tensor<T> &v,
                                     Tensor<T> &qk,
                                     Tensor<T> &qkv,
                                     Tensor<void *> &q_array,
                                     Tensor<void *> &k_array,
                                     Tensor<void *> &v_array,
                                     Tensor<void *> &qk_array,
                                     Tensor<void *> &qkv_array)
{
    const unsigned int batch_size = q.shape()[0];
    const unsigned int q_head_num = q.shape()[1];
    const unsigned int kv_head_num = k.shape()[1];
    assert(q_head_num % kv_head_num == 0 && "q_head_num is not divisible by kv_head_num");
    const unsigned int q_heads_per_kv_head = q_head_num / kv_head_num;
    assert(q_array.numel() == batch_size * q_head_num && "Incorrect q_array size");
    assert(k_array.numel() == batch_size * q_head_num && "Incorrect k_array size");
    assert(v_array.numel() == batch_size * q_head_num && "Incorrect v_array size");
    assert(qk_array.numel() == batch_size * q_head_num && "Incorrect qk_array size");
    assert(qkv_array.numel() == batch_size * q_head_num && "Incorrect qkv_array size");

    const unsigned int stride_q = q.shape()[2] * q.shape()[3];
    const unsigned int stride_k = k.shape()[2] * k.shape()[3];
    const unsigned int stride_v = v.shape()[2] * v.shape()[3];
    const unsigned int stride_qk = qk.shape()[2] * qk.shape()[3];
    const unsigned int stride_qkv = qkv.shape()[2] * qkv.shape()[3];

    const dim3 threads_per_block{q_heads_per_kv_head, std::min(std::max(WARP_SIZE / q_heads_per_kv_head, 1u), kv_head_num)};
    const dim3 n_threads{std::min(batch_size, NUM_BLOCKS_X) * threads_per_block.x, threads_per_block.y};
    CUDA_LAUNCH(build_q_k_v_qk_qkv_array, n_threads, threads_per_block)(batch_size,
                                                                        kv_head_num,
                                                                        q_heads_per_kv_head,
                                                                        stride_q,
                                                                        stride_k,
                                                                        stride_v,
                                                                        stride_qk,
                                                                        stride_qkv,
                                                                        q.data(),
                                                                        k.data(),
                                                                        v.data(),
                                                                        qk.data(),
                                                                        qkv.data(),
                                                                        q_array.data(),
                                                                        k_array.data(),
                                                                        v_array.data(),
                                                                        qk_array.data(),
                                                                        qkv_array.data());
    CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_build_q_k_v_qk_qkv_array<float>(Tensor<float> &q,
                                                     Tensor<float> &k,
                                                     Tensor<float> &v,
                                                     Tensor<float> &qk,
                                                     Tensor<float> &qkv,
                                                     Tensor<void *> &q_array,
                                                     Tensor<void *> &k_array,
                                                     Tensor<void *> &v_array,
                                                     Tensor<void *> &qk_array,
                                                     Tensor<void *> &qkv_array);

template void launch_build_q_k_v_qk_qkv_array<half>(Tensor<half> &q,
                                                    Tensor<half> &k,
                                                    Tensor<half> &v,
                                                    Tensor<half> &qk,
                                                    Tensor<half> &qkv,
                                                    Tensor<void *> &q_array,
                                                    Tensor<void *> &k_array,
                                                    Tensor<void *> &v_array,
                                                    Tensor<void *> &qk_array,
                                                    Tensor<void *> &qkv_array);
