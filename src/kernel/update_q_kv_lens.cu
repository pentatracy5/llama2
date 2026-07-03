#include <kernel/update_q_kv_lens.h>
#include <core/Tensor.cuh>
#include <common/macro.h>
#include <common/config.h>
#include <kernel/reduction.cuh>

__global__ void update_q_kv_lens(const unsigned int batch_size,
                                 const unsigned int *q_lens,
                                 unsigned int *kv_lens,
                                 unsigned int *max_q_len,
                                 unsigned int *max_kv_len)
{
    extern __shared__ unsigned int smem[];
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % WARP_SIZE;
    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int x = tid < batch_size ? q_lens[tid] : 0;
    x = shuffle_warp_reduce<WARP_SIZE, unsigned int, MaxOp>(x);
    if (lane_id == 0)
        smem[warp_id] = x;
    __syncthreads();
    if (tid < WARPS_PER_BLOCK)
        x = shuffle_warp_reduce<WARPS_PER_BLOCK, unsigned int, MaxOp>(smem[tid]);
    __syncthreads();
    if (tid == 0)
        max_q_len[0] = x;
    x = tid < batch_size ? kv_lens[tid] + q_lens[tid] : 0;
    if (tid < batch_size)
        kv_lens[tid] = x;
    x = shuffle_warp_reduce<WARP_SIZE, unsigned int, MaxOp>(x);
    if (lane_id == 0)
        smem[warp_id] = x;
    __syncthreads();
    if (tid < WARPS_PER_BLOCK)
        x = shuffle_warp_reduce<WARPS_PER_BLOCK, unsigned int, MaxOp>(smem[tid]);
    __syncthreads();
    if (tid == 0)
        max_kv_len[0] = x;
}

void launch_update_q_kv_lens(const Tensor<unsigned int> &q_lens,
                             Tensor<unsigned int> &kv_lens,
                             Tensor<unsigned int> &max_q_len,
                             Tensor<unsigned int> &max_kv_len)
{
    const dim3 threads_per_block{THREADS_PER_BLOCK};
    const dim3 n_threads{threads_per_block};
    const unsigned int shared_mem_bytes = WARPS_PER_BLOCK * sizeof(unsigned int);
    assert(q_lens.shape()[0] <= threads_per_block.x && "Batch size is too large");
    max_q_len.to_device();
    max_kv_len.to_device();
    CUDA_LAUNCH_SHAREDMEM(update_q_kv_lens, n_threads, threads_per_block, shared_mem_bytes)(q_lens.shape()[0],
                                                                                            q_lens.data(),
                                                                                            kv_lens.data(),
                                                                                            max_q_len.data(),
                                                                                            max_kv_len.data());
    CUDA_KERNEL_LAUNCH_CHECK();
    max_q_len.to_host_pinned();
    max_kv_len.to_host_pinned();
}