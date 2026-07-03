#pragma once

#include <cuda_runtime.h>
#include <common/config.h>
#include <common/types.h>

template <typename T>
__device__ __forceinline__ VecN<T, 2> vec2_rope(const unsigned int token_idx,
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
__device__ __forceinline__ void head_dim_128_warp_shuffle_rope(const unsigned int lane_group_id,
                                                               const unsigned int lane_id_in_group,
                                                               const unsigned int token_idx,
                                                               const unsigned int head_dim,
                                                               const unsigned int seq_len,
                                                               VecN<T, 4> &x4)
{
#pragma unroll 4
    for (unsigned int i = 0; i < 4; i++)
    {
        unsigned int theta_idx = lane_id_in_group * 4 + i;
        VecN<float, 2> x2;
        x2[lane_group_id] = float(x4[i]);
        x2[1 - lane_group_id] = __shfl_xor_sync(0xffff, x2[lane_group_id], 16);
        x2 = vec2_rope<float>(token_idx, theta_idx, head_dim, seq_len, x2);
        x4[i] = x2[lane_group_id];
    }
}