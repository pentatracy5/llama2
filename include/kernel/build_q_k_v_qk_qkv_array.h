#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_build_q_k_v_qk_qkv_array(Tensor<T> &q,               // (batch_size, q_head_num, q_cache_len, head_size)
                                     Tensor<T> &k,               // (batch_size, kv_head_num, kv_cache_len, head_size)
                                     Tensor<T> &v,               // (batch_size, kv_head_num, kv_cache_len, head_size)
                                     Tensor<T> &qk,              // (batch_size, q_head_num, q_cache_len, kv_cache_len)
                                     Tensor<T> &qkv,             // (batch_size, q_head_num, q_cache_len, head_size)
                                     Tensor<void *> &q_array,    // (batch_size * q_head_num)
                                     Tensor<void *> &k_array,    // (batch_size * q_head_num)
                                     Tensor<void *> &v_array,    // (batch_size * q_head_num)
                                     Tensor<void *> &qk_array,   // (batch_size * q_head_num)
                                     Tensor<void *> &qkv_array); // (batch_size * q_head_num)
