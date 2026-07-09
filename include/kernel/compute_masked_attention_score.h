#pragma once

template <typename T>
class Tensor;

template <typename T, typename MASK_T>
void launch_compute_masked_attention_score(const unsigned int head_dim,
                                           const unsigned int max_q_len,
                                           const unsigned int max_kv_len,
                                           const Tensor<unsigned int> &q_lens, // (batch_size)
                                           const Tensor<MASK_T> &mask,         // (batch_size, q_cache_len, kv_cache_len)
                                           Tensor<T> &qk);                     // (batch_size, q_head_num, q_cache_len, kv_cache_len)
