#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_fused_reshape_rope_kvcache_decode(const Tensor<T> &input,               // (batch_size, (q_head_num + 2 * kv_head_num) * head_dim)
                                              const Tensor<unsigned char> &is_done, // (batch_size)
                                              const Tensor<unsigned int> &kv_lens,  // (batch_size) which includes input seq lens and history lens
                                              Tensor<T> &q,                         // (batch_size, q_head_num, 1, head_dim)
                                              Tensor<T> &k,                         // (batch_size, kv_head_num, kv_cache_len, head_dim)
                                              Tensor<T> &v);                        // (batch_size, kv_head_num, kv_cache_len, head_dim)
