#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_fused_pad_reshape_transpose_rope_kvcache_llama2(const Tensor<T> &input,                          // (num_actual_tokens, (q_head_num + 2 * kv_head_num) * head_dim)
                                                            const Tensor<unsigned int> &q_lens,              // (batch_size)
                                                            const Tensor<unsigned int> &kv_lens,             // (batch_size) which includes input seq lens and history lens
                                                            const Tensor<unsigned int> &unpad_to_padded_idx, // (num_actual_tokens) compute from kernel build_padding_index_maps
                                                            Tensor<T> &q,                                    // (batch_size, q_head_num, q_cache_len, head_dim)
                                                            Tensor<T> &k,                                    // (batch_size, kv_head_num, kv_cache_len, head_dim)
                                                            Tensor<T> &v);                                   // (batch_size, kv_head_num, kv_cache_len, head_dim)