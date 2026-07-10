#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_fused_transpose_unpad(const unsigned int num_input_tokens,
                                  const Tensor<T> &qkv,                         // (batch_size, q_head_num, q_cache_len, head_size)
                                  const Tensor<unsigned int> &unpad_to_pad_idx, // (batch_size, q_cache_len)
                                                                                // only the first num_input_tokens entries are written
                                  Tensor<T> &unpad_qkv);                        // (num_input_tokens, embed_dim)
