#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_fused_transpose_unpad(const unsigned int max_q_len,
                                  const Tensor<T> &qkv,                    // (batch_size, q_head_num, q_cache_len, head_size)
                                  const Tensor<unsigned int> &seq_offsets, // (batch_size)
                                  const Tensor<unsigned int> &q_lens,      // (batch_size)
                                  Tensor<T> &unpad_qkv);                   // (num_input_tokens, embed_dim)
