#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_build_causal_mask(const unsigned int max_q_len,
                              const unsigned int max_kv_len,
                              const Tensor<unsigned int> &q_lens,  // (batch_size)
                              const Tensor<unsigned int> &kv_lens, // (batch_size)
                              Tensor<T> &mask);                    // (batch_size, q_cache_len, kv_cache_len)
