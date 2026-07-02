#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_build_causal_mask(const unsigned int max_q_len,
                              const unsigned int max_kv_len,
                              const Tensor<unsigned int> &q_lens,
                              const Tensor<unsigned int> &kv_lens,
                              Tensor<T> &mask);
