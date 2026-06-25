#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_build_causal_mask(const Tensor<unsigned int> &max_seq_len, // data on CPU or CPU_PINNED
                              const Tensor<unsigned int> &q_lens,
                              const Tensor<unsigned int> &kv_lens,
                              Tensor<T> &mask);
