#pragma once

template <typename T>
class Tensor;

void launch_update_q_kv_lens(const Tensor<unsigned int> &q_lens, // (batch_size)
                             Tensor<unsigned int> &kv_lens,      // (batch_size)
                             Tensor<unsigned int> &max_q_len,    // (1)
                             Tensor<unsigned int> &max_kv_len);  // (1)