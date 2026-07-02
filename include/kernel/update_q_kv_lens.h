#pragma once

template <typename T>
class Tensor;

void launch_update_q_kv_lens(const Tensor<unsigned int> &q_lens,
                             Tensor<unsigned int> &kv_lens,
                             Tensor<unsigned int> &max_q_len,
                             Tensor<unsigned int> &max_kv_len);