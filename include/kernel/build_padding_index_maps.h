#pragma once

template <typename T>
class Tensor;

void launch_build_padding_index_maps(const unsigned int num_input_tokens,
                                     const Tensor<unsigned int> &seq_lens,   // (batch_size)
                                     Tensor<unsigned int> &unpad_to_pad_idx, // (batch_size, q_cache_len)
                                                                             // only the first num_input_tokens entries are written
                                     Tensor<unsigned int> &seq_offsets);     // (batch_size)
