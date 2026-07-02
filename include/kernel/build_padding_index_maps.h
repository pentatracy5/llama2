#pragma once

template <typename T>
class Tensor;

void launch_build_padding_index_maps(const Tensor<unsigned int> &input_ids,     // (num_actual_tokens)
                                     const Tensor<unsigned int> &seq_lens,      // (batch_size)
                                     Tensor<unsigned int> &unpad_to_padded_idx, // (batch_size, q_cache_len)
                                                                                // only the first num_actual_tokens entries are written
                                     Tensor<unsigned int> &seq_offsets);        // (batch_size)
