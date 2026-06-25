#pragma once

template <typename T>
class Tensor;

void launch_build_padding_index_maps(const Tensor<unsigned int> &input_ids,
                                     const Tensor<unsigned int> &max_seq_len, // data on CPU or CPU_PINNED
                                     const Tensor<unsigned int> &seq_lens,
                                     Tensor<unsigned int> &unpad_to_padded_idx,
                                     Tensor<unsigned int> &seq_offsets);
