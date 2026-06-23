#pragma once

template <typename T>
class Tensor;

void launch_build_padding_index_maps(const Tensor<int> &input_ids,
                                     const Tensor<int> &max_seq_len,
                                     const Tensor<int> &seq_lens,
                                     Tensor<int> &unpad_to_padded_idx,
                                     Tensor<int> &seq_offsets);
