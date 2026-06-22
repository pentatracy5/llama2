#pragma once

template <typename T>
class Tensor;

void ComputePadMappingLauncher(const unsigned int num_input_ids,
                               const unsigned int max_seq_len,
                               const Tensor<int> &seq_lens,
                               Tensor<int> &pad2unpad_map,
                               Tensor<int> &unpad2pad_map);