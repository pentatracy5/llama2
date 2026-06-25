#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_embedding(const Tensor<unsigned int> &input_ids,
                      const Tensor<T> &embed_table,
                      Tensor<T> &output);
