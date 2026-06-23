#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_embedding(const Tensor<int> &input_ids,
                      Tensor<T> &output,
                      const Tensor<T> &embed_table);
