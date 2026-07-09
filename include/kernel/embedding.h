#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_embedding(const Tensor<unsigned int> &input_ids, // (num_input_tokens)
                      const Tensor<T> &embed_table,          // (vocab_size, embed_dim)
                      Tensor<T> &output);                    // (num_input_tokens, embed_dim)
