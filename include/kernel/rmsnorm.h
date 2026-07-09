#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_rmsnorm(const Tensor<T> &input_tokens, // (num_input_tokens, embed_dim)
                    const Tensor<T> &weights,      // (embed_dim)
                    Tensor<T> &output_tokens);     // (num_input_tokens, embed_dim)
