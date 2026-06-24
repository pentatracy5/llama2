#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_rmsnorm(const Tensor<T> &input_tokens,
                    const Tensor<T> &weights,
                    Tensor<T> &output_tokens);
