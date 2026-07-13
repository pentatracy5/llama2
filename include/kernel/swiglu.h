#pragma once

template <typename T>
class Tensor;

template <typename T>
void launch_swiglu(const Tensor<T> &input, // (num_input_tokens, 2 * intermediate_size)
                   Tensor<T> &output);     // (num_input_tokens, intermediate_size)