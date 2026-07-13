#pragma once

template <typename T>
class Tensor;

//    residual                    x
//       |                        |
//       |_________________ add_residual ? x + residual writeback to x : do nothing
//                                |
//                            rmsnorm(x)
//        ________________________|
//       |
//       |writeback
//       |
//    residual
template <typename T, bool add_residual>
void launch_fused_add_residual_rmsnorm(const Tensor<T> &weights, // (embed_dim)
                                       Tensor<T> &x,             // (num_input_tokens, embed_dim)
                                       Tensor<T> &residual);     // (num_input_tokens, embed_dim)
