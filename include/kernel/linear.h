#pragma once

struct cublasContext;
typedef struct cublasContext *cublasHandle_t;

template <typename T>
class Tensor;

template <typename T>
void launch_linear(cublasHandle_t &cublas_handle,
                   const bool trans_A,            // for qk dot product example, false
                   const bool trans_B,            // for qk dot product example, true
                   const unsigned int M,          // for qk dot product example, max_q_len
                   const unsigned int N,          // for qk dot product example, max_kv_len
                   const unsigned int K,          // for qk dot product example, head_dim
                   const Tensor<T> &A,            // row major, for qk dot product example, q (batch_size, q_head_num, q_cache_len, head_dim)
                   const Tensor<T> &B,            // row major, for qk dot product example, k (batch_size, kv_head_num, kv_cache_len, head_dim)
                   Tensor<T> &C,                  // for qk dot product example, qk (batch_size, q_head_num, q_cache_len, kv_cache_len)
                   const Tensor<void *> &A_array, // pointers to each 2-D slice of A.
                                                  // For plain 2-D GEMM (dim==2) pass an empty Tensor<void *>.
                   const Tensor<void *> &B_array, // pointers to each 2-D slice of B.
                                                  // For plain 2-D GEMM (dim==2) pass an empty Tensor<void *>.
                   Tensor<void *> &C_array);      // pointers to each 2-D slice of C; used only for batched GEMM (dim > 2).
                                                  // For plain 2-D GEMM (dim==2) pass an empty Tensor<void *>.
