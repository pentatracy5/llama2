#pragma once

struct cublasContext;
typedef struct cublasContext *cublasHandle_t;

template <typename T>
class Tensor;

template <typename T>
void launch_linear(cublasHandle_t &cublas_handle,
                   const bool trans_A,
                   const bool trans_B,
                   const Tensor<T> &A,
                   const Tensor<T> &B,
                   Tensor<T> &C);
