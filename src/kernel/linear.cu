#include <kernel/linear.h>
#include <core/Tensor.cuh>
#include <common/macro.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <numeric>
#include <functional>

template <typename T>
void set_cublas_data_type(cudaDataType &A_type,
                          cudaDataType &B_type,
                          cudaDataType &C_type,
                          cudaDataType &compute_type)
{
    static_assert(false, "Unsupported cublas data type");
}

template <>
void set_cublas_data_type<float>(cudaDataType &A_type,
                                 cudaDataType &B_type,
                                 cudaDataType &C_type,
                                 cudaDataType &compute_type)
{
    A_type = CUDA_R_32F;
    B_type = CUDA_R_32F;
    C_type = CUDA_R_32F;
    compute_type = CUDA_R_32F;
}

template <>
void set_cublas_data_type<half>(cudaDataType &A_type,
                                cudaDataType &B_type,
                                cudaDataType &C_type,
                                cudaDataType &compute_type)
{
    A_type = CUDA_R_16F;
    B_type = CUDA_R_16F;
    C_type = CUDA_R_16F;
    compute_type = CUDA_R_16F;
}

template <typename T>
void launch_linear(cublasHandle_t &cublas_handle,
                   const bool trans_A,
                   const bool trans_B,
                   const unsigned int M,
                   const unsigned int N,
                   const unsigned int K,
                   const Tensor<T> &A,
                   const Tensor<T> &B,
                   Tensor<T> &C,
                   const Tensor<void *> &A_array,
                   const Tensor<void *> &B_array,
                   Tensor<void *> &C_array)
{
    cublasOperation_t op_A = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_B = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int m = N;
    const int n = M;
    const int k = K;

    const T alpha = static_cast<T>(1);
    const T beta = static_cast<T>(0);
    const void *pAlpha = static_cast<const void *>(&alpha);
    const void *pBeta = static_cast<const void *>(&beta);

    cudaDataType A_type;
    cudaDataType B_type;
    cudaDataType C_type;
    cudaDataType compute_type;
    set_cublas_data_type<T>(A_type, B_type, C_type, compute_type);

    const unsigned int dim = C.ndim();
    assert(dim >= 2 && "Matrix dim should be at least 2");
    assert(A.ndim() == dim && "Matrix dim do not match");
    assert(B.ndim() == dim && "Matrix dim do not match");

    const int lda = B.shape()[dim - 1];
    const int ldb = A.shape()[dim - 1];
    const int ldc = C.shape()[dim - 1];

    const int Am = trans_B ? B.shape()[dim - 2] : B.shape()[dim - 1];
    const int Ak = trans_B ? B.shape()[dim - 1] : B.shape()[dim - 2];
    const int Bk = trans_A ? A.shape()[dim - 2] : A.shape()[dim - 1];
    const int Bn = trans_A ? A.shape()[dim - 1] : A.shape()[dim - 2];
    const int Cm = C.shape()[dim - 1];
    const int Cn = C.shape()[dim - 2];
    assert(Am >= m && "M dim of Matrix B is illegal");
    assert(Ak >= k && "K dim of Matrix B is illegal");
    assert(Bk >= k && "K dim of Matrix A is illegal");
    assert(Bn >= n && "N dim of Matrix A is illegal");
    assert(Cm >= m && "M dim of Matrix C is illegal");
    assert(Cn >= n && "N dim of Matrix C is illegal");

    if (dim > 2)
    {
        const int batch_count = static_cast<int>(std::accumulate(C.shape().begin(), C.shape().end() - 2, static_cast<unsigned int>(1), std::multiplies<unsigned int>()));

        assert(batch_count == A_array.numel() && "Illegal A array size");
        assert(batch_count == B_array.numel() && "Illegal B array size");
        assert(batch_count == C_array.numel() && "Illegal C array size");

        const void *const *A_arr = B_array.data();
        const void *const *B_arr = A_array.data();
        void *const *C_arr = C_array.data();

        CUBLAS_CHECK(cublasGemmBatchedEx(cublas_handle,
                                         op_A,
                                         op_B,
                                         m,
                                         n,
                                         k,
                                         pAlpha,
                                         A_arr,
                                         A_type,
                                         lda,
                                         B_arr,
                                         B_type,
                                         ldb,
                                         pBeta,
                                         C_arr,
                                         C_type,
                                         ldc,
                                         batch_count,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT));
    }
    else
    {
        const void *pA = static_cast<const void *>(B.data());
        const void *pB = static_cast<const void *>(A.data());
        void *pC = static_cast<void *>(C.data());

        CUBLAS_CHECK(cublasGemmEx(cublas_handle,
                                  op_A,
                                  op_B,
                                  m,
                                  n,
                                  k,
                                  pAlpha,
                                  pA,
                                  A_type,
                                  lda,
                                  pB,
                                  B_type,
                                  ldb,
                                  pBeta,
                                  pC,
                                  C_type,
                                  ldc,
                                  compute_type,
                                  CUBLAS_GEMM_DEFAULT));
    }
}

template void launch_linear<float>(cublasHandle_t &cublas_handle,
                                   const bool trans_A,
                                   const bool trans_B,
                                   const unsigned int M,
                                   const unsigned int N,
                                   const unsigned int K,
                                   const Tensor<float> &A,
                                   const Tensor<float> &B,
                                   Tensor<float> &C,
                                   const Tensor<void *> &A_array,
                                   const Tensor<void *> &B_array,
                                   Tensor<void *> &C_array);

template void launch_linear<half>(cublasHandle_t &cublas_handle,
                                  const bool trans_A,
                                  const bool trans_B,
                                  const unsigned int M,
                                  const unsigned int N,
                                  const unsigned int K,
                                  const Tensor<half> &A,
                                  const Tensor<half> &B,
                                  Tensor<half> &C,
                                  const Tensor<void *> &A_array,
                                  const Tensor<void *> &B_array,
                                  Tensor<void *> &C_array);
