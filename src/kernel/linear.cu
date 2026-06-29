#include <kernel/linear.h>
#include <core/Tensor.cuh>
#include <common/macro.h>
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
                   const Tensor<T> &A,
                   const Tensor<T> &B,
                   Tensor<T> &C)
{
    cudaDataType A_type;
    cudaDataType B_type;
    cudaDataType C_type;
    cudaDataType compute_type;
    set_cublas_data_type<T>(A_type, B_type, C_type, compute_type);

    cublasOperation_t op_A = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_B = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;

    const unsigned int dim = A.ndim();
    assert(dim >= 2 && "Matrix dim should be at least 2");
    assert(B.ndim() == dim && "Matrix dim do not match");
    assert(C.ndim() == dim && "Matrix dim do not match");

    const int lda = B.shape()[dim - 1];
    const int ldb = A.shape()[dim - 1];
    const int ldc = C.shape()[dim - 1];

    const int Am = trans_B ? B.shape()[dim - 2] : B.shape()[dim - 1];
    const int Ak = trans_B ? B.shape()[dim - 1] : B.shape()[dim - 2];
    const int Bk = trans_A ? A.shape()[dim - 2] : A.shape()[dim - 1];
    const int Bn = trans_A ? A.shape()[dim - 1] : A.shape()[dim - 2];
    const int Cm = C.shape()[dim - 1];
    const int Cn = C.shape()[dim - 2];
    assert(Am == Cm && "N of Matrix B C do not match");
    assert(Bn == Cn && "M of Matrix A C do not match");
    assert(Ak == Bk && "K of Matrix A B do not match");

    const long long int stride_A = static_cast<long long int>(Am * Ak);
    const long long int stride_B = static_cast<long long int>(Bk * Bn);
    const long long int stride_C = static_cast<long long int>(Cm * Cn);
    const int batch_count = static_cast<int>(std::accumulate(A.shape().begin(), A.shape().end() - 2, static_cast<unsigned int>(1), std::multiplies<unsigned int>()));

    const T alpha = static_cast<T>(1);
    const T beta = static_cast<T>(0);
    const void *pA = static_cast<const void *>(B.data());
    const void *pB = static_cast<const void *>(A.data());
    void *pC = static_cast<void *>(C.data());
    const void *pAlpha = static_cast<const void *>(&alpha);
    const void *pBeta = static_cast<const void *>(&beta);

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas_handle,
                                            op_A,
                                            op_B,
                                            Am,
                                            Cn,
                                            Bk,
                                            pAlpha,
                                            pA,
                                            A_type,
                                            lda,
                                            stride_A,
                                            pB,
                                            B_type,
                                            ldb,
                                            stride_B,
                                            pBeta,
                                            pC,
                                            C_type,
                                            ldc,
                                            stride_C,
                                            batch_count,
                                            compute_type,
                                            CUBLAS_GEMM_DEFAULT));
}

template void launch_linear<float>(cublasHandle_t &cublas_handle,
                                   const bool trans_A,
                                   const bool trans_B,
                                   const Tensor<float> &A,
                                   const Tensor<float> &B,
                                   Tensor<float> &C);

template void launch_linear<half>(cublasHandle_t &cublas_handle,
                                  const bool trans_A,
                                  const bool trans_B,
                                  const Tensor<half> &A,
                                  const Tensor<half> &B,
                                  Tensor<half> &C);
