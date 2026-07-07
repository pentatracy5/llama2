#include <gtest/gtest.h>
#include <kernel/linear.h>
#include <core/Tensor.cuh>
#include <common/macro.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <vector>
#include <random>
#include <numeric>
#include <type_traits>

// ------------------------------------------------------------------
// Type conversion helpers between float and the tested dtype.
// ------------------------------------------------------------------
template <typename T>
static T scalar_from_float(float x);

template <>
float scalar_from_float<float>(float x)
{
    return x;
}

template <>
half scalar_from_float<half>(float x)
{
    return __float2half(x);
}

template <typename T>
static float scalar_to_float(T x);

template <>
float scalar_to_float<float>(float x)
{
    return x;
}

template <>
float scalar_to_float<half>(half x)
{
    return __half2float(x);
}

template <typename T>
static std::vector<T> vector_from_float(const std::vector<float> &src)
{
    std::vector<T> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = scalar_from_float<T>(src[i]);
    return dst;
}

template <typename T>
static std::vector<float> vector_to_float(const std::vector<T> &src)
{
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = scalar_to_float(src[i]);
    return dst;
}

// ------------------------------------------------------------------
// Helper: compute the expected row-major GEMM result on the host.
//
// Caller provides the logical dimensions M, N, K and the transpose
// flags.  A_h / B_h are stored in row-major order with their actual
// shapes:
//   trans_A == false : A_h is (M, K)
//   trans_A == true  : A_h is (K, M)
//   trans_B == false : B_h is (K, N)
//   trans_B == true  : B_h is (N, K)
// The output C_h is always (M, N) in row-major order.
// ------------------------------------------------------------------
static void compute_expected_linear(const std::vector<float> &A_h,
                                    const std::vector<float> &B_h,
                                    bool trans_A,
                                    bool trans_B,
                                    unsigned int M,
                                    unsigned int N,
                                    unsigned int K,
                                    std::vector<float> &C_h)
{
    C_h.assign(M * N, 0.0f);

    const unsigned int a_row_stride = trans_A ? 1 : K;
    const unsigned int a_col_stride = trans_A ? M : 1;
    const unsigned int b_row_stride = trans_B ? 1 : N;
    const unsigned int b_col_stride = trans_B ? K : 1;

    for (unsigned int i = 0; i < M; ++i)
    {
        for (unsigned int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (unsigned int kk = 0; kk < K; ++kk)
            {
                const unsigned int a_idx = i * a_row_stride + kk * a_col_stride;
                const unsigned int b_idx = kk * b_row_stride + j * b_col_stride;
                sum += A_h[a_idx] * B_h[b_idx];
            }
            C_h[i * N + j] = sum;
        }
    }
}

// ------------------------------------------------------------------
// cuBLAS handle wrapper for RAII cleanup in tests.
// ------------------------------------------------------------------
class CuBLASHandle
{
public:
    CuBLASHandle()
    {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }

    ~CuBLASHandle()
    {
        cublasDestroy(handle_);
    }

    cublasHandle_t &get() { return handle_; }

    CuBLASHandle(const CuBLASHandle &) = delete;
    CuBLASHandle &operator=(const CuBLASHandle &) = delete;

private:
    cublasHandle_t handle_;
};

// ------------------------------------------------------------------
// Generic runner for plain 2-D GEMM.
// ------------------------------------------------------------------
template <typename T>
static void run_linear_2d_test(bool trans_A,
                               bool trans_B,
                               unsigned int M,
                               unsigned int N,
                               unsigned int K,
                               float tolerance)
{
    const unsigned int a_numel = trans_A ? K * M : M * K;
    const unsigned int b_numel = trans_B ? N * K : K * N;

    // Deterministic small values.  For FP16 we scale down to avoid
    // overflow during accumulation (FP16 max ~= 65504).
    const float scale = std::is_same<T, half>::value ? 1.0f / 16.0f : 1.0f;
    std::vector<float> A_h(a_numel);
    std::vector<float> B_h(b_numel);
    for (unsigned int i = 0; i < a_numel; ++i)
        A_h[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * scale;
    for (unsigned int i = 0; i < b_numel; ++i)
        B_h[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * scale;

    Tensor<T> A_d({trans_A ? K : M, trans_A ? M : K}, GPU);
    Tensor<T> B_d({trans_B ? N : K, trans_B ? K : N}, GPU);
    Tensor<T> C_d({M, N}, GPU);

    const std::vector<T> A_d_h = vector_from_float<T>(A_h);
    const std::vector<T> B_d_h = vector_from_float<T>(B_h);

    CUDA_CHECK(cudaMemcpy(A_d.data(), A_d_h.data(), A_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d.data(), B_d_h.data(), B_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));

    Tensor<void *> empty_array({});

    CuBLASHandle handle;
    launch_linear(handle.get(),
                  trans_A,
                  trans_B,
                  M,
                  N,
                  K,
                  A_d,
                  B_d,
                  C_d,
                  empty_array,
                  empty_array,
                  empty_array);

    std::vector<T> C_d_h(C_d.numel());
    CUDA_CHECK(cudaMemcpy(C_d_h.data(), C_d.data(), C_d_h.size() * sizeof(T), cudaMemcpyDeviceToHost));
    const std::vector<float> C_h = vector_to_float(C_d_h);

    std::vector<float> expected;
    compute_expected_linear(A_h, B_h, trans_A, trans_B, M, N, K, expected);

    ASSERT_EQ(C_h.size(), expected.size());
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        ASSERT_NEAR(C_h[idx], expected[idx], tolerance)
            << "idx=" << idx << ", M=" << M << ", N=" << N << ", K=" << K;
    }
}

// ------------------------------------------------------------------
// Generic runner for batched GEMM (3-D tensors: batch, M, N).
// ------------------------------------------------------------------
template <typename T>
static void run_linear_batched_test(bool trans_A,
                                    bool trans_B,
                                    unsigned int batch,
                                    unsigned int M,
                                    unsigned int N,
                                    unsigned int K,
                                    float tolerance)
{
    const unsigned int a_slice = trans_A ? K * M : M * K;
    const unsigned int b_slice = trans_B ? N * K : K * N;
    const unsigned int c_slice = M * N;

    const float scale = std::is_same<T, half>::value ? 1.0f / 16.0f : 1.0f;
    std::vector<float> A_h(batch * a_slice);
    std::vector<float> B_h(batch * b_slice);
    for (unsigned int i = 0; i < A_h.size(); ++i)
        A_h[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * scale;
    for (unsigned int i = 0; i < B_h.size(); ++i)
        B_h[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * scale;

    Tensor<T> A_d({batch, trans_A ? K : M, trans_A ? M : K}, GPU);
    Tensor<T> B_d({batch, trans_B ? N : K, trans_B ? K : N}, GPU);
    Tensor<T> C_d({batch, M, N}, GPU);

    const std::vector<T> A_d_h = vector_from_float<T>(A_h);
    const std::vector<T> B_d_h = vector_from_float<T>(B_h);

    CUDA_CHECK(cudaMemcpy(A_d.data(), A_d_h.data(), A_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d.data(), B_d_h.data(), B_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));

    Tensor<void *> A_array({batch}, GPU);
    Tensor<void *> B_array({batch}, GPU);
    Tensor<void *> C_array({batch}, GPU);

    std::vector<void *> A_ptrs(batch);
    std::vector<void *> B_ptrs(batch);
    std::vector<void *> C_ptrs(batch);
    for (unsigned int b = 0; b < batch; ++b)
    {
        A_ptrs[b] = static_cast<void *>(A_d.data() + b * a_slice);
        B_ptrs[b] = static_cast<void *>(B_d.data() + b * b_slice);
        C_ptrs[b] = static_cast<void *>(C_d.data() + b * c_slice);
    }

    CUDA_CHECK(cudaMemcpy(A_array.data(), A_ptrs.data(), batch * sizeof(void *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_array.data(), B_ptrs.data(), batch * sizeof(void *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_array.data(), C_ptrs.data(), batch * sizeof(void *), cudaMemcpyHostToDevice));

    CuBLASHandle handle;
    launch_linear(handle.get(),
                  trans_A,
                  trans_B,
                  M,
                  N,
                  K,
                  A_d,
                  B_d,
                  C_d,
                  A_array,
                  B_array,
                  C_array);

    std::vector<T> C_d_h(C_d.numel());
    CUDA_CHECK(cudaMemcpy(C_d_h.data(), C_d.data(), C_d_h.size() * sizeof(T), cudaMemcpyDeviceToHost));
    const std::vector<float> C_h = vector_to_float(C_d_h);

    for (unsigned int b = 0; b < batch; ++b)
    {
        std::vector<float> a_slice_h(A_h.begin() + b * a_slice, A_h.begin() + (b + 1) * a_slice);
        std::vector<float> b_slice_h(B_h.begin() + b * b_slice, B_h.begin() + (b + 1) * b_slice);
        std::vector<float> expected;
        compute_expected_linear(a_slice_h, b_slice_h, trans_A, trans_B, M, N, K, expected);

        for (unsigned int idx = 0; idx < c_slice; ++idx)
        {
            const unsigned int global_idx = b * c_slice + idx;
            ASSERT_NEAR(C_h[global_idx], expected[idx], tolerance)
                << "batch=" << b << ", idx=" << idx << ", M=" << M << ", N=" << N << ", K=" << K;
        }
    }
}

// ------------------------------------------------------------------
// Linear kernel tests: plain 2-D GEMM, FP32.
// ------------------------------------------------------------------
TEST(LinearTest, Fp32NoTransposeSmall)
{
    run_linear_2d_test<float>(false, false, 3, 4, 5, 1e-4f);
}

TEST(LinearTest, Fp32TransA)
{
    run_linear_2d_test<float>(true, false, 3, 4, 5, 1e-4f);
}

TEST(LinearTest, Fp32TransB)
{
    run_linear_2d_test<float>(false, true, 3, 4, 5, 1e-4f);
}

TEST(LinearTest, Fp32TransAB)
{
    run_linear_2d_test<float>(true, true, 3, 4, 5, 1e-4f);
}

TEST(LinearTest, Fp32SingleElement)
{
    run_linear_2d_test<float>(false, false, 1, 1, 1, 1e-4f);
}

TEST(LinearTest, Fp32LargeK)
{
    run_linear_2d_test<float>(false, false, 4, 6, 64, 1e-3f);
}

// ------------------------------------------------------------------
// Linear kernel tests: batched GEMM, FP32.
// ------------------------------------------------------------------
TEST(LinearTest, Fp32BatchedNoTranspose)
{
    run_linear_batched_test<float>(false, false, 3, 4, 5, 6, 1e-4f);
}

TEST(LinearTest, Fp32BatchedTransA)
{
    run_linear_batched_test<float>(true, false, 2, 4, 5, 6, 1e-4f);
}

TEST(LinearTest, Fp32BatchedTransB)
{
    run_linear_batched_test<float>(false, true, 2, 4, 5, 6, 1e-4f);
}

TEST(LinearTest, Fp32BatchedTransAB)
{
    run_linear_batched_test<float>(true, true, 2, 4, 5, 6, 1e-4f);
}

TEST(LinearTest, Fp32BatchedSingleBatch)
{
    run_linear_batched_test<float>(false, false, 1, 4, 5, 6, 1e-4f);
}

TEST(LinearTest, Fp32BatchedLargeBatch)
{
    run_linear_batched_test<float>(false, false, 32, 4, 5, 6, 1e-3f);
}

// ------------------------------------------------------------------
// Linear kernel tests: plain 2-D GEMM, FP16.
// ------------------------------------------------------------------
TEST(LinearHalfTest, Fp16NoTransposeSmall)
{
    run_linear_2d_test<half>(false, false, 3, 4, 5, 1e-2f);
}

TEST(LinearHalfTest, Fp16TransA)
{
    run_linear_2d_test<half>(true, false, 3, 4, 5, 1e-2f);
}

TEST(LinearHalfTest, Fp16TransB)
{
    run_linear_2d_test<half>(false, true, 3, 4, 5, 1e-2f);
}

TEST(LinearHalfTest, Fp16TransAB)
{
    run_linear_2d_test<half>(true, true, 3, 4, 5, 1e-2f);
}

TEST(LinearHalfTest, Fp16LargeK)
{
    run_linear_2d_test<half>(false, false, 4, 6, 64, 2e-2f);
}

// ------------------------------------------------------------------
// Linear kernel tests: batched GEMM, FP16.
// ------------------------------------------------------------------
TEST(LinearHalfTest, Fp16BatchedNoTranspose)
{
    run_linear_batched_test<half>(false, false, 3, 4, 5, 6, 1e-2f);
}

TEST(LinearHalfTest, Fp16BatchedTransA)
{
    run_linear_batched_test<half>(true, false, 2, 4, 5, 6, 1e-2f);
}

TEST(LinearHalfTest, Fp16BatchedTransB)
{
    run_linear_batched_test<half>(false, true, 2, 4, 5, 6, 1e-2f);
}

TEST(LinearHalfTest, Fp16BatchedTransAB)
{
    run_linear_batched_test<half>(true, true, 2, 4, 5, 6, 1e-2f);
}

TEST(LinearHalfTest, Fp16BatchedLargeBatch)
{
    run_linear_batched_test<half>(false, false, 32, 4, 5, 6, 2e-2f);
}
