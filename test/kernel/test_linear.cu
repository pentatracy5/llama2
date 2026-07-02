#include <gtest/gtest.h>
#include <kernel/linear.h>
#include <core/Tensor.cuh>
#include <cublas_v2.h>

#include <vector>
#include <random>
#include <cuda_fp16.h>

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
// A/B logical shapes already account for the requested transposes:
//   - trans_A == false: A is [..., M, K]
//   - trans_A == true : A is [..., K, M], used as A^T
//   - trans_B == false: B is [..., K, N]
//   - trans_B == true : B is [..., N, K], used as B^T
// C is always [..., M, N] in row-major.
// ------------------------------------------------------------------
static void compute_expected_linear(const std::vector<float> &A_h,
                                    const std::vector<float> &B_h,
                                    const std::vector<unsigned int> &A_shape,
                                    const std::vector<unsigned int> &B_shape,
                                    const std::vector<unsigned int> &C_shape,
                                    bool trans_A,
                                    bool trans_B,
                                    std::vector<float> &C_h)
{
    const unsigned int dim = static_cast<unsigned int>(A_shape.size());
    const unsigned int M = C_shape[dim - 2];
    const unsigned int N = C_shape[dim - 1];
    const unsigned int K = trans_A ? A_shape[dim - 2] : A_shape[dim - 1];

    unsigned int batch = 1;
    for (unsigned int i = 0; i + 2 < dim; ++i)
        batch *= C_shape[i];

    C_h.resize(batch * M * N);

    const unsigned int A_rows = A_shape[dim - 2];
    const unsigned int A_cols = A_shape[dim - 1];
    const unsigned int B_rows = B_shape[dim - 2];
    const unsigned int B_cols = B_shape[dim - 1];

    for (unsigned int b = 0; b < batch; ++b)
    {
        const size_t a_batch_offset = static_cast<size_t>(b) * A_rows * A_cols;
        const size_t b_batch_offset = static_cast<size_t>(b) * B_rows * B_cols;
        const size_t c_batch_offset = static_cast<size_t>(b) * M * N;

        for (unsigned int i = 0; i < M; ++i)
        {
            for (unsigned int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (unsigned int k = 0; k < K; ++k)
                {
                    const float a_val = trans_A ? A_h[a_batch_offset + k * A_cols + i]
                                                : A_h[a_batch_offset + i * A_cols + k];
                    const float b_val = trans_B ? B_h[b_batch_offset + j * B_cols + k]
                                                : B_h[b_batch_offset + k * B_cols + j];
                    sum += a_val * b_val;
                }
                C_h[c_batch_offset + i * N + j] = sum;
            }
        }
    }
}

// ------------------------------------------------------------------
// Generic runner: copy data to GPU, launch linear, copy back, verify.
// ------------------------------------------------------------------
template <typename T>
static void run_linear_test(const std::vector<float> &A_h,
                            const std::vector<float> &B_h,
                            const std::vector<unsigned int> &A_shape,
                            const std::vector<unsigned int> &B_shape,
                            const std::vector<unsigned int> &C_shape,
                            bool trans_A,
                            bool trans_B,
                            float tolerance)
{
    std::vector<float> expected_h;
    compute_expected_linear(A_h, B_h, A_shape, B_shape, C_shape,
                            trans_A, trans_B, expected_h);

    Tensor<T> A_d(A_shape, GPU);
    Tensor<T> B_d(B_shape, GPU);
    Tensor<T> C_d(C_shape, GPU);

    const std::vector<T> A_d_h = vector_from_float<T>(A_h);
    const std::vector<T> B_d_h = vector_from_float<T>(B_h);

    CUDA_CHECK(cudaMemcpy(A_d.data(), A_d_h.data(),
                          A_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d.data(), B_d_h.data(),
                          B_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));

    const unsigned int dim = static_cast<unsigned int>(A_shape.size());
    const unsigned int M = C_shape[dim - 2];
    const unsigned int N = C_shape[dim - 1];
    const unsigned int K = trans_A ? A_shape[dim - 2] : A_shape[dim - 1];

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    launch_linear<T>(handle, trans_A, trans_B, M, N, K, A_d, B_d, C_d);

    CUBLAS_CHECK(cublasDestroy(handle));

    std::vector<T> C_d_h(C_d.numel());
    CUDA_CHECK(cudaMemcpy(C_d_h.data(), C_d.data(),
                          C_d_h.size() * sizeof(T), cudaMemcpyDeviceToHost));
    const std::vector<float> C_h = vector_to_float(C_d_h);

    for (size_t idx = 0; idx < expected_h.size(); ++idx)
    {
        ASSERT_NEAR(C_h[idx], expected_h[idx], tolerance)
            << "idx=" << idx;
    }
}

// ------------------------------------------------------------------
// Helper: run a submatrix GEMM where the tensor shapes are larger than
// the actual M, N, K used in the cublas call.
// ------------------------------------------------------------------
template <typename T>
static void run_linear_submatrix_test(const std::vector<float> &A_h,
                                      const std::vector<float> &B_h,
                                      const std::vector<unsigned int> &A_shape,
                                      const std::vector<unsigned int> &B_shape,
                                      const std::vector<unsigned int> &C_shape,
                                      const unsigned int M,
                                      const unsigned int N,
                                      const unsigned int K,
                                      const bool trans_A,
                                      const bool trans_B,
                                      const float tolerance)
{
    const unsigned int dim = static_cast<unsigned int>(A_shape.size());
    const unsigned int A_rows = A_shape[dim - 2];
    const unsigned int A_cols = A_shape[dim - 1];
    const unsigned int B_rows = B_shape[dim - 2];
    const unsigned int B_cols = B_shape[dim - 1];
    const unsigned int C_rows = C_shape[dim - 2];
    const unsigned int C_cols = C_shape[dim - 1];

    unsigned int batch = 1;
    for (unsigned int i = 0; i + 2 < dim; ++i)
        batch *= C_shape[i];

    std::vector<float> expected_h(batch * M * N);

    for (unsigned int b = 0; b < batch; ++b)
    {
        const size_t a_batch_offset = static_cast<size_t>(b) * A_rows * A_cols;
        const size_t b_batch_offset = static_cast<size_t>(b) * B_rows * B_cols;
        const size_t expected_batch_offset = static_cast<size_t>(b) * M * N;

        for (unsigned int i = 0; i < M; ++i)
        {
            for (unsigned int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (unsigned int k = 0; k < K; ++k)
                {
                    const float a_val = trans_A ? A_h[a_batch_offset + k * A_cols + i]
                                                : A_h[a_batch_offset + i * A_cols + k];
                    const float b_val = trans_B ? B_h[b_batch_offset + j * B_cols + k]
                                                : B_h[b_batch_offset + k * B_cols + j];
                    sum += a_val * b_val;
                }
                expected_h[expected_batch_offset + i * N + j] = sum;
            }
        }
    }

    Tensor<T> A_d(A_shape, GPU);
    Tensor<T> B_d(B_shape, GPU);
    Tensor<T> C_d(C_shape, GPU);

    const std::vector<T> A_d_h = vector_from_float<T>(A_h);
    const std::vector<T> B_d_h = vector_from_float<T>(B_h);

    CUDA_CHECK(cudaMemcpy(A_d.data(), A_d_h.data(),
                          A_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d.data(), B_d_h.data(),
                          B_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    launch_linear<T>(handle, trans_A, trans_B, M, N, K, A_d, B_d, C_d);

    CUBLAS_CHECK(cublasDestroy(handle));

    std::vector<T> C_d_h(C_d.numel());
    CUDA_CHECK(cudaMemcpy(C_d_h.data(), C_d.data(),
                          C_d_h.size() * sizeof(T), cudaMemcpyDeviceToHost));
    const std::vector<float> C_h = vector_to_float(C_d_h);

    for (unsigned int b = 0; b < batch; ++b)
    {
        const size_t expected_batch_offset = static_cast<size_t>(b) * M * N;
        const size_t c_batch_offset = static_cast<size_t>(b) * C_rows * C_cols;
        for (unsigned int i = 0; i < M; ++i)
        {
            for (unsigned int j = 0; j < N; ++j)
            {
                const unsigned int expected_idx = expected_batch_offset + i * N + j;
                const unsigned int c_idx = c_batch_offset + i * C_cols + j;
                ASSERT_NEAR(C_h[c_idx], expected_h[expected_idx], tolerance)
                    << "batch=" << b << " i=" << i << " j=" << j;
            }
        }
    }
}

// ------------------------------------------------------------------
// Linear kernel tests (FP32)
// ------------------------------------------------------------------
TEST(LinearTest, BasicNoTranspose2D)
{
    // A[2,3] * B[3,4] = C[2,4]
    const std::vector<float> A_h = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};
    const std::vector<float> B_h = {
        7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f, 18.0f};

    run_linear_test<float>(A_h, B_h,
                           {2, 3}, {3, 4}, {2, 4},
                           false, false, 1e-4f);
}

TEST(LinearTest, TransposeA2D)
{
    // A is stored as [3,2]; trans_A=true => logical A^T is [2,3]
    // A^T * B[3,4] = C[2,4]
    const std::vector<float> A_h = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f};
    const std::vector<float> B_h = {
        7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f, 18.0f};

    run_linear_test<float>(A_h, B_h,
                           {3, 2}, {3, 4}, {2, 4},
                           true, false, 1e-4f);
}

TEST(LinearTest, TransposeB2D)
{
    // A[2,3] * B^T where B is stored as [4,3] => logical B^T is [3,4]
    // A[2,3] * B^T[3,4] = C[2,4]
    const std::vector<float> A_h = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};
    const std::vector<float> B_h = {
        7.0f, 11.0f, 15.0f,
        8.0f, 12.0f, 16.0f,
        9.0f, 13.0f, 17.0f,
        10.0f, 14.0f, 18.0f};

    run_linear_test<float>(A_h, B_h,
                           {2, 3}, {4, 3}, {2, 4},
                           false, true, 1e-4f);
}

TEST(LinearTest, TransposeBoth2D)
{
    // A stored as [3,2], B stored as [4,3]
    // A^T[2,3] * B^T[3,4] = C[2,4]
    const std::vector<float> A_h = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f};
    const std::vector<float> B_h = {
        7.0f, 11.0f, 15.0f,
        8.0f, 12.0f, 16.0f,
        9.0f, 13.0f, 17.0f,
        10.0f, 14.0f, 18.0f};

    run_linear_test<float>(A_h, B_h,
                           {3, 2}, {4, 3}, {2, 4},
                           true, true, 1e-4f);
}

TEST(LinearTest, BatchedNoTranspose3D)
{
    // 2 batches of A[2,3] * B[3,4] = C[2,4]
    const std::vector<float> A_h = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,

        2.0f, 0.0f, -1.0f,
        1.0f, 3.0f, -2.0f};
    const std::vector<float> B_h = {
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,

        -1.0f, 2.0f, 0.0f, 1.0f,
        3.0f, -1.0f, 2.0f, 0.0f,
        0.0f, 1.0f, 1.0f, -1.0f};

    run_linear_test<float>(A_h, B_h,
                           {2, 2, 3}, {2, 3, 4}, {2, 2, 4},
                           false, false, 1e-4f);
}

TEST(LinearTest, LargeRandom2D)
{
    const unsigned int M = 64;
    const unsigned int K = 48;
    const unsigned int N = 32;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> A_h(M * K);
    std::vector<float> B_h(K * N);
    for (float &v : A_h)
        v = dist(gen);
    for (float &v : B_h)
        v = dist(gen);

    run_linear_test<float>(A_h, B_h,
                           {M, K}, {K, N}, {M, N},
                           false, false, 1e-3f);
}

TEST(LinearTest, SubMatrixTransposeB2D)
{
    // A is [4, 5], B is stored as [6, 5] (trans_B=true => B^T is [5, 6])
    // We only compute a [2, 4] x [4, 3] -> [2, 3] submatrix.
    const std::vector<float> A_h = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f, 20.0f};
    const std::vector<float> B_h = {
        1.0f, 6.0f, 11.0f, 16.0f, 21.0f,
        2.0f, 7.0f, 12.0f, 17.0f, 22.0f,
        3.0f, 8.0f, 13.0f, 18.0f, 23.0f,
        4.0f, 9.0f, 14.0f, 19.0f, 24.0f,
        5.0f, 10.0f, 15.0f, 20.0f, 25.0f,
        26.0f, 27.0f, 28.0f, 29.0f, 30.0f};

    run_linear_submatrix_test<float>(A_h, B_h,
                                     {4, 5}, {6, 5}, {4, 6},
                                     2, 3, 4,
                                     false, true, 1e-4f);
}

// ------------------------------------------------------------------
// Linear kernel tests (FP16)
// ------------------------------------------------------------------
TEST(LinearHalfTest, BasicNoTranspose2D)
{
    const std::vector<float> A_h = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};
    const std::vector<float> B_h = {
        7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f, 18.0f};

    run_linear_test<half>(A_h, B_h,
                          {2, 3}, {3, 4}, {2, 4},
                          false, false, 1e-2f);
}

TEST(LinearHalfTest, TransposeA2D)
{
    const std::vector<float> A_h = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f};
    const std::vector<float> B_h = {
        7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f, 18.0f};

    run_linear_test<half>(A_h, B_h,
                          {3, 2}, {3, 4}, {2, 4},
                          true, false, 1e-2f);
}

TEST(LinearHalfTest, TransposeB2D)
{
    const std::vector<float> A_h = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};
    const std::vector<float> B_h = {
        7.0f, 11.0f, 15.0f,
        8.0f, 12.0f, 16.0f,
        9.0f, 13.0f, 17.0f,
        10.0f, 14.0f, 18.0f};

    run_linear_test<half>(A_h, B_h,
                          {2, 3}, {4, 3}, {2, 4},
                          false, true, 1e-2f);
}

TEST(LinearHalfTest, BatchedNoTranspose3D)
{
    const std::vector<float> A_h = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,

        2.0f, 0.0f, -1.0f,
        1.0f, 3.0f, -2.0f};
    const std::vector<float> B_h = {
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,

        -1.0f, 2.0f, 0.0f, 1.0f,
        3.0f, -1.0f, 2.0f, 0.0f,
        0.0f, 1.0f, 1.0f, -1.0f};

    run_linear_test<half>(A_h, B_h,
                          {2, 2, 3}, {2, 3, 4}, {2, 2, 4},
                          false, false, 1e-2f);
}

TEST(LinearHalfTest, LargeRandom2D)
{
    const unsigned int M = 64;
    const unsigned int K = 48;
    const unsigned int N = 32;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_h(M * K);
    std::vector<float> B_h(K * N);
    for (float &v : A_h)
        v = dist(gen);
    for (float &v : B_h)
        v = dist(gen);

    run_linear_test<half>(A_h, B_h,
                          {M, K}, {K, N}, {M, N},
                          false, false, 2e-2f);
}

TEST(LinearHalfTest, SubMatrixTransposeB2D)
{
    const std::vector<float> A_h = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f, 20.0f};
    const std::vector<float> B_h = {
        1.0f, 6.0f, 11.0f, 16.0f, 21.0f,
        2.0f, 7.0f, 12.0f, 17.0f, 22.0f,
        3.0f, 8.0f, 13.0f, 18.0f, 23.0f,
        4.0f, 9.0f, 14.0f, 19.0f, 24.0f,
        5.0f, 10.0f, 15.0f, 20.0f, 25.0f,
        26.0f, 27.0f, 28.0f, 29.0f, 30.0f};

    run_linear_submatrix_test<half>(A_h, B_h,
                                    {4, 5}, {6, 5}, {4, 6},
                                    2, 3, 4,
                                    false, true, 2e-2f);
}
