#include <gtest/gtest.h>
#include <kernel/fused_add_residual_rmsnorm.h>
#include <core/Tensor.cuh>
#include <common/config.h>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_fp16.h>

// ------------------------------------------------------------------
// Helpers: compute the expected fused add-residual-RMSNorm result.
//
// If add_residual is true:
//     x_after[t, d]   = x[t, d] + residual[t, d]
// else:
//     x_after[t, d]   = x[t, d]
//
// output[t, d]       = x_after[t, d] * weight[d] * inv_rms[t]
// inv_rms[t]         = 1 / sqrt(mean(x_after[t]^2) + eps)
// ------------------------------------------------------------------
static void compute_expected_fused_add_residual_rmsnorm(const std::vector<float> &x_h,
                                                        const std::vector<float> &residual_h,
                                                        const std::vector<float> &weight_h,
                                                        unsigned int num_tokens,
                                                        unsigned int embed_dim,
                                                        bool add_residual,
                                                        std::vector<float> &expected_x_after,
                                                        std::vector<float> &expected_residual_out)
{
    const unsigned int numel = num_tokens * embed_dim;
    expected_x_after.resize(numel);
    expected_residual_out.resize(numel);

    for (unsigned int t = 0; t < num_tokens; ++t)
    {
        float sq_sum = 0.0f;
        for (unsigned int d = 0; d < embed_dim; ++d)
        {
            const unsigned int idx = t * embed_dim + d;
            const float x_val = add_residual ? x_h[idx] + residual_h[idx] : x_h[idx];
            expected_x_after[idx] = x_val;
            sq_sum += x_val * x_val;
        }
        const float inv_rms = 1.0f / std::sqrt(sq_sum / static_cast<float>(embed_dim) + RMSNORM_EPS);

        for (unsigned int d = 0; d < embed_dim; ++d)
        {
            const unsigned int idx = t * embed_dim + d;
            expected_residual_out[idx] = expected_x_after[idx] * weight_h[d] * inv_rms;
        }
    }
}

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
// Generic runner: copy data to GPU, launch kernel, copy back, verify.
// ------------------------------------------------------------------
template <typename T, bool add_residual>
static void run_fused_add_residual_rmsnorm_test(const std::vector<float> &x_h,
                                                const std::vector<float> &residual_h,
                                                const std::vector<float> &weight_h,
                                                unsigned int num_tokens,
                                                unsigned int embed_dim,
                                                float tolerance)
{
    const unsigned int numel = num_tokens * embed_dim;
    ASSERT_EQ(x_h.size(), numel);
    ASSERT_EQ(weight_h.size(), embed_dim);
    if constexpr (add_residual)
    {
        ASSERT_EQ(residual_h.size(), numel);
    }

    const std::vector<T> x_d_h = vector_from_float<T>(x_h);
    const std::vector<T> weight_d_h = vector_from_float<T>(weight_h);
    const std::vector<T> residual_d_h = vector_from_float<T>(residual_h);

    Tensor<T> x_d({num_tokens, embed_dim}, GPU);
    Tensor<T> residual_d({num_tokens, embed_dim}, GPU);
    Tensor<T> weight_d({embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(x_d.data(), x_d_h.data(),
                          x_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(residual_d.data(), residual_d_h.data(),
                          residual_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight_d.data(), weight_d_h.data(),
                          weight_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));

    launch_fused_add_residual_rmsnorm<T, add_residual>(weight_d, x_d, residual_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<T> x_after_d_h(x_d.numel());
    std::vector<T> residual_out_d_h(residual_d.numel());
    CUDA_CHECK(cudaMemcpy(x_after_d_h.data(), x_d.data(),
                          x_after_d_h.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(residual_out_d_h.data(), residual_d.data(),
                          residual_out_d_h.size() * sizeof(T), cudaMemcpyDeviceToHost));

    const std::vector<float> x_after_h = vector_to_float(x_after_d_h);
    const std::vector<float> residual_out_h = vector_to_float(residual_out_d_h);

    std::vector<float> expected_x_after;
    std::vector<float> expected_residual_out;
    compute_expected_fused_add_residual_rmsnorm(x_h, residual_h, weight_h, num_tokens, embed_dim,
                                                add_residual, expected_x_after, expected_residual_out);

    for (unsigned int idx = 0; idx < numel; ++idx)
    {
        ASSERT_NEAR(x_after_h[idx], expected_x_after[idx], tolerance)
            << "x mismatch at idx=" << idx;
        ASSERT_NEAR(residual_out_h[idx], expected_residual_out[idx], tolerance)
            << "residual output mismatch at idx=" << idx;
    }
}

// ------------------------------------------------------------------
// RMSNorm only (add_residual = false) tests
// These mirror the original rmsnorm tests; residual is used as output.
// ------------------------------------------------------------------
TEST(RMSNormTest, SingleTokenUnitWeights)
{
    const unsigned int num_tokens = 1;
    const unsigned int embed_dim = 8;

    const std::vector<float> x_h = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> residual_h(num_tokens * embed_dim, 0.0f);
    const std::vector<float> weight_h(embed_dim, 1.0f);

    run_fused_add_residual_rmsnorm_test<float, false>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(RMSNormTest, MultipleTokensWithWeights)
{
    const unsigned int num_tokens = 3;
    const unsigned int embed_dim = 16;

    const std::vector<float> x_h = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,

        -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f,
        -0.9f, -1.0f, -1.1f, -1.2f, -1.3f, -1.4f, -1.5f, -1.6f,

        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    const std::vector<float> residual_h(num_tokens * embed_dim, 0.0f);

    std::vector<float> weight_h;
    weight_h.reserve(embed_dim);
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h.push_back(static_cast<float>(d + 1));

    run_fused_add_residual_rmsnorm_test<float, false>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(RMSNormTest, ConstantInputAndWeights)
{
    const unsigned int num_tokens = 5;
    const unsigned int embed_dim = 32;

    const std::vector<float> x_h(num_tokens * embed_dim, 2.0f);
    const std::vector<float> residual_h(num_tokens * embed_dim, 0.0f);
    const std::vector<float> weight_h(embed_dim, 0.5f);

    run_fused_add_residual_rmsnorm_test<float, false>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(RMSNormTest, SmallDimMultipleOfVecLen)
{
    const unsigned int num_tokens = 4;
    const unsigned int embed_dim = 4;

    const std::vector<float> x_h = {
        1.0f, 0.0f, -1.0f, 2.0f,
        3.0f, 1.0f, -2.0f, 0.5f,
        0.0f, 0.0f, 0.0f, 0.0f,
        -0.5f, -1.5f, 2.5f, -3.5f};
    const std::vector<float> residual_h(num_tokens * embed_dim, 0.0f);
    const std::vector<float> weight_h = {0.5f, 1.0f, 1.5f, 2.0f};

    run_fused_add_residual_rmsnorm_test<float, false>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(RMSNormTest, LargeBatchRandomValues)
{
    const unsigned int num_tokens = 64;
    const unsigned int embed_dim = 64;
    const unsigned int numel = num_tokens * embed_dim;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> weight_dist(0.1f, 2.0f);

    std::vector<float> x_h(numel);
    std::vector<float> residual_h(numel);
    std::vector<float> weight_h(embed_dim);
    for (unsigned int i = 0; i < numel; ++i)
    {
        x_h[i] = x_dist(gen);
        residual_h[i] = 0.0f;
    }
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h[d] = weight_dist(gen);

    run_fused_add_residual_rmsnorm_test<float, false>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-3f);
}

// ------------------------------------------------------------------
// Fused add-residual + RMSNorm (add_residual = true) tests
// ------------------------------------------------------------------
TEST(FusedAddResidualRMSNormTest, SingleToken)
{
    const unsigned int num_tokens = 1;
    const unsigned int embed_dim = 8;

    const std::vector<float> x_h = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> residual_h = {
        0.5f, -0.5f, 1.0f, -1.0f, 0.25f, -0.25f, 0.125f, -0.125f};
    const std::vector<float> weight_h(embed_dim, 1.0f);

    run_fused_add_residual_rmsnorm_test<float, true>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(FusedAddResidualRMSNormTest, MultipleTokensWithWeights)
{
    const unsigned int num_tokens = 3;
    const unsigned int embed_dim = 16;

    const std::vector<float> x_h = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,

        -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f,
        -0.9f, -1.0f, -1.1f, -1.2f, -1.3f, -1.4f, -1.5f, -1.6f,

        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

    const std::vector<float> residual_h = {
        -0.1f, 0.1f, -0.2f, 0.2f, -0.3f, 0.3f, -0.4f, 0.4f,
        -0.5f, 0.5f, -0.6f, 0.6f, -0.7f, 0.7f, -0.8f, 0.8f,

        0.1f, -0.1f, 0.2f, -0.2f, 0.3f, -0.3f, 0.4f, -0.4f,
        0.5f, -0.5f, 0.6f, -0.6f, 0.7f, -0.7f, 0.8f, -0.8f,

        1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};

    std::vector<float> weight_h;
    weight_h.reserve(embed_dim);
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h.push_back(static_cast<float>(d + 1));

    run_fused_add_residual_rmsnorm_test<float, true>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(FusedAddResidualRMSNormTest, ResidualCancelsX)
{
    // Special case: residual = -x, so x + residual = 0. Output should be 0.
    const unsigned int num_tokens = 2;
    const unsigned int embed_dim = 8;

    const std::vector<float> x_h(num_tokens * embed_dim, 1.5f);
    std::vector<float> residual_h(num_tokens * embed_dim);
    for (float &v : residual_h)
        v = -1.5f;
    const std::vector<float> weight_h(embed_dim, 2.0f);

    run_fused_add_residual_rmsnorm_test<float, true>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(FusedAddResidualRMSNormTest, LargeBatchRandomValues)
{
    const unsigned int num_tokens = 64;
    const unsigned int embed_dim = 64;
    const unsigned int numel = num_tokens * embed_dim;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> residual_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(0.1f, 2.0f);

    std::vector<float> x_h(numel);
    std::vector<float> residual_h(numel);
    std::vector<float> weight_h(embed_dim);
    for (unsigned int i = 0; i < numel; ++i)
    {
        x_h[i] = x_dist(gen);
        residual_h[i] = residual_dist(gen);
    }
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h[d] = weight_dist(gen);

    run_fused_add_residual_rmsnorm_test<float, true>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-3f);
}

// ------------------------------------------------------------------
// FP16 RMSNorm only (add_residual = false) tests
// ------------------------------------------------------------------
TEST(RMSNormHalfTest, SingleTokenUnitWeights)
{
    const unsigned int num_tokens = 1;
    const unsigned int embed_dim = 8;

    const std::vector<float> x_h = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> residual_h(num_tokens * embed_dim, 0.0f);
    const std::vector<float> weight_h(embed_dim, 1.0f);

    run_fused_add_residual_rmsnorm_test<half, false>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-3f);
}

TEST(RMSNormHalfTest, MultipleTokensWithWeights)
{
    const unsigned int num_tokens = 3;
    const unsigned int embed_dim = 16;

    const std::vector<float> x_h = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,

        -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f,
        -0.9f, -1.0f, -1.1f, -1.2f, -1.3f, -1.4f, -1.5f, -1.6f,

        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    const std::vector<float> residual_h(num_tokens * embed_dim, 0.0f);

    std::vector<float> weight_h;
    weight_h.reserve(embed_dim);
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h.push_back(static_cast<float>(d + 1));

    run_fused_add_residual_rmsnorm_test<half, false>(x_h, residual_h, weight_h, num_tokens, embed_dim, 2e-2f);
}

TEST(RMSNormHalfTest, ConstantInputAndWeights)
{
    const unsigned int num_tokens = 5;
    const unsigned int embed_dim = 32;

    const std::vector<float> x_h(num_tokens * embed_dim, 2.0f);
    const std::vector<float> residual_h(num_tokens * embed_dim, 0.0f);
    const std::vector<float> weight_h(embed_dim, 0.5f);

    run_fused_add_residual_rmsnorm_test<half, false>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-3f);
}

TEST(RMSNormHalfTest, LargeBatchRandomValues)
{
    const unsigned int num_tokens = 64;
    const unsigned int embed_dim = 64;
    const unsigned int numel = num_tokens * embed_dim;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> weight_dist(0.1f, 2.0f);

    std::vector<float> x_h(numel);
    std::vector<float> residual_h(numel);
    std::vector<float> weight_h(embed_dim);
    for (unsigned int i = 0; i < numel; ++i)
    {
        x_h[i] = x_dist(gen);
        residual_h[i] = 0.0f;
    }
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h[d] = weight_dist(gen);

    run_fused_add_residual_rmsnorm_test<half, false>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-2f);
}

// ------------------------------------------------------------------
// FP16 fused add-residual + RMSNorm (add_residual = true) tests
// ------------------------------------------------------------------
TEST(FusedAddResidualRMSNormHalfTest, SingleToken)
{
    const unsigned int num_tokens = 1;
    const unsigned int embed_dim = 8;

    const std::vector<float> x_h = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> residual_h = {
        0.5f, -0.5f, 1.0f, -1.0f, 0.25f, -0.25f, 0.125f, -0.125f};
    const std::vector<float> weight_h(embed_dim, 1.0f);

    run_fused_add_residual_rmsnorm_test<half, true>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-3f);
}

TEST(FusedAddResidualRMSNormHalfTest, MultipleTokensWithWeights)
{
    const unsigned int num_tokens = 3;
    const unsigned int embed_dim = 16;

    const std::vector<float> x_h = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,

        -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f,
        -0.9f, -1.0f, -1.1f, -1.2f, -1.3f, -1.4f, -1.5f, -1.6f,

        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

    const std::vector<float> residual_h = {
        -0.1f, 0.1f, -0.2f, 0.2f, -0.3f, 0.3f, -0.4f, 0.4f,
        -0.5f, 0.5f, -0.6f, 0.6f, -0.7f, 0.7f, -0.8f, 0.8f,

        0.1f, -0.1f, 0.2f, -0.2f, 0.3f, -0.3f, 0.4f, -0.4f,
        0.5f, -0.5f, 0.6f, -0.6f, 0.7f, -0.7f, 0.8f, -0.8f,

        1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};

    std::vector<float> weight_h;
    weight_h.reserve(embed_dim);
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h.push_back(static_cast<float>(d + 1));

    run_fused_add_residual_rmsnorm_test<half, true>(x_h, residual_h, weight_h, num_tokens, embed_dim, 5e-2f);
}

TEST(FusedAddResidualRMSNormHalfTest, LargeBatchRandomValues)
{
    const unsigned int num_tokens = 64;
    const unsigned int embed_dim = 64;
    const unsigned int numel = num_tokens * embed_dim;

    std::mt19937 gen(456);
    std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> residual_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(0.1f, 2.0f);

    std::vector<float> x_h(numel);
    std::vector<float> residual_h(numel);
    std::vector<float> weight_h(embed_dim);
    for (unsigned int i = 0; i < numel; ++i)
    {
        x_h[i] = x_dist(gen);
        residual_h[i] = residual_dist(gen);
    }
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h[d] = weight_dist(gen);

    run_fused_add_residual_rmsnorm_test<half, true>(x_h, residual_h, weight_h, num_tokens, embed_dim, 1e-2f);
}
