#include <gtest/gtest.h>
#include <kernel/rmsnorm.h>
#include <core/Tensor.cuh>
#include <common/config.h>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_fp16.h>

// ------------------------------------------------------------------
// Helper: compute the expected RMSNorm result on the host.
// output[t, d] = input[t, d] * weight[d] * inv_rms[t]
// inv_rms[t]   = 1 / sqrt(mean(input[t]^2) + eps)
// ------------------------------------------------------------------
static void compute_expected_rmsnorm(const std::vector<float> &input,
                                     const std::vector<float> &weight,
                                     unsigned int num_tokens,
                                     unsigned int embed_dim,
                                     std::vector<float> &output)
{
    output.resize(input.size());

    for (unsigned int t = 0; t < num_tokens; ++t)
    {
        float sq_sum = 0.0f;
        for (unsigned int d = 0; d < embed_dim; ++d)
        {
            float x = input[t * embed_dim + d];
            sq_sum += x * x;
        }
        float inv_rms = 1.0f / std::sqrt(sq_sum / static_cast<float>(embed_dim) + EPS);

        for (unsigned int d = 0; d < embed_dim; ++d)
        {
            output[t * embed_dim + d] = input[t * embed_dim + d] * weight[d] * inv_rms;
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
template <typename T>
static void run_rmsnorm_test(const std::vector<float> &input_h,
                             const std::vector<float> &weight_h,
                             unsigned int num_tokens,
                             unsigned int embed_dim,
                             float tolerance)
{
    ASSERT_EQ(input_h.size(), num_tokens * embed_dim);
    ASSERT_EQ(weight_h.size(), embed_dim);

    const std::vector<T> input_d_h = vector_from_float<T>(input_h);
    const std::vector<T> weight_d_h = vector_from_float<T>(weight_h);

    Tensor<T> input_d({num_tokens, embed_dim}, GPU);
    Tensor<T> weight_d({embed_dim}, GPU);
    Tensor<T> output_d({num_tokens, embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(input_d.data(), input_d_h.data(),
                          input_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight_d.data(), weight_d_h.data(),
                          weight_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));

    launch_rmsnorm(input_d, weight_d, output_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<T> output_d_h(output_d.numel());
    CUDA_CHECK(cudaMemcpy(output_d_h.data(), output_d.data(),
                          output_d_h.size() * sizeof(T), cudaMemcpyDeviceToHost));

    const std::vector<float> output_h = vector_to_float(output_d_h);

    std::vector<float> expected;
    compute_expected_rmsnorm(input_h, weight_h, num_tokens, embed_dim, expected);

    for (unsigned int idx = 0; idx < expected.size(); ++idx)
    {
        ASSERT_NEAR(output_h[idx], expected[idx], tolerance)
            << "idx=" << idx;
    }
}

// ------------------------------------------------------------------
// RMSNorm kernel tests (FP32)
// ------------------------------------------------------------------
TEST(RMSNormTest, SingleTokenUnitWeights)
{
    const unsigned int num_tokens = 1;
    const unsigned int embed_dim = 8;

    const std::vector<float> input_h = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> weight_h(embed_dim, 1.0f);

    run_rmsnorm_test<float>(input_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(RMSNormTest, MultipleTokensWithWeights)
{
    const unsigned int num_tokens = 3;
    const unsigned int embed_dim = 16;

    const std::vector<float> input_h = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,

        -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f,
        -0.9f, -1.0f, -1.1f, -1.2f, -1.3f, -1.4f, -1.5f, -1.6f,

        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

    std::vector<float> weight_h;
    weight_h.reserve(embed_dim);
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h.push_back(static_cast<float>(d + 1));

    run_rmsnorm_test<float>(input_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(RMSNormTest, ConstantInputAndWeights)
{
    const unsigned int num_tokens = 5;
    const unsigned int embed_dim = 32;

    const std::vector<float> input_h(num_tokens * embed_dim, 2.0f);
    const std::vector<float> weight_h(embed_dim, 0.5f);

    run_rmsnorm_test<float>(input_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(RMSNormTest, SmallDimMultipleOfVecLen)
{
    const unsigned int num_tokens = 4;
    const unsigned int embed_dim = 4;

    const std::vector<float> input_h = {
        1.0f, 0.0f, -1.0f, 2.0f,
        3.0f, 1.0f, -2.0f, 0.5f,
        0.0f, 0.0f, 0.0f, 0.0f,
        -0.5f, -1.5f, 2.5f, -3.5f};
    const std::vector<float> weight_h = {0.5f, 1.0f, 1.5f, 2.0f};

    run_rmsnorm_test<float>(input_h, weight_h, num_tokens, embed_dim, 1e-4f);
}

TEST(RMSNormTest, LargeBatchRandomValues)
{
    const unsigned int num_tokens = 64;
    const unsigned int embed_dim = 64;
    const unsigned int numel = num_tokens * embed_dim;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> input_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> weight_dist(0.1f, 2.0f);

    std::vector<float> input_h(numel);
    std::vector<float> weight_h(embed_dim);
    for (unsigned int i = 0; i < numel; ++i)
        input_h[i] = input_dist(gen);
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h[d] = weight_dist(gen);

    run_rmsnorm_test<float>(input_h, weight_h, num_tokens, embed_dim, 1e-3f);
}

// ------------------------------------------------------------------
// RMSNorm kernel tests (FP16)
// ------------------------------------------------------------------
TEST(RMSNormHalfTest, SingleTokenUnitWeights)
{
    const unsigned int num_tokens = 1;
    const unsigned int embed_dim = 8;

    const std::vector<float> input_h = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> weight_h(embed_dim, 1.0f);

    run_rmsnorm_test<half>(input_h, weight_h, num_tokens, embed_dim, 1e-3f);
}

TEST(RMSNormHalfTest, MultipleTokensWithWeights)
{
    const unsigned int num_tokens = 3;
    const unsigned int embed_dim = 16;

    const std::vector<float> input_h = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,

        -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f,
        -0.9f, -1.0f, -1.1f, -1.2f, -1.3f, -1.4f, -1.5f, -1.6f,

        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

    std::vector<float> weight_h;
    weight_h.reserve(embed_dim);
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h.push_back(static_cast<float>(d + 1));

    run_rmsnorm_test<half>(input_h, weight_h, num_tokens, embed_dim, 2e-2f);
}

TEST(RMSNormHalfTest, ConstantInputAndWeights)
{
    const unsigned int num_tokens = 5;
    const unsigned int embed_dim = 32;

    const std::vector<float> input_h(num_tokens * embed_dim, 2.0f);
    const std::vector<float> weight_h(embed_dim, 0.5f);

    run_rmsnorm_test<half>(input_h, weight_h, num_tokens, embed_dim, 1e-3f);
}

TEST(RMSNormHalfTest, LargeBatchRandomValues)
{
    const unsigned int num_tokens = 64;
    const unsigned int embed_dim = 64;
    const unsigned int numel = num_tokens * embed_dim;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> input_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> weight_dist(0.1f, 2.0f);

    std::vector<float> input_h(numel);
    std::vector<float> weight_h(embed_dim);
    for (unsigned int i = 0; i < numel; ++i)
        input_h[i] = input_dist(gen);
    for (unsigned int d = 0; d < embed_dim; ++d)
        weight_h[d] = weight_dist(gen);

    run_rmsnorm_test<half>(input_h, weight_h, num_tokens, embed_dim, 1e-2f);
}
