#include <gtest/gtest.h>
#include <kernel/rmsnorm.h>
#include <core/Tensor.cuh>
#include <common/config.h>
#include <vector>
#include <cmath>
#include <random>

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
// RMSNorm kernel tests
// ------------------------------------------------------------------
TEST(RMSNormTest, SingleTokenUnitWeights)
{
    const unsigned int num_tokens = 1;
    const unsigned int embed_dim = 8;

    std::vector<float> input_h = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> weight_h(embed_dim, 1.0f);

    Tensor<float> input_d({num_tokens, embed_dim}, GPU);
    Tensor<float> weight_d({embed_dim}, GPU);
    Tensor<float> output_d({num_tokens, embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(input_d.data(), input_h.data(),
                          input_h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight_d.data(), weight_h.data(),
                          weight_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_rmsnorm(input_d, weight_d, output_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> output_h({num_tokens, embed_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d.data(),
                          output_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected;
    compute_expected_rmsnorm(input_h, weight_h, num_tokens, embed_dim, expected);

    for (unsigned int d = 0; d < embed_dim; ++d)
    {
        ASSERT_NEAR(output_h.data()[d], expected[d], 1e-4f)
            << "dim=" << d;
    }
}

TEST(RMSNormTest, MultipleTokensWithWeights)
{
    const unsigned int num_tokens = 3;
    const unsigned int embed_dim = 16;

    std::vector<float> input_h = {
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

    ASSERT_EQ(input_h.size(), num_tokens * embed_dim);

    Tensor<float> input_d({num_tokens, embed_dim}, GPU);
    Tensor<float> weight_d({embed_dim}, GPU);
    Tensor<float> output_d({num_tokens, embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(input_d.data(), input_h.data(),
                          input_h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight_d.data(), weight_h.data(),
                          weight_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_rmsnorm(input_d, weight_d, output_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> output_h({num_tokens, embed_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d.data(),
                          output_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected;
    compute_expected_rmsnorm(input_h, weight_h, num_tokens, embed_dim, expected);

    for (unsigned int t = 0; t < num_tokens; ++t)
    {
        for (unsigned int d = 0; d < embed_dim; ++d)
        {
            unsigned int idx = t * embed_dim + d;
            ASSERT_NEAR(output_h.data()[idx], expected[idx], 1e-4f)
                << "token=" << t << ", dim=" << d;
        }
    }
}

TEST(RMSNormTest, ConstantInputAndWeights)
{
    const unsigned int num_tokens = 5;
    const unsigned int embed_dim = 32;

    std::vector<float> input_h(num_tokens * embed_dim, 2.0f);
    std::vector<float> weight_h(embed_dim, 0.5f);

    Tensor<float> input_d({num_tokens, embed_dim}, GPU);
    Tensor<float> weight_d({embed_dim}, GPU);
    Tensor<float> output_d({num_tokens, embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(input_d.data(), input_h.data(),
                          input_h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight_d.data(), weight_h.data(),
                          weight_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_rmsnorm(input_d, weight_d, output_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> output_h({num_tokens, embed_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d.data(),
                          output_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected;
    compute_expected_rmsnorm(input_h, weight_h, num_tokens, embed_dim, expected);

    for (unsigned int idx = 0; idx < expected.size(); ++idx)
    {
        ASSERT_NEAR(output_h.data()[idx], expected[idx], 1e-4f)
            << "idx=" << idx;
    }
}

TEST(RMSNormTest, SmallDimMultipleOfVecLen)
{
    const unsigned int num_tokens = 4;
    const unsigned int embed_dim = 4;

    std::vector<float> input_h = {
        1.0f, 0.0f, -1.0f, 2.0f,
        3.0f, 1.0f, -2.0f, 0.5f,
        0.0f, 0.0f, 0.0f, 0.0f,
        -0.5f, -1.5f, 2.5f, -3.5f};
    std::vector<float> weight_h = {0.5f, 1.0f, 1.5f, 2.0f};

    Tensor<float> input_d({num_tokens, embed_dim}, GPU);
    Tensor<float> weight_d({embed_dim}, GPU);
    Tensor<float> output_d({num_tokens, embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(input_d.data(), input_h.data(),
                          input_h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight_d.data(), weight_h.data(),
                          weight_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_rmsnorm(input_d, weight_d, output_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> output_h({num_tokens, embed_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d.data(),
                          output_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected;
    compute_expected_rmsnorm(input_h, weight_h, num_tokens, embed_dim, expected);

    for (unsigned int idx = 0; idx < expected.size(); ++idx)
    {
        ASSERT_NEAR(output_h.data()[idx], expected[idx], 1e-4f)
            << "idx=" << idx;
    }
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

    Tensor<float> input_d({num_tokens, embed_dim}, GPU);
    Tensor<float> weight_d({embed_dim}, GPU);
    Tensor<float> output_d({num_tokens, embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(input_d.data(), input_h.data(),
                          input_h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight_d.data(), weight_h.data(),
                          weight_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_rmsnorm(input_d, weight_d, output_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> output_h({num_tokens, embed_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d.data(),
                          output_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected;
    compute_expected_rmsnorm(input_h, weight_h, num_tokens, embed_dim, expected);

    for (unsigned int idx = 0; idx < expected.size(); ++idx)
    {
        ASSERT_NEAR(output_h.data()[idx], expected[idx], 1e-3f)
            << "idx=" << idx;
    }
}
