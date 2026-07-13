#include <gtest/gtest.h>
#include <kernel/swiglu.h>
#include <core/Tensor.cuh>
#include <common/config.h>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_fp16.h>

// ------------------------------------------------------------------
// Helpers: type conversion between float and the tested dtype.
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
// Helper: compute the expected SwiGLU result on CPU.
//
// input shape : (num_input_tokens, 2 * intermediate_size)
// output shape: (num_input_tokens, intermediate_size)
//
// For each token t and each element i in [0, intermediate_size):
//     x = input[t, i]
//     y = input[t, intermediate_size + i]
//     output[t, i] = x * sigmoid(x) * y
// ------------------------------------------------------------------
static void compute_expected_swiglu(const std::vector<float> &input_h,
                                    unsigned int num_input_tokens,
                                    unsigned int intermediate_size,
                                    std::vector<float> &expected_h)
{
    const unsigned int output_numel = num_input_tokens * intermediate_size;
    expected_h.resize(output_numel);

    for (unsigned int t = 0; t < num_input_tokens; ++t)
    {
        for (unsigned int i = 0; i < intermediate_size; ++i)
        {
            const unsigned int input_offset = t * 2 * intermediate_size;
            const float x = input_h[input_offset + i];
            const float y = input_h[input_offset + intermediate_size + i];
            const float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            expected_h[t * intermediate_size + i] = x * sigmoid_x * y;
        }
    }
}

// ------------------------------------------------------------------
// Generic runner: copy data to GPU, launch kernel, copy back, verify.
// ------------------------------------------------------------------
template <typename T>
static void run_swiglu_test(const std::vector<float> &input_h,
                            unsigned int num_input_tokens,
                            unsigned int intermediate_size,
                            float tolerance)
{
    const unsigned int input_numel = num_input_tokens * 2 * intermediate_size;
    const unsigned int output_numel = num_input_tokens * intermediate_size;
    ASSERT_EQ(input_h.size(), input_numel);

    const std::vector<T> input_d_h = vector_from_float<T>(input_h);

    Tensor<T> input_d({num_input_tokens, 2 * intermediate_size}, GPU);
    Tensor<T> output_d({num_input_tokens, intermediate_size}, GPU);

    CUDA_CHECK(cudaMemcpy(input_d.data(), input_d_h.data(),
                          input_d_h.size() * sizeof(T), cudaMemcpyHostToDevice));

    launch_swiglu<T>(input_d, output_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<T> output_d_h(output_d.numel());
    CUDA_CHECK(cudaMemcpy(output_d_h.data(), output_d.data(),
                          output_d_h.size() * sizeof(T), cudaMemcpyDeviceToHost));

    const std::vector<float> output_h = vector_to_float(output_d_h);

    std::vector<float> expected_h;
    compute_expected_swiglu(input_h, num_input_tokens, intermediate_size, expected_h);

    for (unsigned int idx = 0; idx < output_numel; ++idx)
    {
        ASSERT_NEAR(output_h[idx], expected_h[idx], tolerance)
            << "output mismatch at idx=" << idx;
    }
}

// ------------------------------------------------------------------
// Float tests
// ------------------------------------------------------------------
TEST(SwiGLUTest, SingleToken)
{
    const unsigned int num_input_tokens = 1;
    const unsigned int intermediate_size = 8;

    // First half: x values, second half: y values.
    const std::vector<float> input_h = {
        0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 1.5f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    run_swiglu_test<float>(input_h, num_input_tokens, intermediate_size, 1e-4f);
}

TEST(SwiGLUTest, MultipleTokens)
{
    const unsigned int num_input_tokens = 3;
    const unsigned int intermediate_size = 16;

    std::vector<float> input_h;
    input_h.reserve(num_input_tokens * 2 * intermediate_size);
    for (unsigned int t = 0; t < num_input_tokens; ++t)
    {
        for (unsigned int i = 0; i < intermediate_size; ++i)
            input_h.push_back(static_cast<float>(i) * 0.1f - 0.8f);
        for (unsigned int i = 0; i < intermediate_size; ++i)
            input_h.push_back(static_cast<float>(i + 1));
    }

    run_swiglu_test<float>(input_h, num_input_tokens, intermediate_size, 1e-4f);
}

TEST(SwiGLUTest, LargeBatchRandomValues)
{
    const unsigned int num_input_tokens = 64;
    const unsigned int intermediate_size = 64;
    const unsigned int input_numel = num_input_tokens * 2 * intermediate_size;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> input_h(input_numel);
    for (unsigned int i = 0; i < input_numel; ++i)
        input_h[i] = dist(gen);

    run_swiglu_test<float>(input_h, num_input_tokens, intermediate_size, 1e-3f);
}

// ------------------------------------------------------------------
// FP16 tests
// ------------------------------------------------------------------
TEST(SwiGLUHalfTest, SingleToken)
{
    const unsigned int num_input_tokens = 1;
    const unsigned int intermediate_size = 8;

    const std::vector<float> input_h = {
        0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 1.5f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    run_swiglu_test<half>(input_h, num_input_tokens, intermediate_size, 1e-2f);
}

TEST(SwiGLUHalfTest, MultipleTokens)
{
    const unsigned int num_input_tokens = 3;
    const unsigned int intermediate_size = 16;

    std::vector<float> input_h;
    input_h.reserve(num_input_tokens * 2 * intermediate_size);
    for (unsigned int t = 0; t < num_input_tokens; ++t)
    {
        for (unsigned int i = 0; i < intermediate_size; ++i)
            input_h.push_back(static_cast<float>(i) * 0.1f - 0.8f);
        for (unsigned int i = 0; i < intermediate_size; ++i)
            input_h.push_back(static_cast<float>(i + 1));
    }

    run_swiglu_test<half>(input_h, num_input_tokens, intermediate_size, 1e-2f);
}

TEST(SwiGLUHalfTest, LargeBatchRandomValues)
{
    const unsigned int num_input_tokens = 64;
    const unsigned int intermediate_size = 64;
    const unsigned int input_numel = num_input_tokens * 2 * intermediate_size;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> input_h(input_numel);
    for (unsigned int i = 0; i < input_numel; ++i)
        input_h[i] = dist(gen);

    run_swiglu_test<half>(input_h, num_input_tokens, intermediate_size, 1e-2f);
}
