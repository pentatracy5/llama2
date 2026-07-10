#include <gtest/gtest.h>
#include <kernel/fused_transpose_unpad.h>
#include <core/Tensor.cuh>
#include <common/config.h>

#include <vector>
#include <numeric>
#include <algorithm>
#include <cuda_fp16.h>

// ------------------------------------------------------------------
// Helpers: scalar conversion so we can test both float and half.
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
// Helpers: compute prefix sum of sequence lengths and the token-level
// mapping from unpad positions to padded (seq_id, token_id) positions.
// ------------------------------------------------------------------
static std::vector<unsigned int> compute_seq_offsets(const std::vector<unsigned int> &q_lens)
{
    std::vector<unsigned int> seq_offsets(q_lens.size());
    unsigned int accumulate_len = 0;
    for (size_t i = 0; i < q_lens.size(); ++i)
    {
        seq_offsets[i] = accumulate_len;
        accumulate_len += q_lens[i];
    }
    return seq_offsets;
}

static std::vector<unsigned int> compute_unpad_to_pad_idx(
    unsigned int batch_size,
    unsigned int q_cache_len,
    const std::vector<unsigned int> &q_lens)
{
    const std::vector<unsigned int> seq_offsets = compute_seq_offsets(q_lens);
    std::vector<unsigned int> unpad_to_pad_idx(batch_size * q_cache_len, 0);
    for (unsigned int b = 0; b < batch_size; ++b)
    {
        for (unsigned int t = 0; t < q_lens[b]; ++t)
        {
            const unsigned int global_token = seq_offsets[b] + t;
            unpad_to_pad_idx[global_token] = b * q_cache_len + t;
        }
    }
    return unpad_to_pad_idx;
}

// ------------------------------------------------------------------
// Generic runner.
// ------------------------------------------------------------------
template <typename T>
static void run_fused_transpose_unpad_test(unsigned int batch_size,
                                           unsigned int q_head_num,
                                           unsigned int q_cache_len,
                                           unsigned int head_size,
                                           const std::vector<unsigned int> &q_lens,
                                           float tolerance)
{
    const unsigned int num_input_tokens =
        std::accumulate(q_lens.begin(), q_lens.end(), 0u);

    ASSERT_EQ(q_lens.size(), batch_size);

    // Fill padded qkv with deterministic values.  Padding positions are also
    // filled, but the kernel should only read the first q_lens[b] tokens.
    const unsigned int qkv_numel = batch_size * q_head_num * q_cache_len * head_size;
    std::vector<float> qkv_f(qkv_numel);
    for (unsigned int i = 0; i < qkv_numel; ++i)
        qkv_f[i] = static_cast<float>(i % 53) * 0.1f - 1.0f;

    const std::vector<T> qkv_h = vector_from_float<T>(qkv_f);
    // Use the actual values that will be copied to the device (after the
    // float<->half round trip) when computing the expected result.
    const std::vector<float> qkv_expected_f = vector_to_float(qkv_h);

    Tensor<T> qkv_d({batch_size, q_head_num, q_cache_len, head_size}, GPU);
    CUDA_CHECK(cudaMemcpy(qkv_d.data(), qkv_h.data(),
                          qkv_h.size() * sizeof(T), cudaMemcpyHostToDevice));

    const std::vector<unsigned int> unpad_to_pad_idx =
        compute_unpad_to_pad_idx(batch_size, q_cache_len, q_lens);
    Tensor<unsigned int> unpad_to_pad_idx_d({batch_size, q_cache_len}, GPU);
    CUDA_CHECK(cudaMemcpy(unpad_to_pad_idx_d.data(), unpad_to_pad_idx.data(),
                          unpad_to_pad_idx.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<T> unpad_qkv_d({num_input_tokens, q_head_num * head_size}, GPU);
    CUDA_CHECK(cudaMemset(unpad_qkv_d.data(), 0,
                          unpad_qkv_d.numel() * sizeof(T)));

    launch_fused_transpose_unpad<T>(num_input_tokens, qkv_d, unpad_to_pad_idx_d, unpad_qkv_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<T> unpad_qkv_h(unpad_qkv_d.numel());
    CUDA_CHECK(cudaMemcpy(unpad_qkv_h.data(), unpad_qkv_d.data(),
                          unpad_qkv_h.size() * sizeof(T), cudaMemcpyDeviceToHost));
    const std::vector<float> output_f = vector_to_float(unpad_qkv_h);

    // Build expected result on the host.  Invalid/padding positions stay 0
    // because we memset the output tensor before launching.
    const std::vector<unsigned int> seq_offsets = compute_seq_offsets(q_lens);
    std::vector<float> expected(num_input_tokens * q_head_num * head_size, 0.0f);
    for (unsigned int b = 0; b < batch_size; ++b)
    {
        for (unsigned int t = 0; t < q_lens[b]; ++t)
        {
            const unsigned int global_token = seq_offsets[b] + t;
            for (unsigned int h = 0; h < q_head_num; ++h)
            {
                for (unsigned int d = 0; d < head_size; ++d)
                {
                    const unsigned int src_offset =
                        ((b * q_head_num + h) * q_cache_len + t) * head_size + d;
                    const unsigned int dst_offset =
                        (global_token * q_head_num + h) * head_size + d;
                    expected[dst_offset] = qkv_expected_f[src_offset];
                }
            }
        }
    }

    ASSERT_EQ(output_f.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        ASSERT_NEAR(output_f[i], expected[i], tolerance)
            << "mismatch at idx=" << i;
    }
}

// ------------------------------------------------------------------
// fused_transpose_unpad kernel tests (FP32)
// ------------------------------------------------------------------
TEST(FusedTransposeUnpadTest, BasicTwoSequences)
{
    const unsigned int batch_size = 3;
    const unsigned int q_head_num = 4;
    const unsigned int q_cache_len = 16;
    const unsigned int head_size = 16;
    const std::vector<unsigned int> q_lens = {2, 5, 3};

    run_fused_transpose_unpad_test<float>(batch_size, q_head_num, q_cache_len,
                                          head_size, q_lens, 1e-5f);
}

TEST(FusedTransposeUnpadTest, SingleSequence)
{
    const unsigned int batch_size = 1;
    const unsigned int q_head_num = 2;
    const unsigned int q_cache_len = 8;
    const unsigned int head_size = 16;
    const std::vector<unsigned int> q_lens = {6};

    run_fused_transpose_unpad_test<float>(batch_size, q_head_num, q_cache_len,
                                          head_size, q_lens, 1e-5f);
}

TEST(FusedTransposeUnpadTest, IncrementalDecoding)
{
    // One new token per sequence; most of the q_cache_len is padding.
    const unsigned int batch_size = 4;
    const unsigned int q_head_num = 2;
    const unsigned int q_cache_len = 8;
    const unsigned int head_size = 16;
    const std::vector<unsigned int> q_lens = {1, 1, 1, 1};

    run_fused_transpose_unpad_test<float>(batch_size, q_head_num, q_cache_len,
                                          head_size, q_lens, 1e-5f);
}

TEST(FusedTransposeUnpadTest, LargerHeadDim)
{
    // head_size = 64 exercises the inner vector loop (vlen = 4 for float).
    const unsigned int batch_size = 2;
    const unsigned int q_head_num = 2;
    const unsigned int q_cache_len = 32;
    const unsigned int head_size = 64;
    const std::vector<unsigned int> q_lens = {7, 9};

    run_fused_transpose_unpad_test<float>(batch_size, q_head_num, q_cache_len,
                                          head_size, q_lens, 1e-5f);
}

// ------------------------------------------------------------------
// fused_transpose_unpad kernel tests (FP16)
// ------------------------------------------------------------------
TEST(FusedTransposeUnpadTest, HalfBasic)
{
    const unsigned int batch_size = 2;
    const unsigned int q_head_num = 2;
    const unsigned int q_cache_len = 8;
    const unsigned int head_size = 16; // divisible by 8 for half vector loads
    const std::vector<unsigned int> q_lens = {3, 4};

    run_fused_transpose_unpad_test<half>(batch_size, q_head_num, q_cache_len,
                                         head_size, q_lens, 1e-3f);
}
