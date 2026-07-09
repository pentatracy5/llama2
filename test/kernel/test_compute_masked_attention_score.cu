#include <gtest/gtest.h>
#include <kernel/compute_masked_attention_score.h>
#include <core/Tensor.cuh>
#include <common/config.h>

#include <vector>
#include <cmath>
#include <random>
#include <limits>

// ------------------------------------------------------------------
// Helper: compute expected masked attention score on the host.
//
// Mirrors the kernel behavior:
//   - scale = 1 / sqrt(head_dim)
//   - masked positions contribute 0
//   - denominator uses SOFTMAX_EPS (to match the kernel)
//   - q positions >= q_lens are left as 0
//   - kv positions >= max_kv_len are left as 0
// ------------------------------------------------------------------
template <typename T>
static void compute_expected_masked_attention_score(const std::vector<T> &qk_host,
                                                    const std::vector<unsigned char> &mask_host,
                                                    const std::vector<unsigned int> &q_lens,
                                                    const unsigned int batch_size,
                                                    const unsigned int q_head_num,
                                                    const unsigned int q_cache_len,
                                                    const unsigned int kv_cache_len,
                                                    const unsigned int max_kv_len,
                                                    const unsigned int head_dim,
                                                    std::vector<T> &expected)
{
    expected.resize(qk_host.size());
    std::fill(expected.begin(), expected.end(), T(0));

    const float scale = 1.f / sqrtf(static_cast<float>(head_dim));

    for (unsigned int seq_id = 0; seq_id < batch_size; ++seq_id)
    {
        const unsigned int q_len = q_lens[seq_id];
        for (unsigned int head_id = 0; head_id < q_head_num; ++head_id)
        {
            for (unsigned int q_id = 0; q_id < q_len; ++q_id)
            {
                // compute max over unmasked positions
                float max_val = -std::numeric_limits<float>::infinity();
                for (unsigned int k_id = 0; k_id < max_kv_len; ++k_id)
                {
                    const unsigned int mask_offset = ((seq_id * q_cache_len + q_id) * kv_cache_len + k_id);
                    if (mask_host[mask_offset])
                    {
                        const unsigned int qk_offset = (((seq_id * q_head_num + head_id) * q_cache_len + q_id) * kv_cache_len + k_id);
                        const float val = static_cast<float>(qk_host[qk_offset]) * scale;
                        max_val = std::max(max_val, val);
                    }
                }

                // compute sum of exp(score - max)
                float sum = 0.f;
                for (unsigned int k_id = 0; k_id < max_kv_len; ++k_id)
                {
                    const unsigned int mask_offset = ((seq_id * q_cache_len + q_id) * kv_cache_len + k_id);
                    if (mask_host[mask_offset])
                    {
                        const unsigned int qk_offset = (((seq_id * q_head_num + head_id) * q_cache_len + q_id) * kv_cache_len + k_id);
                        const float val = static_cast<float>(qk_host[qk_offset]) * scale;
                        sum += expf(val - max_val);
                    }
                }

                // write softmax output
                for (unsigned int k_id = 0; k_id < kv_cache_len; ++k_id)
                {
                    const unsigned int mask_offset = ((seq_id * q_cache_len + q_id) * kv_cache_len + k_id);
                    const unsigned int qk_offset = (((seq_id * q_head_num + head_id) * q_cache_len + q_id) * kv_cache_len + k_id);
                    if (mask_host[mask_offset] && k_id < max_kv_len)
                    {
                        const float val = static_cast<float>(qk_host[qk_offset]) * scale;
                        expected[qk_offset] = static_cast<T>(expf(val - max_val) / (sum + SOFTMAX_EPS));
                    }
                    else
                    {
                        expected[qk_offset] = T(0);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Helper: fill qk with deterministic pseudo-random values.
// ------------------------------------------------------------------
template <typename T>
static void fill_random_qk(std::vector<T> &qk_host)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (auto &v : qk_host)
    {
        v = static_cast<T>(dist(gen));
    }
}

// ------------------------------------------------------------------
// Helper: run one test case and compare with expected output.
// ------------------------------------------------------------------
template <typename T>
static void run_masked_attention_score_test(const unsigned int batch_size,
                                            const unsigned int q_head_num,
                                            const unsigned int q_cache_len,
                                            const unsigned int kv_cache_len,
                                            const unsigned int max_q_len,
                                            const unsigned int max_kv_len,
                                            const unsigned int head_dim,
                                            const std::vector<unsigned int> &q_lens,
                                            const std::vector<unsigned char> &mask_host,
                                            const std::vector<T> &qk_host_input,
                                            const float tolerance)
{
    // Zero out padding positions in the input so that positions the kernel does
    // not touch remain 0 and can be compared against expected zeros.
    std::vector<T> qk_host = qk_host_input;
    for (unsigned int seq_id = 0; seq_id < batch_size; ++seq_id)
    {
        for (unsigned int head_id = 0; head_id < q_head_num; ++head_id)
        {
            for (unsigned int q_id = 0; q_id < q_cache_len; ++q_id)
            {
                const bool q_valid = q_id < q_lens[seq_id];
                for (unsigned int k_id = 0; k_id < kv_cache_len; ++k_id)
                {
                    const unsigned int qk_offset = (((seq_id * q_head_num + head_id) * q_cache_len + q_id) * kv_cache_len + k_id);
                    if (!q_valid || k_id >= max_kv_len)
                    {
                        qk_host[qk_offset] = T(0);
                    }
                }
            }
        }
    }

    // device tensors
    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned char> mask_d({batch_size, q_cache_len, kv_cache_len}, GPU);
    Tensor<T> qk_d({batch_size, q_head_num, q_cache_len, kv_cache_len}, GPU);

    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mask_d.data(), mask_host.data(),
                          mask_host.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(qk_d.data(), qk_host.data(),
                          qk_host.size() * sizeof(T), cudaMemcpyHostToDevice));

    // launch kernel
    launch_compute_masked_attention_score<T, unsigned char>(head_dim,
                                                            max_q_len,
                                                            max_kv_len,
                                                            q_lens_d,
                                                            mask_d,
                                                            qk_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    // copy result back
    Tensor<T> qk_h({batch_size, q_head_num, q_cache_len, kv_cache_len}, CPU);
    CUDA_CHECK(cudaMemcpy(qk_h.data(), qk_d.data(),
                          qk_d.numel() * sizeof(T), cudaMemcpyDeviceToHost));

    // compute expected
    std::vector<T> expected;
    compute_expected_masked_attention_score(qk_host, mask_host, q_lens,
                                            batch_size, q_head_num, q_cache_len,
                                            kv_cache_len, max_kv_len, head_dim,
                                            expected);

    // compare
    for (unsigned int i = 0; i < qk_host.size(); ++i)
    {
        const float actual = static_cast<float>(qk_h.data()[i]);
        const float expect = static_cast<float>(expected[i]);
        ASSERT_NEAR(actual, expect, tolerance)
            << "mismatch at idx=" << i
            << ", actual=" << actual
            << ", expected=" << expect;
    }
}

// ------------------------------------------------------------------
// Test: basic masked softmax with all positions valid.
// ------------------------------------------------------------------
TEST(ComputeMaskedAttentionScoreTest, BasicAllValidFloat)
{
    const unsigned int batch_size = 1;
    const unsigned int q_head_num = 1;
    const unsigned int q_cache_len = 4;
    const unsigned int kv_cache_len = 4;
    const unsigned int max_q_len = 4;
    const unsigned int max_kv_len = 4;
    const unsigned int head_dim = 8;

    const std::vector<unsigned int> q_lens = {4};
    const std::vector<unsigned char> mask_host(batch_size * q_cache_len * kv_cache_len, 1);
    std::vector<float> qk_host(batch_size * q_head_num * q_cache_len * kv_cache_len);
    fill_random_qk(qk_host);

    run_masked_attention_score_test(batch_size, q_head_num, q_cache_len, kv_cache_len,
                                    max_q_len, max_kv_len, head_dim, q_lens,
                                    mask_host, qk_host, 1e-5f);
}

// ------------------------------------------------------------------
// Test: some positions are masked out.
// ------------------------------------------------------------------
TEST(ComputeMaskedAttentionScoreTest, PartialMaskFloat)
{
    const unsigned int batch_size = 1;
    const unsigned int q_head_num = 2;
    const unsigned int q_cache_len = 4;
    const unsigned int kv_cache_len = 6;
    const unsigned int max_q_len = 4;
    const unsigned int max_kv_len = 6;
    const unsigned int head_dim = 16;

    const std::vector<unsigned int> q_lens = {4};
    std::vector<unsigned char> mask_host(batch_size * q_cache_len * kv_cache_len, 0);
    // mask a triangular-ish pattern
    for (unsigned int q_id = 0; q_id < q_cache_len; ++q_id)
    {
        for (unsigned int k_id = 0; k_id <= q_id && k_id < max_kv_len; ++k_id)
        {
            mask_host[(0 * q_cache_len + q_id) * kv_cache_len + k_id] = 1;
        }
    }

    std::vector<float> qk_host(batch_size * q_head_num * q_cache_len * kv_cache_len);
    fill_random_qk(qk_host);

    run_masked_attention_score_test(batch_size, q_head_num, q_cache_len, kv_cache_len,
                                    max_q_len, max_kv_len, head_dim, q_lens,
                                    mask_host, qk_host, 1e-5f);
}

// ------------------------------------------------------------------
// Test: batch with different q_lens and padding positions.
// ------------------------------------------------------------------
TEST(ComputeMaskedAttentionScoreTest, BatchWithPaddingFloat)
{
    const unsigned int batch_size = 3;
    const unsigned int q_head_num = 2;
    const unsigned int q_cache_len = 8;
    const unsigned int kv_cache_len = 8;
    const unsigned int max_q_len = 6;
    const unsigned int max_kv_len = 8;
    const unsigned int head_dim = 32;

    const std::vector<unsigned int> q_lens = {3, 6, 2};
    std::vector<unsigned char> mask_host(batch_size * q_cache_len * kv_cache_len, 0);
    for (unsigned int seq_id = 0; seq_id < batch_size; ++seq_id)
    {
        for (unsigned int q_id = 0; q_id < q_lens[seq_id]; ++q_id)
        {
            for (unsigned int k_id = 0; k_id < max_kv_len; ++k_id)
            {
                // simple pattern: even positions are valid
                if ((k_id % 2) == 0)
                {
                    mask_host[(seq_id * q_cache_len + q_id) * kv_cache_len + k_id] = 1;
                }
            }
        }
    }

    std::vector<float> qk_host(batch_size * q_head_num * q_cache_len * kv_cache_len);
    fill_random_qk(qk_host);

    run_masked_attention_score_test(batch_size, q_head_num, q_cache_len, kv_cache_len,
                                    max_q_len, max_kv_len, head_dim, q_lens,
                                    mask_host, qk_host, 1e-4f);
}

// ------------------------------------------------------------------
// Test: all positions masked -> output should be all zeros.
// ------------------------------------------------------------------
TEST(ComputeMaskedAttentionScoreTest, AllMaskedFloat)
{
    const unsigned int batch_size = 2;
    const unsigned int q_head_num = 2;
    const unsigned int q_cache_len = 4;
    const unsigned int kv_cache_len = 4;
    const unsigned int max_q_len = 4;
    const unsigned int max_kv_len = 4;
    const unsigned int head_dim = 8;

    const std::vector<unsigned int> q_lens = {4, 4};
    const std::vector<unsigned char> mask_host(batch_size * q_cache_len * kv_cache_len, 0);
    std::vector<float> qk_host(batch_size * q_head_num * q_cache_len * kv_cache_len);
    fill_random_qk(qk_host);

    run_masked_attention_score_test(batch_size, q_head_num, q_cache_len, kv_cache_len,
                                    max_q_len, max_kv_len, head_dim, q_lens,
                                    mask_host, qk_host, 1e-6f);
}

// ------------------------------------------------------------------
// Test: max_kv_len smaller than kv_cache_len.
// ------------------------------------------------------------------
TEST(ComputeMaskedAttentionScoreTest, MaxKVSsmallerThanKVCacheFloat)
{
    const unsigned int batch_size = 1;
    const unsigned int q_head_num = 1;
    const unsigned int q_cache_len = 4;
    const unsigned int kv_cache_len = 8;
    const unsigned int max_q_len = 4;
    const unsigned int max_kv_len = 5;
    const unsigned int head_dim = 8;

    const std::vector<unsigned int> q_lens = {4};
    std::vector<unsigned char> mask_host(batch_size * q_cache_len * kv_cache_len, 0);
    for (unsigned int q_id = 0; q_id < q_cache_len; ++q_id)
    {
        for (unsigned int k_id = 0; k_id < max_kv_len; ++k_id)
        {
            mask_host[(0 * q_cache_len + q_id) * kv_cache_len + k_id] = 1;
        }
    }

    std::vector<float> qk_host(batch_size * q_head_num * q_cache_len * kv_cache_len);
    fill_random_qk(qk_host);

    run_masked_attention_score_test(batch_size, q_head_num, q_cache_len, kv_cache_len,
                                    max_q_len, max_kv_len, head_dim, q_lens,
                                    mask_host, qk_host, 1e-5f);
}

// ------------------------------------------------------------------
// Test: half precision path.
// ------------------------------------------------------------------
TEST(ComputeMaskedAttentionScoreTest, BasicHalf)
{
    const unsigned int batch_size = 2;
    const unsigned int q_head_num = 4;
    const unsigned int q_cache_len = 4;
    const unsigned int kv_cache_len = 4;
    const unsigned int max_q_len = 4;
    const unsigned int max_kv_len = 4;
    const unsigned int head_dim = 16;

    const std::vector<unsigned int> q_lens = {3, 4};
    std::vector<unsigned char> mask_host(batch_size * q_cache_len * kv_cache_len, 0);
    for (unsigned int seq_id = 0; seq_id < batch_size; ++seq_id)
    {
        for (unsigned int q_id = 0; q_id < q_lens[seq_id]; ++q_id)
        {
            for (unsigned int k_id = 0; k_id <= q_id && k_id < max_kv_len; ++k_id)
            {
                mask_host[(seq_id * q_cache_len + q_id) * kv_cache_len + k_id] = 1;
            }
        }
    }

    std::vector<half> qk_host(batch_size * q_head_num * q_cache_len * kv_cache_len);
    fill_random_qk(qk_host);

    run_masked_attention_score_test<half>(batch_size, q_head_num, q_cache_len, kv_cache_len,
                                          max_q_len, max_kv_len, head_dim, q_lens,
                                          mask_host, qk_host, 1e-3f);
}

// ------------------------------------------------------------------
// Test: single token decode scenario.
// ------------------------------------------------------------------
TEST(ComputeMaskedAttentionScoreTest, SingleTokenFloat)
{
    const unsigned int batch_size = 1;
    const unsigned int q_head_num = 1;
    const unsigned int q_cache_len = 1;
    const unsigned int kv_cache_len = 1;
    const unsigned int max_q_len = 1;
    const unsigned int max_kv_len = 1;
    const unsigned int head_dim = 8;

    const std::vector<unsigned int> q_lens = {1};
    const std::vector<unsigned char> mask_host(batch_size * q_cache_len * kv_cache_len, 1);
    std::vector<float> qk_host(batch_size * q_head_num * q_cache_len * kv_cache_len);
    fill_random_qk(qk_host);

    run_masked_attention_score_test(batch_size, q_head_num, q_cache_len, kv_cache_len,
                                    max_q_len, max_kv_len, head_dim, q_lens,
                                    mask_host, qk_host, 1e-5f);
}
