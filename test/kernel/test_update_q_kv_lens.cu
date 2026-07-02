#include <gtest/gtest.h>
#include <kernel/update_q_kv_lens.h>
#include <core/Tensor.cuh>

#include <vector>
#include <algorithm>

// ------------------------------------------------------------------
// Helper: compute expected results on the host.
// ------------------------------------------------------------------
static void compute_expected(const std::vector<unsigned int> &q_lens,
                             const std::vector<unsigned int> &kv_lens,
                             std::vector<unsigned int> &expected_kv_lens,
                             unsigned int &expected_max_q_len,
                             unsigned int &expected_max_kv_len)
{
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());
    expected_kv_lens.resize(batch_size);

    expected_max_q_len = 0;
    expected_max_kv_len = 0;
    for (unsigned int i = 0; i < batch_size; ++i)
    {
        expected_max_q_len = std::max(expected_max_q_len, q_lens[i]);
        expected_kv_lens[i] = kv_lens[i] + q_lens[i];
        expected_max_kv_len = std::max(expected_max_kv_len, expected_kv_lens[i]);
    }
}

// ------------------------------------------------------------------
// update_q_kv_lens kernel tests
// ------------------------------------------------------------------
TEST(UpdateQKVLensTest, BasicTwoSequences)
{
    const std::vector<unsigned int> q_lens = {2, 3};
    const std::vector<unsigned int> kv_lens = {4, 5};
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> max_q_len({1}, CPU);
    Tensor<unsigned int> max_kv_len({1}, CPU);

    launch_update_q_kv_lens(q_lens_d, kv_lens_d, max_q_len, max_kv_len);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned int> kv_lens_h({batch_size}, CPU);
    CUDA_CHECK(cudaMemcpy(kv_lens_h.data(), kv_lens_d.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> expected_kv_lens;
    unsigned int expected_max_q_len = 0;
    unsigned int expected_max_kv_len = 0;
    compute_expected(q_lens, kv_lens, expected_kv_lens,
                     expected_max_q_len, expected_max_kv_len);

    for (unsigned int i = 0; i < batch_size; ++i)
    {
        ASSERT_EQ(kv_lens_h.data()[i], expected_kv_lens[i])
            << "kv_lens mismatch at idx=" << i;
    }
    ASSERT_EQ(max_q_len.data()[0], expected_max_q_len);
    ASSERT_EQ(max_kv_len.data()[0], expected_max_kv_len);
}

TEST(UpdateQKVLensTest, SingleSequence)
{
    const std::vector<unsigned int> q_lens = {3};
    const std::vector<unsigned int> kv_lens = {7};
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> max_q_len({1}, CPU);
    Tensor<unsigned int> max_kv_len({1}, CPU);

    launch_update_q_kv_lens(q_lens_d, kv_lens_d, max_q_len, max_kv_len);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned int> kv_lens_h({batch_size}, CPU);
    CUDA_CHECK(cudaMemcpy(kv_lens_h.data(), kv_lens_d.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> expected_kv_lens;
    unsigned int expected_max_q_len = 0;
    unsigned int expected_max_kv_len = 0;
    compute_expected(q_lens, kv_lens, expected_kv_lens,
                     expected_max_q_len, expected_max_kv_len);

    ASSERT_EQ(kv_lens_h.data()[0], expected_kv_lens[0]);
    ASSERT_EQ(max_q_len.data()[0], expected_max_q_len);
    ASSERT_EQ(max_kv_len.data()[0], expected_max_kv_len);
}

TEST(UpdateQKVLensTest, FirstTokenZeroKVCache)
{
    // First forward pass: no history, kv_lens are all zero.
    const std::vector<unsigned int> q_lens = {2, 5, 3};
    const std::vector<unsigned int> kv_lens = {0, 0, 0};
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> max_q_len({1}, CPU);
    Tensor<unsigned int> max_kv_len({1}, CPU);

    launch_update_q_kv_lens(q_lens_d, kv_lens_d, max_q_len, max_kv_len);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned int> kv_lens_h({batch_size}, CPU);
    CUDA_CHECK(cudaMemcpy(kv_lens_h.data(), kv_lens_d.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> expected_kv_lens;
    unsigned int expected_max_q_len = 0;
    unsigned int expected_max_kv_len = 0;
    compute_expected(q_lens, kv_lens, expected_kv_lens,
                     expected_max_q_len, expected_max_kv_len);

    for (unsigned int i = 0; i < batch_size; ++i)
    {
        ASSERT_EQ(kv_lens_h.data()[i], expected_kv_lens[i])
            << "kv_lens mismatch at idx=" << i;
    }
    ASSERT_EQ(max_q_len.data()[0], expected_max_q_len);
    ASSERT_EQ(max_kv_len.data()[0], expected_max_kv_len);
}

TEST(UpdateQKVLensTest, IncrementalDecoding)
{
    // Each sequence has a single new query token and a growing KV cache.
    const std::vector<unsigned int> q_lens = {1, 1, 1, 1};
    const std::vector<unsigned int> kv_lens = {3, 5, 7, 2};
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> max_q_len({1}, CPU);
    Tensor<unsigned int> max_kv_len({1}, CPU);

    launch_update_q_kv_lens(q_lens_d, kv_lens_d, max_q_len, max_kv_len);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned int> kv_lens_h({batch_size}, CPU);
    CUDA_CHECK(cudaMemcpy(kv_lens_h.data(), kv_lens_d.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> expected_kv_lens;
    unsigned int expected_max_q_len = 0;
    unsigned int expected_max_kv_len = 0;
    compute_expected(q_lens, kv_lens, expected_kv_lens,
                     expected_max_q_len, expected_max_kv_len);

    for (unsigned int i = 0; i < batch_size; ++i)
    {
        ASSERT_EQ(kv_lens_h.data()[i], expected_kv_lens[i])
            << "kv_lens mismatch at idx=" << i;
    }
    ASSERT_EQ(max_q_len.data()[0], expected_max_q_len);
    ASSERT_EQ(max_kv_len.data()[0], expected_max_kv_len);
}

TEST(UpdateQKVLensTest, LargeBatch)
{
    const unsigned int batch_size = 64;
    std::vector<unsigned int> q_lens(batch_size);
    std::vector<unsigned int> kv_lens(batch_size);
    for (unsigned int i = 0; i < batch_size; ++i)
    {
        q_lens[i] = (i % 8) + 1;
        kv_lens[i] = (i % 16) * 2;
    }

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> max_q_len({1}, CPU);
    Tensor<unsigned int> max_kv_len({1}, CPU);

    launch_update_q_kv_lens(q_lens_d, kv_lens_d, max_q_len, max_kv_len);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned int> kv_lens_h({batch_size}, CPU);
    CUDA_CHECK(cudaMemcpy(kv_lens_h.data(), kv_lens_d.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> expected_kv_lens;
    unsigned int expected_max_q_len = 0;
    unsigned int expected_max_kv_len = 0;
    compute_expected(q_lens, kv_lens, expected_kv_lens,
                     expected_max_q_len, expected_max_kv_len);

    for (unsigned int i = 0; i < batch_size; ++i)
    {
        ASSERT_EQ(kv_lens_h.data()[i], expected_kv_lens[i])
            << "kv_lens mismatch at idx=" << i;
    }
    ASSERT_EQ(max_q_len.data()[0], expected_max_q_len);
    ASSERT_EQ(max_kv_len.data()[0], expected_max_kv_len);
}
