#include <gtest/gtest.h>
#include <kernel/build_causal_mask.h>
#include <core/Tensor.cuh>

#include <vector>

// ------------------------------------------------------------------
// Helper: compute expected causal mask on the host.
// mask[seq_id, q_id, kv_id] == 1 iff
//   q_id < q_len and kv_id < kv_len and
//   q_id >= kv_id - (kv_len - q_len)
// ------------------------------------------------------------------
static void compute_expected_causal_mask(const std::vector<unsigned int> &q_lens,
                                         const std::vector<unsigned int> &kv_lens,
                                         unsigned int max_q_len,
                                         unsigned int max_kv_len,
                                         std::vector<unsigned char> &mask)
{
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());
    mask.resize(batch_size * max_q_len * max_kv_len);

    for (unsigned int seq_id = 0; seq_id < batch_size; ++seq_id)
    {
        const unsigned int q_len = q_lens[seq_id];
        const unsigned int kv_len = kv_lens[seq_id];
        const unsigned int seq_offset = seq_id * max_q_len * max_kv_len;

        for (unsigned int q_id = 0; q_id < max_q_len; ++q_id)
        {
            const unsigned int q_offset = q_id * max_kv_len + seq_offset;
            for (unsigned int kv_id = 0; kv_id < max_kv_len; ++kv_id)
            {
                const unsigned int kv_offset = q_offset + kv_id;
                const bool is_one = q_id < q_len &&
                                    kv_id < kv_len &&
                                    q_id >= kv_id - (kv_len - q_len);
                mask[kv_offset] = static_cast<unsigned char>(is_one);
            }
        }
    }
}

// ------------------------------------------------------------------
// build_causal_mask kernel tests
// ------------------------------------------------------------------
TEST(BuildCausalMaskTest, BasicTwoSequences)
{
    const std::vector<unsigned int> q_lens = {2, 3};
    const std::vector<unsigned int> kv_lens = {4, 5};
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());
    const unsigned int max_q_len = 3;
    const unsigned int max_kv_len = 5;

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> max_seq_len_h({1}, CPU);
    max_seq_len_h.data()[0] = max_q_len;

    Tensor<unsigned char> mask_d({batch_size, max_q_len, max_kv_len}, GPU);

    launch_build_causal_mask(max_seq_len_h, q_lens_d, kv_lens_d, mask_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned char> mask_h({batch_size, max_q_len, max_kv_len}, CPU);
    CUDA_CHECK(cudaMemcpy(mask_h.data(), mask_d.data(),
                          mask_d.numel() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    std::vector<unsigned char> expected;
    compute_expected_causal_mask(q_lens, kv_lens, max_q_len, max_kv_len, expected);

    for (unsigned int i = 0; i < mask_d.numel(); ++i)
    {
        ASSERT_EQ(mask_h.data()[i], expected[i]) << "mask mismatch at idx=" << i;
    }
}

TEST(BuildCausalMaskTest, EqualQueryAndKVLengths)
{
    const std::vector<unsigned int> q_lens = {3, 4};
    const std::vector<unsigned int> kv_lens = {3, 4};
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());
    const unsigned int max_q_len = 4;
    const unsigned int max_kv_len = 4;

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> max_seq_len_h({1}, CPU);
    max_seq_len_h.data()[0] = max_q_len;

    Tensor<unsigned char> mask_d({batch_size, max_q_len, max_kv_len}, GPU);

    launch_build_causal_mask(max_seq_len_h, q_lens_d, kv_lens_d, mask_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned char> mask_h({batch_size, max_q_len, max_kv_len}, CPU);
    CUDA_CHECK(cudaMemcpy(mask_h.data(), mask_d.data(),
                          mask_d.numel() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    std::vector<unsigned char> expected;
    compute_expected_causal_mask(q_lens, kv_lens, max_q_len, max_kv_len, expected);

    for (unsigned int i = 0; i < mask_d.numel(); ++i)
    {
        ASSERT_EQ(mask_h.data()[i], expected[i]) << "mask mismatch at idx=" << i;
    }
}

TEST(BuildCausalMaskTest, SingleToken)
{
    const std::vector<unsigned int> q_lens = {1};
    const std::vector<unsigned int> kv_lens = {1};
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());
    const unsigned int max_q_len = 4;
    const unsigned int max_kv_len = 4;

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> max_seq_len_h({1}, CPU);
    max_seq_len_h.data()[0] = max_q_len;

    Tensor<unsigned char> mask_d({batch_size, max_q_len, max_kv_len}, GPU);

    launch_build_causal_mask(max_seq_len_h, q_lens_d, kv_lens_d, mask_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned char> mask_h({batch_size, max_q_len, max_kv_len}, CPU);
    CUDA_CHECK(cudaMemcpy(mask_h.data(), mask_d.data(),
                          mask_d.numel() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    std::vector<unsigned char> expected;
    compute_expected_causal_mask(q_lens, kv_lens, max_q_len, max_kv_len, expected);

    for (unsigned int i = 0; i < mask_d.numel(); ++i)
    {
        ASSERT_EQ(mask_h.data()[i], expected[i]) << "mask mismatch at idx=" << i;
    }
}

TEST(BuildCausalMaskTest, IncrementalDecoding)
{
    // Each sequence has a single new query token and a growing KV cache.
    const std::vector<unsigned int> q_lens = {1, 1, 1};
    const std::vector<unsigned int> kv_lens = {3, 5, 7};
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());
    const unsigned int max_q_len = 1;
    const unsigned int max_kv_len = 8;

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> max_seq_len_h({1}, CPU);
    max_seq_len_h.data()[0] = max_q_len;

    Tensor<unsigned char> mask_d({batch_size, max_q_len, max_kv_len}, GPU);

    launch_build_causal_mask(max_seq_len_h, q_lens_d, kv_lens_d, mask_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned char> mask_h({batch_size, max_q_len, max_kv_len}, CPU);
    CUDA_CHECK(cudaMemcpy(mask_h.data(), mask_d.data(),
                          mask_d.numel() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    std::vector<unsigned char> expected;
    compute_expected_causal_mask(q_lens, kv_lens, max_q_len, max_kv_len, expected);

    for (unsigned int i = 0; i < mask_d.numel(); ++i)
    {
        ASSERT_EQ(mask_h.data()[i], expected[i]) << "mask mismatch at idx=" << i;
    }
}

TEST(BuildCausalMaskTest, LargeBatchRandomLengths)
{
    const unsigned int batch_size = 32;
    const unsigned int max_q_len = 16;
    const unsigned int max_kv_len = 32;

    std::vector<unsigned int> q_lens(batch_size);
    std::vector<unsigned int> kv_lens(batch_size);
    for (unsigned int i = 0; i < batch_size; ++i)
    {
        q_lens[i] = (i % max_q_len) + 1;
        kv_lens[i] = q_lens[i] + (i % (max_kv_len - max_q_len + 1));
    }

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> max_seq_len_h({1}, CPU);
    max_seq_len_h.data()[0] = max_q_len;

    Tensor<unsigned char> mask_d({batch_size, max_q_len, max_kv_len}, GPU);

    launch_build_causal_mask(max_seq_len_h, q_lens_d, kv_lens_d, mask_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned char> mask_h({batch_size, max_q_len, max_kv_len}, CPU);
    CUDA_CHECK(cudaMemcpy(mask_h.data(), mask_d.data(),
                          mask_d.numel() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    std::vector<unsigned char> expected;
    compute_expected_causal_mask(q_lens, kv_lens, max_q_len, max_kv_len, expected);

    for (unsigned int i = 0; i < mask_d.numel(); ++i)
    {
        ASSERT_EQ(mask_h.data()[i], expected[i]) << "mask mismatch at idx=" << i;
    }
}
