#include <gtest/gtest.h>
#include <kernel/fused_reshape_rope_kvcache_decode.h>
#include <core/Tensor.cuh>
#include <common/config.h>

#include <cmath>
#include <vector>

// ------------------------------------------------------------------
// Helpers: compute expected results on the host for the generic RoPE
// scheme (any even head_dim). Each pair is formed by the adjacent
// elements (2*i, 2*i+1) of the head.
// ------------------------------------------------------------------
static void apply_rotation_decode(const std::vector<float> &head,
                                  unsigned int head_dim,
                                  unsigned int token_pos,
                                  unsigned int seq_len,
                                  std::vector<float> &out)
{
    out.resize(head_dim);
    for (unsigned int idx_in_head = 0; idx_in_head < head_dim / 2; ++idx_in_head)
    {
        float x0 = head[2 * idx_in_head];
        float x1 = head[2 * idx_in_head + 1];

        float theta = 1.f / std::pow(ROPE_BASE, 2.f * idx_in_head / head_dim);
        float alpha = 1.f / std::pow(std::max(1.f, static_cast<float>(seq_len) / TRAIN_SEQ_LEN),
                                     2.f * idx_in_head / (head_dim - 2));
        float cos_value = std::cos(token_pos * alpha * theta);
        float sin_value = std::sin(token_pos * alpha * theta);

        out[2 * idx_in_head] = cos_value * x0 - sin_value * x1;
        out[2 * idx_in_head + 1] = sin_value * x0 + cos_value * x1;
    }
}

static void compute_expected_decode(const std::vector<unsigned char> &is_done,
                                    const std::vector<unsigned int> &kv_lens,
                                    unsigned int kv_cache_len,
                                    unsigned int q_head_num,
                                    unsigned int kv_head_num,
                                    unsigned int head_dim,
                                    const std::vector<float> &input,
                                    std::vector<float> &expected_q,
                                    std::vector<float> &expected_k,
                                    std::vector<float> &expected_v)
{
    const unsigned int batch_size = static_cast<unsigned int>(is_done.size());
    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;

    expected_q.assign(batch_size * q_head_num * 1 * head_dim, 0.f);
    expected_k.assign(batch_size * kv_head_num * kv_cache_len * head_dim, 0.f);
    expected_v.assign(batch_size * kv_head_num * kv_cache_len * head_dim, 0.f);

    for (unsigned int seq_idx = 0; seq_idx < batch_size; ++seq_idx)
    {
        if (is_done[seq_idx])
            continue;

        const unsigned int kv_len = kv_lens[seq_idx];
        const unsigned int history_len = kv_len - 1;

        for (unsigned int h = 0; h < q_head_num; ++h)
        {
            unsigned int head_offset = (seq_idx * total_head_num + h) * head_dim;
            std::vector<float> head(input.begin() + head_offset,
                                    input.begin() + head_offset + head_dim);
            std::vector<float> rotated;
            apply_rotation_decode(head, head_dim, history_len, kv_len, rotated);

            unsigned int q_offset = (seq_idx * q_head_num + h) * head_dim;
            for (unsigned int d = 0; d < head_dim; ++d)
                expected_q[q_offset + d] = rotated[d];
        }

        for (unsigned int h = 0; h < kv_head_num; ++h)
        {
            unsigned int k_head_offset = (seq_idx * total_head_num + q_head_num + h) * head_dim;
            std::vector<float> k_head(input.begin() + k_head_offset,
                                      input.begin() + k_head_offset + head_dim);
            std::vector<float> rotated_k;
            apply_rotation_decode(k_head, head_dim, history_len, kv_len, rotated_k);

            unsigned int k_offset = ((seq_idx * kv_head_num + h) * kv_cache_len + history_len) * head_dim;
            for (unsigned int d = 0; d < head_dim; ++d)
                expected_k[k_offset + d] = rotated_k[d];
        }

        for (unsigned int h = 0; h < kv_head_num; ++h)
        {
            unsigned int v_head_offset = (seq_idx * total_head_num + q_head_num + kv_head_num + h) * head_dim;
            unsigned int v_offset = ((seq_idx * kv_head_num + h) * kv_cache_len + history_len) * head_dim;
            for (unsigned int d = 0; d < head_dim; ++d)
                expected_v[v_offset + d] = input[v_head_offset + d];
        }
    }
}

// ------------------------------------------------------------------
// fused_reshape_rope_kvcache_decode kernel tests
// ------------------------------------------------------------------
TEST(FusedReshapeRopeKVCacheDecodeTest, BasicTwoSequences)
{
    const unsigned int batch_size = 2;
    const unsigned int q_head_num = 4;
    const unsigned int kv_head_num = 2;
    const unsigned int head_dim = 64;
    const unsigned int kv_cache_len = 64;
    const std::vector<unsigned char> is_done = {0, 0};
    const std::vector<unsigned int> kv_lens = {3, 6};

    Tensor<unsigned char> is_done_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(is_done_d.data(), is_done.data(),
                          batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;
    std::vector<float> input(batch_size * total_head_num * head_dim);
    for (unsigned int i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i % 17) * 0.1f - 0.8f;

    Tensor<float> input_d({batch_size, total_head_num * head_dim}, GPU);
    CUDA_CHECK(cudaMemcpy(input_d.data(), input.data(),
                          input.size() * sizeof(float), cudaMemcpyHostToDevice));

    Tensor<float> q_d({batch_size, q_head_num, 1, head_dim}, GPU);
    Tensor<float> k_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);
    Tensor<float> v_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);

    launch_fused_reshape_rope_kvcache_decode(input_d, is_done_d, kv_lens_d, q_d, k_d, v_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> q_h({batch_size, q_head_num, 1, head_dim}, CPU);
    Tensor<float> k_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    Tensor<float> v_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(q_h.data(), q_d.data(), q_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_h.data(), k_d.data(), k_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_h.data(), v_d.data(), v_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_q, expected_k, expected_v;
    compute_expected_decode(is_done, kv_lens, kv_cache_len, q_head_num, kv_head_num, head_dim, input,
                            expected_q, expected_k, expected_v);

    for (unsigned int i = 0; i < q_h.numel(); ++i)
        ASSERT_NEAR(q_h.data()[i], expected_q[i], 1e-3f)
            << "q mismatch at idx=" << i;

    for (unsigned int i = 0; i < k_h.numel(); ++i)
        ASSERT_NEAR(k_h.data()[i], expected_k[i], 1e-3f)
            << "k mismatch at idx=" << i;

    for (unsigned int i = 0; i < v_h.numel(); ++i)
        ASSERT_NEAR(v_h.data()[i], expected_v[i], 1e-4f)
            << "v mismatch at idx=" << i;
}

TEST(FusedReshapeRopeKVCacheDecodeTest, FirstDecodeStep)
{
    const unsigned int batch_size = 1;
    const unsigned int q_head_num = 8;
    const unsigned int kv_head_num = 4;
    const unsigned int head_dim = 32;
    const unsigned int kv_cache_len = 32;
    const std::vector<unsigned char> is_done = {0};
    const std::vector<unsigned int> kv_lens = {1};

    Tensor<unsigned char> is_done_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(is_done_d.data(), is_done.data(),
                          batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;
    std::vector<float> input(batch_size * total_head_num * head_dim);
    for (unsigned int i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i) * 0.005f - 0.5f;

    Tensor<float> input_d({batch_size, total_head_num * head_dim}, GPU);
    CUDA_CHECK(cudaMemcpy(input_d.data(), input.data(),
                          input.size() * sizeof(float), cudaMemcpyHostToDevice));

    Tensor<float> q_d({batch_size, q_head_num, 1, head_dim}, GPU);
    Tensor<float> k_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);
    Tensor<float> v_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);

    launch_fused_reshape_rope_kvcache_decode(input_d, is_done_d, kv_lens_d, q_d, k_d, v_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> q_h({batch_size, q_head_num, 1, head_dim}, CPU);
    Tensor<float> k_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    Tensor<float> v_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(q_h.data(), q_d.data(), q_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_h.data(), k_d.data(), k_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_h.data(), v_d.data(), v_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_q, expected_k, expected_v;
    compute_expected_decode(is_done, kv_lens, kv_cache_len, q_head_num, kv_head_num, head_dim, input,
                            expected_q, expected_k, expected_v);

    for (unsigned int i = 0; i < q_h.numel(); ++i)
        ASSERT_NEAR(q_h.data()[i], expected_q[i], 1e-3f)
            << "q mismatch at idx=" << i;

    for (unsigned int i = 0; i < k_h.numel(); ++i)
        ASSERT_NEAR(k_h.data()[i], expected_k[i], 1e-3f)
            << "k mismatch at idx=" << i;

    for (unsigned int i = 0; i < v_h.numel(); ++i)
        ASSERT_NEAR(v_h.data()[i], expected_v[i], 1e-4f)
            << "v mismatch at idx=" << i;
}

TEST(FusedReshapeRopeKVCacheDecodeTest, WithDoneSequences)
{
    const unsigned int batch_size = 4;
    const unsigned int q_head_num = 8;
    const unsigned int kv_head_num = 4;
    const unsigned int head_dim = 64;
    const unsigned int kv_cache_len = 64;
    const std::vector<unsigned char> is_done = {0, 1, 0, 1};
    const std::vector<unsigned int> kv_lens = {4, 9, 15, 20};

    Tensor<unsigned char> is_done_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(is_done_d.data(), is_done.data(),
                          batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;
    std::vector<float> input(batch_size * total_head_num * head_dim);
    for (unsigned int i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i % 23) * 0.05f - 0.5f;

    Tensor<float> input_d({batch_size, total_head_num * head_dim}, GPU);
    CUDA_CHECK(cudaMemcpy(input_d.data(), input.data(),
                          input.size() * sizeof(float), cudaMemcpyHostToDevice));

    Tensor<float> q_d({batch_size, q_head_num, 1, head_dim}, GPU);
    Tensor<float> k_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);
    Tensor<float> v_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);

    launch_fused_reshape_rope_kvcache_decode(input_d, is_done_d, kv_lens_d, q_d, k_d, v_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> q_h({batch_size, q_head_num, 1, head_dim}, CPU);
    Tensor<float> k_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    Tensor<float> v_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(q_h.data(), q_d.data(), q_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_h.data(), k_d.data(), k_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_h.data(), v_d.data(), v_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_q, expected_k, expected_v;
    compute_expected_decode(is_done, kv_lens, kv_cache_len, q_head_num, kv_head_num, head_dim, input,
                            expected_q, expected_k, expected_v);

    for (unsigned int i = 0; i < q_h.numel(); ++i)
        ASSERT_NEAR(q_h.data()[i], expected_q[i], 1e-3f)
            << "q mismatch at idx=" << i;

    for (unsigned int i = 0; i < k_h.numel(); ++i)
        ASSERT_NEAR(k_h.data()[i], expected_k[i], 1e-3f)
            << "k mismatch at idx=" << i;

    for (unsigned int i = 0; i < v_h.numel(); ++i)
        ASSERT_NEAR(v_h.data()[i], expected_v[i], 1e-4f)
            << "v mismatch at idx=" << i;
}

TEST(FusedReshapeRopeKVCacheDecodeTest, LargeBatchLoopOverGrid)
{
    // batch_size > NUM_BLOCKS_X (256) forces a block to loop over multiple sequences.
    const unsigned int batch_size = 300;
    const unsigned int q_head_num = 4;
    const unsigned int kv_head_num = 2;
    const unsigned int head_dim = 64;
    const unsigned int kv_cache_len = 128;
    std::vector<unsigned char> is_done(batch_size, 0);
    std::vector<unsigned int> kv_lens(batch_size);
    for (unsigned int i = 0; i < batch_size; ++i)
        kv_lens[i] = (i % 63) + 1;

    Tensor<unsigned char> is_done_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(is_done_d.data(), is_done.data(),
                          batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;
    std::vector<float> input(batch_size * total_head_num * head_dim);
    for (unsigned int i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i % 31) * 0.03f - 0.4f;

    Tensor<float> input_d({batch_size, total_head_num * head_dim}, GPU);
    CUDA_CHECK(cudaMemcpy(input_d.data(), input.data(),
                          input.size() * sizeof(float), cudaMemcpyHostToDevice));

    Tensor<float> q_d({batch_size, q_head_num, 1, head_dim}, GPU);
    Tensor<float> k_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);
    Tensor<float> v_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);

    launch_fused_reshape_rope_kvcache_decode(input_d, is_done_d, kv_lens_d, q_d, k_d, v_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> q_h({batch_size, q_head_num, 1, head_dim}, CPU);
    Tensor<float> k_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    Tensor<float> v_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(q_h.data(), q_d.data(), q_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_h.data(), k_d.data(), k_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_h.data(), v_d.data(), v_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_q, expected_k, expected_v;
    compute_expected_decode(is_done, kv_lens, kv_cache_len, q_head_num, kv_head_num, head_dim, input,
                            expected_q, expected_k, expected_v);

    for (unsigned int i = 0; i < q_h.numel(); ++i)
        ASSERT_NEAR(q_h.data()[i], expected_q[i], 1e-3f)
            << "q mismatch at idx=" << i;

    for (unsigned int i = 0; i < k_h.numel(); ++i)
        ASSERT_NEAR(k_h.data()[i], expected_k[i], 1e-3f)
            << "k mismatch at idx=" << i;

    for (unsigned int i = 0; i < v_h.numel(); ++i)
        ASSERT_NEAR(v_h.data()[i], expected_v[i], 1e-4f)
            << "v mismatch at idx=" << i;
}

TEST(FusedReshapeRopeKVCacheDecodeTest, LoopOverHeads)
{
    // total_head_num > blockDim.y (16) forces threads to loop over heads.
    const unsigned int batch_size = 2;
    const unsigned int q_head_num = 16;
    const unsigned int kv_head_num = 8;
    const unsigned int head_dim = 64;
    const unsigned int kv_cache_len = 64;
    const std::vector<unsigned char> is_done = {0, 0};
    const std::vector<unsigned int> kv_lens = {5, 10};

    Tensor<unsigned char> is_done_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(is_done_d.data(), is_done.data(),
                          batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;
    std::vector<float> input(batch_size * total_head_num * head_dim);
    for (unsigned int i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i % 19) * 0.07f - 0.6f;

    Tensor<float> input_d({batch_size, total_head_num * head_dim}, GPU);
    CUDA_CHECK(cudaMemcpy(input_d.data(), input.data(),
                          input.size() * sizeof(float), cudaMemcpyHostToDevice));

    Tensor<float> q_d({batch_size, q_head_num, 1, head_dim}, GPU);
    Tensor<float> k_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);
    Tensor<float> v_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);

    launch_fused_reshape_rope_kvcache_decode(input_d, is_done_d, kv_lens_d, q_d, k_d, v_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> q_h({batch_size, q_head_num, 1, head_dim}, CPU);
    Tensor<float> k_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    Tensor<float> v_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(q_h.data(), q_d.data(), q_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_h.data(), k_d.data(), k_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_h.data(), v_d.data(), v_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_q, expected_k, expected_v;
    compute_expected_decode(is_done, kv_lens, kv_cache_len, q_head_num, kv_head_num, head_dim, input,
                            expected_q, expected_k, expected_v);

    for (unsigned int i = 0; i < q_h.numel(); ++i)
        ASSERT_NEAR(q_h.data()[i], expected_q[i], 1e-3f)
            << "q mismatch at idx=" << i;

    for (unsigned int i = 0; i < k_h.numel(); ++i)
        ASSERT_NEAR(k_h.data()[i], expected_k[i], 1e-3f)
            << "k mismatch at idx=" << i;

    for (unsigned int i = 0; i < v_h.numel(); ++i)
        ASSERT_NEAR(v_h.data()[i], expected_v[i], 1e-4f)
            << "v mismatch at idx=" << i;
}
