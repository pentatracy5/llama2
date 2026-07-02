#include <gtest/gtest.h>
#include <kernel/fused_pad_reshape_transpose_rope_kvcache.h>
#include <core/Tensor.cuh>
#include <common/config.h>

#include <cmath>
#include <vector>

// ------------------------------------------------------------------
// Helpers: compute expected results on the host.
// ------------------------------------------------------------------
static void compute_unpad_to_padded_idx(const std::vector<unsigned int> &q_lens,
                                        unsigned int q_cache_len,
                                        std::vector<unsigned int> &unpad_to_padded_idx)
{
    unsigned int num_actual_tokens = 0;
    for (unsigned int len : q_lens)
        num_actual_tokens += len;

    unpad_to_padded_idx.resize(num_actual_tokens);

    unsigned int accumulate_len = 0;
    for (unsigned int seq_id = 0; seq_id < q_lens.size(); ++seq_id)
    {
        for (unsigned int i = 0; i < q_lens[seq_id]; ++i)
        {
            unsigned int unpad_idx = accumulate_len + i;
            unpad_to_padded_idx[unpad_idx] = seq_id * q_cache_len + i;
        }
        accumulate_len += q_lens[seq_id];
    }
}

static void apply_rotation(const std::vector<float> &head,
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

static void compute_expected(const std::vector<unsigned int> &q_lens,
                             const std::vector<unsigned int> &kv_lens,
                             const std::vector<unsigned int> &unpad_to_padded_idx,
                             unsigned int q_cache_len,
                             unsigned int kv_cache_len,
                             unsigned int q_head_num,
                             unsigned int kv_head_num,
                             unsigned int head_dim,
                             const std::vector<float> &input,
                             std::vector<float> &expected_q,
                             std::vector<float> &expected_k,
                             std::vector<float> &expected_v)
{
    const unsigned int batch_size = static_cast<unsigned int>(q_lens.size());
    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;

    expected_q.assign(batch_size * q_head_num * q_cache_len * head_dim, 0.f);
    expected_k.assign(batch_size * kv_head_num * kv_cache_len * head_dim, 0.f);
    expected_v.assign(batch_size * kv_head_num * kv_cache_len * head_dim, 0.f);

    for (unsigned int idx = 0; idx < unpad_to_padded_idx.size(); ++idx)
    {
        unsigned int padded_idx = unpad_to_padded_idx[idx];
        unsigned int seq_idx = padded_idx / q_cache_len;
        unsigned int token_idx = padded_idx - q_cache_len * seq_idx;
        unsigned int kv_len = kv_lens[seq_idx];
        unsigned int history_len = kv_len - q_lens[seq_idx];

        for (unsigned int h = 0; h < q_head_num; ++h)
        {
            unsigned int head_offset = (idx * total_head_num + h) * head_dim;
            std::vector<float> head(input.begin() + head_offset,
                                    input.begin() + head_offset + head_dim);
            std::vector<float> rotated;
            apply_rotation(head, head_dim, token_idx + history_len, kv_len, rotated);

            unsigned int q_offset = ((seq_idx * q_head_num + h) * q_cache_len + token_idx) * head_dim;
            for (unsigned int d = 0; d < head_dim; ++d)
                expected_q[q_offset + d] = rotated[d];
        }

        for (unsigned int h = 0; h < kv_head_num; ++h)
        {
            unsigned int k_head_offset = (idx * total_head_num + q_head_num + h) * head_dim;
            std::vector<float> k_head(input.begin() + k_head_offset,
                                      input.begin() + k_head_offset + head_dim);
            std::vector<float> rotated_k;
            apply_rotation(k_head, head_dim, token_idx + history_len, kv_len, rotated_k);

            unsigned int k_offset = ((seq_idx * kv_head_num + h) * kv_cache_len +
                                     token_idx + history_len) *
                                    head_dim;
            for (unsigned int d = 0; d < head_dim; ++d)
                expected_k[k_offset + d] = rotated_k[d];
        }

        for (unsigned int h = 0; h < kv_head_num; ++h)
        {
            unsigned int v_head_offset = (idx * total_head_num + q_head_num + kv_head_num + h) * head_dim;

            unsigned int v_offset = ((seq_idx * kv_head_num + h) * kv_cache_len +
                                     token_idx + history_len) *
                                    head_dim;
            for (unsigned int d = 0; d < head_dim; ++d)
                expected_v[v_offset + d] = input[v_head_offset + d];
        }
    }
}

// ------------------------------------------------------------------
// fused_pad_reshape_transpose_rope_kvcache kernel tests
// ------------------------------------------------------------------
TEST(FusedPadReshapeTransposeRopeKVCacheTest, BasicTwoSequences)
{
    const unsigned int batch_size = 2;
    const unsigned int q_head_num = 2;
    const unsigned int kv_head_num = 1;
    const unsigned int head_dim = 8;
    const unsigned int q_cache_len = 16;
    const unsigned int kv_cache_len = 32;
    const std::vector<unsigned int> q_lens = {2, 3};
    const std::vector<unsigned int> kv_lens = {3, 6};
    const unsigned int num_actual_tokens = 5;

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    std::vector<unsigned int> unpad_to_padded_idx;
    compute_unpad_to_padded_idx(q_lens, q_cache_len, unpad_to_padded_idx);

    Tensor<unsigned int> unpad_to_padded_idx_d({num_actual_tokens}, GPU);
    CUDA_CHECK(cudaMemcpy(unpad_to_padded_idx_d.data(), unpad_to_padded_idx.data(),
                          num_actual_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;
    std::vector<float> input(num_actual_tokens * total_head_num * head_dim);
    for (unsigned int i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i % 17) * 0.1f - 0.8f;

    Tensor<float> input_d({num_actual_tokens, total_head_num * head_dim}, GPU);
    CUDA_CHECK(cudaMemcpy(input_d.data(), input.data(),
                          input.size() * sizeof(float), cudaMemcpyHostToDevice));

    Tensor<float> q_d({batch_size, q_head_num, q_cache_len, head_dim}, GPU);
    Tensor<float> k_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);
    Tensor<float> v_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);

    launch_fused_pad_reshape_transpose_rope_kvcache(input_d, q_lens_d, kv_lens_d,
                                                    unpad_to_padded_idx_d, q_d, k_d, v_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> q_h({batch_size, q_head_num, q_cache_len, head_dim}, CPU);
    Tensor<float> k_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    Tensor<float> v_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(q_h.data(), q_d.data(), q_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_h.data(), k_d.data(), k_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_h.data(), v_d.data(), v_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_q, expected_k, expected_v;
    compute_expected(q_lens, kv_lens, unpad_to_padded_idx, q_cache_len, kv_cache_len,
                     q_head_num, kv_head_num, head_dim, input,
                     expected_q, expected_k, expected_v);

    for (unsigned int i = 0; i < q_h.numel(); ++i)
        ASSERT_NEAR(q_h.data()[i], expected_q[i], 1e-4f)
            << "q mismatch at idx=" << i;

    for (unsigned int i = 0; i < k_h.numel(); ++i)
        ASSERT_NEAR(k_h.data()[i], expected_k[i], 1e-4f)
            << "k mismatch at idx=" << i;

    for (unsigned int i = 0; i < v_h.numel(); ++i)
        ASSERT_NEAR(v_h.data()[i], expected_v[i], 1e-4f)
            << "v mismatch at idx=" << i;
}

TEST(FusedPadReshapeTransposeRopeKVCacheTest, SingleSequenceFirstForward)
{
    const unsigned int batch_size = 1;
    const unsigned int q_head_num = 2;
    const unsigned int kv_head_num = 2;
    const unsigned int head_dim = 4;
    const unsigned int q_cache_len = 8;
    const unsigned int kv_cache_len = 16;
    const std::vector<unsigned int> q_lens = {3};
    const std::vector<unsigned int> kv_lens = {3};
    const unsigned int num_actual_tokens = 3;

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    std::vector<unsigned int> unpad_to_padded_idx;
    compute_unpad_to_padded_idx(q_lens, q_cache_len, unpad_to_padded_idx);

    Tensor<unsigned int> unpad_to_padded_idx_d({num_actual_tokens}, GPU);
    CUDA_CHECK(cudaMemcpy(unpad_to_padded_idx_d.data(), unpad_to_padded_idx.data(),
                          num_actual_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;
    std::vector<float> input(num_actual_tokens * total_head_num * head_dim);
    for (unsigned int i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i) * 0.05f - 0.5f;

    Tensor<float> input_d({num_actual_tokens, total_head_num * head_dim}, GPU);
    CUDA_CHECK(cudaMemcpy(input_d.data(), input.data(),
                          input.size() * sizeof(float), cudaMemcpyHostToDevice));

    Tensor<float> q_d({batch_size, q_head_num, q_cache_len, head_dim}, GPU);
    Tensor<float> k_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);
    Tensor<float> v_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);

    launch_fused_pad_reshape_transpose_rope_kvcache(input_d, q_lens_d, kv_lens_d,
                                                    unpad_to_padded_idx_d, q_d, k_d, v_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> q_h({batch_size, q_head_num, q_cache_len, head_dim}, CPU);
    Tensor<float> k_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    Tensor<float> v_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(q_h.data(), q_d.data(), q_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_h.data(), k_d.data(), k_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_h.data(), v_d.data(), v_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_q, expected_k, expected_v;
    compute_expected(q_lens, kv_lens, unpad_to_padded_idx, q_cache_len, kv_cache_len,
                     q_head_num, kv_head_num, head_dim, input,
                     expected_q, expected_k, expected_v);

    for (unsigned int i = 0; i < q_h.numel(); ++i)
        ASSERT_NEAR(q_h.data()[i], expected_q[i], 1e-4f)
            << "q mismatch at idx=" << i;

    for (unsigned int i = 0; i < k_h.numel(); ++i)
        ASSERT_NEAR(k_h.data()[i], expected_k[i], 1e-4f)
            << "k mismatch at idx=" << i;

    for (unsigned int i = 0; i < v_h.numel(); ++i)
        ASSERT_NEAR(v_h.data()[i], expected_v[i], 1e-4f)
            << "v mismatch at idx=" << i;
}

TEST(FusedPadReshapeTransposeRopeKVCacheTest, IncrementalDecoding)
{
    // Each sequence contributes only one new query token, while the KV cache already
    // contains history. This mirrors the autoregressive decoding phase.
    const unsigned int batch_size = 3;
    const unsigned int q_head_num = 4;
    const unsigned int kv_head_num = 2;
    const unsigned int head_dim = 8;
    const unsigned int q_cache_len = 8;
    const unsigned int kv_cache_len = 32;
    const std::vector<unsigned int> q_lens = {1, 1, 1};
    const std::vector<unsigned int> kv_lens = {4, 9, 15};
    const unsigned int num_actual_tokens = 3;

    Tensor<unsigned int> q_lens_d({batch_size}, GPU);
    Tensor<unsigned int> kv_lens_d({batch_size}, GPU);
    CUDA_CHECK(cudaMemcpy(q_lens_d.data(), q_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_lens_d.data(), kv_lens.data(),
                          batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    std::vector<unsigned int> unpad_to_padded_idx;
    compute_unpad_to_padded_idx(q_lens, q_cache_len, unpad_to_padded_idx);

    Tensor<unsigned int> unpad_to_padded_idx_d({num_actual_tokens}, GPU);
    CUDA_CHECK(cudaMemcpy(unpad_to_padded_idx_d.data(), unpad_to_padded_idx.data(),
                          num_actual_tokens * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const unsigned int total_head_num = q_head_num + 2 * kv_head_num;
    std::vector<float> input(num_actual_tokens * total_head_num * head_dim);
    for (unsigned int i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i % 23) * 0.05f - 0.5f;

    Tensor<float> input_d({num_actual_tokens, total_head_num * head_dim}, GPU);
    CUDA_CHECK(cudaMemcpy(input_d.data(), input.data(),
                          input.size() * sizeof(float), cudaMemcpyHostToDevice));

    Tensor<float> q_d({batch_size, q_head_num, q_cache_len, head_dim}, GPU);
    Tensor<float> k_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);
    Tensor<float> v_d({batch_size, kv_head_num, kv_cache_len, head_dim}, GPU);

    launch_fused_pad_reshape_transpose_rope_kvcache(input_d, q_lens_d, kv_lens_d,
                                                    unpad_to_padded_idx_d, q_d, k_d, v_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> q_h({batch_size, q_head_num, q_cache_len, head_dim}, CPU);
    Tensor<float> k_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    Tensor<float> v_h({batch_size, kv_head_num, kv_cache_len, head_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(q_h.data(), q_d.data(), q_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_h.data(), k_d.data(), k_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_h.data(), v_d.data(), v_h.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_q, expected_k, expected_v;
    compute_expected(q_lens, kv_lens, unpad_to_padded_idx, q_cache_len, kv_cache_len,
                     q_head_num, kv_head_num, head_dim, input,
                     expected_q, expected_k, expected_v);

    for (unsigned int i = 0; i < q_h.numel(); ++i)
        ASSERT_NEAR(q_h.data()[i], expected_q[i], 1e-4f)
            << "q mismatch at idx=" << i;

    for (unsigned int i = 0; i < k_h.numel(); ++i)
        ASSERT_NEAR(k_h.data()[i], expected_k[i], 1e-4f)
            << "k mismatch at idx=" << i;

    for (unsigned int i = 0; i < v_h.numel(); ++i)
        ASSERT_NEAR(v_h.data()[i], expected_v[i], 1e-4f)
            << "v mismatch at idx=" << i;
}
