#include <gtest/gtest.h>
#include <kernel/build_padding_index_maps.h>
#include <core/Tensor.cuh>

#include <vector>

// ------------------------------------------------------------------
// Helper: compute expected maps on the host.
// ------------------------------------------------------------------
static void compute_expected_maps(const std::vector<unsigned int> &seq_lens,
                                  unsigned int max_seq_len,
                                  std::vector<unsigned int> &unpad_to_pad_idx,
                                  std::vector<unsigned int> &seq_offsets)
{
    unsigned int num_seqs = static_cast<unsigned int>(seq_lens.size());
    unsigned int num_input_tokens = 0;
    for (unsigned int len : seq_lens)
        num_input_tokens += static_cast<unsigned int>(len);

    unpad_to_pad_idx.resize(num_input_tokens);
    seq_offsets.resize(num_seqs);

    unsigned int accumulate_len = 0;
    for (unsigned int seq_id = 0; seq_id < num_seqs; ++seq_id)
    {
        seq_offsets[seq_id] = accumulate_len;
        for (unsigned int i = 0; i < seq_lens[seq_id]; ++i)
        {
            unsigned int unpad_idx = accumulate_len + static_cast<unsigned int>(i);
            unpad_to_pad_idx[unpad_idx] = static_cast<unsigned int>(seq_id * max_seq_len + i);
        }
        accumulate_len += static_cast<unsigned int>(seq_lens[seq_id]);
    }
}

// ------------------------------------------------------------------
// build_padding_index_maps kernel tests
// ------------------------------------------------------------------
TEST(BuildPaddingIndexMapsTest, BasicTwoSequences)
{
    const unsigned int max_seq_len = 5;
    const std::vector<unsigned int> seq_lens = {2, 3};
    const unsigned int num_input_tokens = 5;
    const unsigned int num_seqs = 2;

    Tensor<unsigned int> seq_lens_d({num_seqs}, GPU);
    CUDA_CHECK(cudaMemcpy(seq_lens_d.data(), seq_lens.data(),
                          num_seqs * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> seq_offsets_d({num_seqs}, GPU);
    Tensor<unsigned int> unpad_to_pad_idx_d({num_seqs, max_seq_len}, GPU);

    Tensor<unsigned int> input_ids({num_input_tokens}, GPU);

    launch_build_padding_index_maps(input_ids, seq_lens_d,
                                    unpad_to_pad_idx_d, seq_offsets_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned int> seq_offsets_h({num_seqs}, CPU);
    Tensor<unsigned int> unpad_to_pad_idx_h({num_input_tokens}, CPU);
    CUDA_CHECK(cudaMemcpy(seq_offsets_h.data(), seq_offsets_d.data(),
                          num_seqs * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(unpad_to_pad_idx_h.data(), unpad_to_pad_idx_d.data(),
                          num_input_tokens * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> expected_unpad_to_pad_idx, expected_seq_offsets;
    compute_expected_maps(seq_lens, max_seq_len,
                          expected_unpad_to_pad_idx, expected_seq_offsets);

    for (unsigned int i = 0; i < num_input_tokens; ++i)
        ASSERT_EQ(unpad_to_pad_idx_h.data()[i], expected_unpad_to_pad_idx[i])
            << "unpad_to_pad_idx mismatch at idx=" << i;

    for (unsigned int i = 0; i < num_seqs; ++i)
        ASSERT_EQ(seq_offsets_h.data()[i], expected_seq_offsets[i])
            << "seq_offsets mismatch at seq_id=" << i;
}

TEST(BuildPaddingIndexMapsTest, SingleTokenSingleSequence)
{
    const unsigned int max_seq_len = 4;
    const std::vector<unsigned int> seq_lens = {1};
    const unsigned int num_input_tokens = 1;
    const unsigned int num_seqs = 1;

    Tensor<unsigned int> seq_lens_d({num_seqs}, GPU);
    CUDA_CHECK(cudaMemcpy(seq_lens_d.data(), seq_lens.data(),
                          num_seqs * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> seq_offsets_d({num_seqs}, GPU);
    Tensor<unsigned int> unpad_to_pad_idx_d({num_seqs, max_seq_len}, GPU);

    Tensor<unsigned int> input_ids({num_input_tokens}, GPU);

    launch_build_padding_index_maps(input_ids, seq_lens_d,
                                    unpad_to_pad_idx_d, seq_offsets_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned int> seq_offsets_h({num_seqs}, CPU);
    Tensor<unsigned int> unpad_to_pad_idx_h({num_input_tokens}, CPU);
    CUDA_CHECK(cudaMemcpy(seq_offsets_h.data(), seq_offsets_d.data(),
                          num_seqs * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(unpad_to_pad_idx_h.data(), unpad_to_pad_idx_d.data(),
                          num_input_tokens * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(unpad_to_pad_idx_h.data()[0], 0);
    ASSERT_EQ(seq_offsets_h.data()[0], 0);
}

TEST(BuildPaddingIndexMapsTest, UnequalLengths)
{
    const unsigned int max_seq_len = 8;
    const std::vector<unsigned int> seq_lens = {1, 4, 2, 3};
    const unsigned int num_input_tokens = 10;
    const unsigned int num_seqs = 4;

    Tensor<unsigned int> seq_lens_d({num_seqs}, GPU);
    CUDA_CHECK(cudaMemcpy(seq_lens_d.data(), seq_lens.data(),
                          num_seqs * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> seq_offsets_d({num_seqs}, GPU);
    Tensor<unsigned int> unpad_to_pad_idx_d({num_seqs, max_seq_len}, GPU);

    Tensor<unsigned int> input_ids({num_input_tokens}, GPU);

    launch_build_padding_index_maps(input_ids, seq_lens_d,
                                    unpad_to_pad_idx_d, seq_offsets_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned int> seq_offsets_h({num_seqs}, CPU);
    Tensor<unsigned int> unpad_to_pad_idx_h({num_input_tokens}, CPU);
    CUDA_CHECK(cudaMemcpy(seq_offsets_h.data(), seq_offsets_d.data(),
                          num_seqs * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(unpad_to_pad_idx_h.data(), unpad_to_pad_idx_d.data(),
                          num_input_tokens * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> expected_unpad_to_pad_idx, expected_seq_offsets;
    compute_expected_maps(seq_lens, max_seq_len,
                          expected_unpad_to_pad_idx, expected_seq_offsets);

    for (unsigned int i = 0; i < num_input_tokens; ++i)
        ASSERT_EQ(unpad_to_pad_idx_h.data()[i], expected_unpad_to_pad_idx[i])
            << "unpad_to_pad_idx mismatch at idx=" << i;

    for (unsigned int i = 0; i < num_seqs; ++i)
        ASSERT_EQ(seq_offsets_h.data()[i], expected_seq_offsets[i])
            << "seq_offsets mismatch at seq_id=" << i;
}

TEST(BuildPaddingIndexMapsTest, LargeBatch)
{
    const unsigned int max_seq_len = 16;
    const unsigned int num_seqs = 32;
    std::vector<unsigned int> seq_lens(num_seqs);
    unsigned int num_input_tokens = 0;
    for (unsigned int i = 0; i < num_seqs; ++i)
    {
        seq_lens[i] = static_cast<unsigned int>((i % max_seq_len) + 1);
        num_input_tokens += static_cast<unsigned int>(seq_lens[i]);
    }

    Tensor<unsigned int> seq_lens_d({num_seqs}, GPU);
    CUDA_CHECK(cudaMemcpy(seq_lens_d.data(), seq_lens.data(),
                          num_seqs * sizeof(unsigned int), cudaMemcpyHostToDevice));

    Tensor<unsigned int> seq_offsets_d({num_seqs}, GPU);
    Tensor<unsigned int> unpad_to_pad_idx_d({num_seqs, max_seq_len}, GPU);

    Tensor<unsigned int> input_ids({num_input_tokens}, GPU);

    launch_build_padding_index_maps(input_ids, seq_lens_d,
                                    unpad_to_pad_idx_d, seq_offsets_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<unsigned int> seq_offsets_h({num_seqs}, CPU);
    Tensor<unsigned int> unpad_to_pad_idx_h({num_input_tokens}, CPU);
    CUDA_CHECK(cudaMemcpy(seq_offsets_h.data(), seq_offsets_d.data(),
                          num_seqs * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(unpad_to_pad_idx_h.data(), unpad_to_pad_idx_d.data(),
                          num_input_tokens * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> expected_unpad_to_pad_idx, expected_seq_offsets;
    compute_expected_maps(seq_lens, max_seq_len,
                          expected_unpad_to_pad_idx, expected_seq_offsets);

    for (unsigned int i = 0; i < num_input_tokens; ++i)
        ASSERT_EQ(unpad_to_pad_idx_h.data()[i], expected_unpad_to_pad_idx[i])
            << "unpad_to_pad_idx mismatch at idx=" << i;

    for (unsigned int i = 0; i < num_seqs; ++i)
        ASSERT_EQ(seq_offsets_h.data()[i], expected_seq_offsets[i])
            << "seq_offsets mismatch at seq_id=" << i;
}
