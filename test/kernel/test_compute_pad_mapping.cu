#include <gtest/gtest.h>
#include <kernel/compute_pad_mapping.h>
#include <core/Tensor.cuh>

#include <vector>

// ------------------------------------------------------------------
// Helper: compute expected maps on the host.
// ------------------------------------------------------------------
static void compute_expected_maps(const std::vector<int> &seq_lens,
                                  unsigned int max_seq_len,
                                  std::vector<int> &unpad2pad_map,
                                  std::vector<int> &pad2unpad_map)
{
    unsigned int num_seqs = static_cast<unsigned int>(seq_lens.size());
    unsigned int num_input_ids = 0;
    for (int len : seq_lens)
        num_input_ids += static_cast<unsigned int>(len);

    unpad2pad_map.resize(num_input_ids);
    pad2unpad_map.resize(num_seqs);

    unsigned int accumulate_len = 0;
    for (unsigned int seq_id = 0; seq_id < num_seqs; ++seq_id)
    {
        pad2unpad_map[seq_id] = accumulate_len;
        for (int i = 0; i < seq_lens[seq_id]; ++i)
        {
            unsigned int unpad_idx = accumulate_len + static_cast<unsigned int>(i);
            unpad2pad_map[unpad_idx] = static_cast<int>(seq_id * max_seq_len + i);
        }
        accumulate_len += static_cast<unsigned int>(seq_lens[seq_id]);
    }
}

// ------------------------------------------------------------------
// ComputePadMapping kernel tests
// ------------------------------------------------------------------
TEST(ComputePadMappingTest, BasicTwoSequences)
{
    const unsigned int max_seq_len = 5;
    const std::vector<int> seq_lens = {2, 3};
    const unsigned int num_input_ids = 5;
    const unsigned int num_seqs = 2;

    Tensor<int> seq_lens_d({num_seqs}, GPU);
    CUDA_CHECK(cudaMemcpy(seq_lens_d.data(), seq_lens.data(),
                          num_seqs * sizeof(int), cudaMemcpyHostToDevice));

    Tensor<int> pad2unpad_d({num_seqs}, GPU);
    Tensor<int> unpad2pad_d({num_input_ids}, GPU);

    ComputePadMappingLauncher(num_input_ids, max_seq_len, seq_lens_d,
                              pad2unpad_d, unpad2pad_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<int> pad2unpad_h({num_seqs}, CPU);
    Tensor<int> unpad2pad_h({num_input_ids}, CPU);
    CUDA_CHECK(cudaMemcpy(pad2unpad_h.data(), pad2unpad_d.data(),
                          num_seqs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(unpad2pad_h.data(), unpad2pad_d.data(),
                          num_input_ids * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> expected_unpad2pad, expected_pad2unpad;
    compute_expected_maps(seq_lens, max_seq_len,
                          expected_unpad2pad, expected_pad2unpad);

    for (unsigned int i = 0; i < num_input_ids; ++i)
        ASSERT_EQ(unpad2pad_h.data()[i], expected_unpad2pad[i])
            << "unpad2pad mismatch at idx=" << i;

    for (unsigned int i = 0; i < num_seqs; ++i)
        ASSERT_EQ(pad2unpad_h.data()[i], expected_pad2unpad[i])
            << "pad2unpad mismatch at seq_id=" << i;
}

TEST(ComputePadMappingTest, SingleTokenSingleSequence)
{
    const unsigned int max_seq_len = 4;
    const std::vector<int> seq_lens = {1};
    const unsigned int num_input_ids = 1;
    const unsigned int num_seqs = 1;

    Tensor<int> seq_lens_d({num_seqs}, GPU);
    CUDA_CHECK(cudaMemcpy(seq_lens_d.data(), seq_lens.data(),
                          num_seqs * sizeof(int), cudaMemcpyHostToDevice));

    Tensor<int> pad2unpad_d({num_seqs}, GPU);
    Tensor<int> unpad2pad_d({num_input_ids}, GPU);

    ComputePadMappingLauncher(num_input_ids, max_seq_len, seq_lens_d,
                              pad2unpad_d, unpad2pad_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<int> pad2unpad_h({num_seqs}, CPU);
    Tensor<int> unpad2pad_h({num_input_ids}, CPU);
    CUDA_CHECK(cudaMemcpy(pad2unpad_h.data(), pad2unpad_d.data(),
                          num_seqs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(unpad2pad_h.data(), unpad2pad_d.data(),
                          num_input_ids * sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(unpad2pad_h.data()[0], 0);
    ASSERT_EQ(pad2unpad_h.data()[0], 0);
}

TEST(ComputePadMappingTest, UnequalLengths)
{
    const unsigned int max_seq_len = 8;
    const std::vector<int> seq_lens = {1, 4, 2, 3};
    const unsigned int num_input_ids = 10;
    const unsigned int num_seqs = 4;

    Tensor<int> seq_lens_d({num_seqs}, GPU);
    CUDA_CHECK(cudaMemcpy(seq_lens_d.data(), seq_lens.data(),
                          num_seqs * sizeof(int), cudaMemcpyHostToDevice));

    Tensor<int> pad2unpad_d({num_seqs}, GPU);
    Tensor<int> unpad2pad_d({num_input_ids}, GPU);

    ComputePadMappingLauncher(num_input_ids, max_seq_len, seq_lens_d,
                              pad2unpad_d, unpad2pad_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<int> pad2unpad_h({num_seqs}, CPU);
    Tensor<int> unpad2pad_h({num_input_ids}, CPU);
    CUDA_CHECK(cudaMemcpy(pad2unpad_h.data(), pad2unpad_d.data(),
                          num_seqs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(unpad2pad_h.data(), unpad2pad_d.data(),
                          num_input_ids * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> expected_unpad2pad, expected_pad2unpad;
    compute_expected_maps(seq_lens, max_seq_len,
                          expected_unpad2pad, expected_pad2unpad);

    for (unsigned int i = 0; i < num_input_ids; ++i)
        ASSERT_EQ(unpad2pad_h.data()[i], expected_unpad2pad[i])
            << "unpad2pad mismatch at idx=" << i;

    for (unsigned int i = 0; i < num_seqs; ++i)
        ASSERT_EQ(pad2unpad_h.data()[i], expected_pad2unpad[i])
            << "pad2unpad mismatch at seq_id=" << i;
}

TEST(ComputePadMappingTest, LargeBatch)
{
    const unsigned int max_seq_len = 16;
    const unsigned int num_seqs = 32;
    std::vector<int> seq_lens(num_seqs);
    unsigned int num_input_ids = 0;
    for (unsigned int i = 0; i < num_seqs; ++i)
    {
        seq_lens[i] = static_cast<int>((i % max_seq_len) + 1);
        num_input_ids += static_cast<unsigned int>(seq_lens[i]);
    }

    Tensor<int> seq_lens_d({num_seqs}, GPU);
    CUDA_CHECK(cudaMemcpy(seq_lens_d.data(), seq_lens.data(),
                          num_seqs * sizeof(int), cudaMemcpyHostToDevice));

    Tensor<int> pad2unpad_d({num_seqs}, GPU);
    Tensor<int> unpad2pad_d({num_input_ids}, GPU);

    ComputePadMappingLauncher(num_input_ids, max_seq_len, seq_lens_d,
                              pad2unpad_d, unpad2pad_d);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<int> pad2unpad_h({num_seqs}, CPU);
    Tensor<int> unpad2pad_h({num_input_ids}, CPU);
    CUDA_CHECK(cudaMemcpy(pad2unpad_h.data(), pad2unpad_d.data(),
                          num_seqs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(unpad2pad_h.data(), unpad2pad_d.data(),
                          num_input_ids * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> expected_unpad2pad, expected_pad2unpad;
    compute_expected_maps(seq_lens, max_seq_len,
                          expected_unpad2pad, expected_pad2unpad);

    for (unsigned int i = 0; i < num_input_ids; ++i)
        ASSERT_EQ(unpad2pad_h.data()[i], expected_unpad2pad[i])
            << "unpad2pad mismatch at idx=" << i;

    for (unsigned int i = 0; i < num_seqs; ++i)
        ASSERT_EQ(pad2unpad_h.data()[i], expected_pad2unpad[i])
            << "pad2unpad mismatch at seq_id=" << i;
}
