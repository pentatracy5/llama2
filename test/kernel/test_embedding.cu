#include <gtest/gtest.h>
#include <kernel/embedding.h>
#include <core/Tensor.cuh>

#include <vector>
#include <cstring>

// ------------------------------------------------------------------
// Helper: fill a 1-D buffer on the host with sequential values.
// ------------------------------------------------------------------
template <typename T>
void fill_sequential(T *data, size_t n, T start = T(0))
{
    for (size_t i = 0; i < n; ++i)
        data[i] = start + static_cast<T>(i);
}

// ------------------------------------------------------------------
// Embedding kernel tests
// ------------------------------------------------------------------
TEST(EmbeddingTest, BasicLookup)
{
    const unsigned int vocab_size = 4;
    const unsigned int embed_dim = 8;
    const unsigned int token_num = 4;

    // Host buffers
    std::vector<unsigned int> input_ids_h = {0, 2, 1, 3};
    std::vector<float> table_h(vocab_size * embed_dim);
    fill_sequential(table_h.data(), table_h.size(), 0.1f);

    // Device tensors
    Tensor<unsigned int> input_ids({token_num}, GPU);
    Tensor<float> embed_table({vocab_size, embed_dim}, GPU);
    Tensor<float> output({token_num, embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(input_ids.data(), input_ids_h.data(),
                          token_num * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embed_table.data(), table_h.data(),
                          table_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_embedding(input_ids, embed_table, output);
    CUDA_KERNEL_LAUNCH_CHECK();

    // Copy result back to host
    Tensor<float> output_h({token_num, embed_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output.data(),
                          output.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify each output row equals the corresponding embedding table row
    for (unsigned int t = 0; t < token_num; ++t)
    {
        unsigned int id = input_ids_h[t];
        for (unsigned int d = 0; d < embed_dim; ++d)
        {
            float expected = table_h[id * embed_dim + d];
            float actual = output_h.data()[t * embed_dim + d];
            ASSERT_EQ(actual, expected)
                << "token=" << t << ", dim=" << d;
        }
    }
}

TEST(EmbeddingTest, SingleTokenFourDim)
{
    const unsigned int vocab_size = 3;
    const unsigned int embed_dim = 4;
    const unsigned int token_num = 1;

    std::vector<unsigned int> input_ids_h = {2};
    std::vector<float> table_h = {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f};

    Tensor<unsigned int> input_ids({token_num}, GPU);
    Tensor<float> embed_table({vocab_size, embed_dim}, GPU);
    Tensor<float> output({token_num, embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(input_ids.data(), input_ids_h.data(),
                          token_num * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embed_table.data(), table_h.data(),
                          table_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_embedding(input_ids, embed_table, output);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> output_h({token_num, embed_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output.data(),
                          output.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    for (unsigned int d = 0; d < embed_dim; ++d)
    {
        ASSERT_EQ(output_h.data()[d], table_h[2 * embed_dim + d]);
    }
}

TEST(EmbeddingTest, RepeatedIndex)
{
    const unsigned int vocab_size = 5;
    const unsigned int embed_dim = 4;
    const unsigned int token_num = 6;

    std::vector<unsigned int> input_ids_h = {1, 1, 3, 3, 1, 3};
    std::vector<float> table_h(vocab_size * embed_dim);
    fill_sequential(table_h.data(), table_h.size(), 0.0f);

    Tensor<unsigned int> input_ids({token_num}, GPU);
    Tensor<float> embed_table({vocab_size, embed_dim}, GPU);
    Tensor<float> output({token_num, embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(input_ids.data(), input_ids_h.data(),
                          token_num * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embed_table.data(), table_h.data(),
                          table_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_embedding(input_ids, embed_table, output);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> output_h({token_num, embed_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output.data(),
                          output.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    for (unsigned int t = 0; t < token_num; ++t)
    {
        unsigned int id = input_ids_h[t];
        for (unsigned int d = 0; d < embed_dim; ++d)
        {
            ASSERT_EQ(output_h.data()[t * embed_dim + d],
                      table_h[id * embed_dim + d])
                << "token=" << t << ", dim=" << d;
        }
    }
}

TEST(EmbeddingTest, MultiTokenBatch)
{
    const unsigned int vocab_size = 10;
    const unsigned int embed_dim = 16;
    const unsigned int token_num = 32;

    std::vector<unsigned int> input_ids_h(token_num);
    for (unsigned int i = 0; i < token_num; ++i)
        input_ids_h[i] = static_cast<unsigned int>(i % vocab_size);

    std::vector<float> table_h(vocab_size * embed_dim);
    fill_sequential(table_h.data(), table_h.size(), 0.5f);

    Tensor<unsigned int> input_ids({token_num}, GPU);
    Tensor<float> embed_table({vocab_size, embed_dim}, GPU);
    Tensor<float> output({token_num, embed_dim}, GPU);

    CUDA_CHECK(cudaMemcpy(input_ids.data(), input_ids_h.data(),
                          token_num * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embed_table.data(), table_h.data(),
                          table_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_embedding(input_ids, embed_table, output);
    CUDA_KERNEL_LAUNCH_CHECK();

    Tensor<float> output_h({token_num, embed_dim}, CPU);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output.data(),
                          output.numel() * sizeof(float), cudaMemcpyDeviceToHost));

    for (unsigned int t = 0; t < token_num; ++t)
    {
        unsigned int id = input_ids_h[t];
        for (unsigned int d = 0; d < embed_dim; ++d)
        {
            ASSERT_EQ(output_h.data()[t * embed_dim + d],
                      table_h[id * embed_dim + d])
                << "token=" << t << ", dim=" << d;
        }
    }
}
