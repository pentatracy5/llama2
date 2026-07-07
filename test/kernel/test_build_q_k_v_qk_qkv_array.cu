#include <gtest/gtest.h>
#include <kernel/build_q_k_v_qk_qkv_array.h>
#include <core/Tensor.cuh>

#include <cuda_runtime.h>
#include <cassert>
#include <vector>

// ------------------------------------------------------------------
// Helper: verify pointer arrays for one configuration.
//
// Tensors are 4-D row-major:
//   q/qkv: (batch_size, q_head_num, q_cache_len, head_size)
//   k/v  : (batch_size, kv_head_num, kv_cache_len, head_size)
//   qk   : (batch_size, q_head_num, q_cache_len, kv_cache_len)
//
// For each (batch, q_head) we store one 2-D matrix pointer.  k/v are
// broadcast according to the GQA grouping:
//   kv_head = q_head / (q_head_num / kv_head_num)
// ------------------------------------------------------------------
template <typename T>
static void run_pointer_array_test(unsigned int batch_size,
                                   unsigned int q_head_num,
                                   unsigned int kv_head_num,
                                   unsigned int q_cache_len,
                                   unsigned int kv_cache_len,
                                   unsigned int head_size)
{
    assert(q_head_num % kv_head_num == 0);
    const unsigned int q_heads_per_kv_head = q_head_num / kv_head_num;

    Tensor<T> q({batch_size, q_head_num, q_cache_len, head_size}, GPU);
    Tensor<T> k({batch_size, kv_head_num, kv_cache_len, head_size}, GPU);
    Tensor<T> v({batch_size, kv_head_num, kv_cache_len, head_size}, GPU);
    Tensor<T> qk({batch_size, q_head_num, q_cache_len, kv_cache_len}, GPU);
    Tensor<T> qkv({batch_size, q_head_num, q_cache_len, head_size}, GPU);

    const unsigned int array_len = batch_size * q_head_num;
    Tensor<void *> q_array({array_len}, GPU);
    Tensor<void *> k_array({array_len}, GPU);
    Tensor<void *> v_array({array_len}, GPU);
    Tensor<void *> qk_array({array_len}, GPU);
    Tensor<void *> qkv_array({array_len}, GPU);

    launch_build_q_k_v_qk_qkv_array(q, k, v, qk, qkv,
                                    q_array, k_array, v_array, qk_array, qkv_array);
    CUDA_KERNEL_LAUNCH_CHECK();

    // Copy pointer arrays back to host for verification.
    Tensor<void *> q_array_h({array_len}, CPU);
    Tensor<void *> k_array_h({array_len}, CPU);
    Tensor<void *> v_array_h({array_len}, CPU);
    Tensor<void *> qk_array_h({array_len}, CPU);
    Tensor<void *> qkv_array_h({array_len}, CPU);

    CUDA_CHECK(cudaMemcpy(q_array_h.data(), q_array.data(),
                          array_len * sizeof(void *), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_array_h.data(), k_array.data(),
                          array_len * sizeof(void *), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_array_h.data(), v_array.data(),
                          array_len * sizeof(void *), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qk_array_h.data(), qk_array.data(),
                          array_len * sizeof(void *), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qkv_array_h.data(), qkv_array.data(),
                          array_len * sizeof(void *), cudaMemcpyDeviceToHost));

    const unsigned int q_stride = q_cache_len * head_size;
    const unsigned int kv_stride = kv_cache_len * head_size;
    const unsigned int qk_stride = q_cache_len * kv_cache_len;
    const unsigned int qkv_stride = q_cache_len * head_size;

    for (unsigned int b = 0; b < batch_size; ++b)
    {
        for (unsigned int qh = 0; qh < q_head_num; ++qh)
        {
            const unsigned int idx = b * q_head_num + qh;
            const unsigned int kv_head = qh / q_heads_per_kv_head;
            const unsigned int kv_idx = b * kv_head_num + kv_head;

            void *expected_q = static_cast<void *>(q.data() + idx * q_stride);
            void *expected_k = static_cast<void *>(k.data() + kv_idx * kv_stride);
            void *expected_v = static_cast<void *>(v.data() + kv_idx * kv_stride);
            void *expected_qk = static_cast<void *>(qk.data() + idx * qk_stride);
            void *expected_qkv = static_cast<void *>(qkv.data() + idx * qkv_stride);

            ASSERT_EQ(q_array_h.data()[idx], expected_q)
                << "q pointer mismatch at batch=" << b << ", q_head=" << qh;
            ASSERT_EQ(k_array_h.data()[idx], expected_k)
                << "k pointer mismatch at batch=" << b << ", q_head=" << qh;
            ASSERT_EQ(v_array_h.data()[idx], expected_v)
                << "v pointer mismatch at batch=" << b << ", q_head=" << qh;
            ASSERT_EQ(qk_array_h.data()[idx], expected_qk)
                << "qk pointer mismatch at batch=" << b << ", q_head=" << qh;
            ASSERT_EQ(qkv_array_h.data()[idx], expected_qkv)
                << "qkv pointer mismatch at batch=" << b << ", q_head=" << qh;
        }
    }
}

// ------------------------------------------------------------------
// build_q_k_v_qk_qkv_array kernel tests
// ------------------------------------------------------------------
TEST(BuildQKVArrayTest, MhaSingleBatch)
{
    // Multi-head attention: each query head has its own kv head.
    run_pointer_array_test<float>(/*batch_size=*/1,
                                  /*q_head_num=*/4,
                                  /*kv_head_num=*/4,
                                  /*q_cache_len=*/8,
                                  /*kv_cache_len=*/16,
                                  /*head_size=*/32);
}

TEST(BuildQKVArrayTest, GqaSingleBatch)
{
    // GQA: 8 query heads share 2 kv heads (4 query heads per kv head).
    run_pointer_array_test<float>(/*batch_size=*/1,
                                  /*q_head_num=*/8,
                                  /*kv_head_num=*/2,
                                  /*q_cache_len=*/4,
                                  /*kv_cache_len=*/8,
                                  /*head_size=*/16);
}

TEST(BuildQKVArrayTest, GqaMultiBatch)
{
    run_pointer_array_test<float>(/*batch_size=*/3,
                                  /*q_head_num=*/6,
                                  /*kv_head_num=*/2,
                                  /*q_cache_len=*/4,
                                  /*kv_cache_len=*/10,
                                  /*head_size=*/8);
}

TEST(BuildQKVArrayTest, ManyQueryHeadsPerKvHead)
{
    // Stress the x-dimension loop: q_heads_per_kv_head > warp size.
    run_pointer_array_test<float>(/*batch_size=*/2,
                                  /*q_head_num=*/64,
                                  /*kv_head_num=*/1,
                                  /*q_cache_len=*/2,
                                  /*kv_cache_len=*/4,
                                  /*head_size=*/8);
}

TEST(BuildQKVArrayTest, ManyKvHeads)
{
    // Stress the y-dimension loop: kv_head_num > warp size.
    run_pointer_array_test<float>(/*batch_size=*/2,
                                  /*q_head_num=*/64,
                                  /*kv_head_num=*/64,
                                  /*q_cache_len=*/2,
                                  /*kv_cache_len=*/4,
                                  /*head_size=*/8);
}

TEST(BuildQKVArrayTest, LargeBatch)
{
    // batch_size > NUM_BLOCKS_X (256) to exercise the batch loop.
    run_pointer_array_test<float>(/*batch_size=*/300,
                                  /*q_head_num=*/4,
                                  /*kv_head_num=*/2,
                                  /*q_cache_len=*/4,
                                  /*kv_cache_len=*/8,
                                  /*head_size=*/16);
}

TEST(BuildQKVArrayTest, HalfPrecision)
{
    // Pointer byte offsets depend on sizeof(T); verify with half.
    run_pointer_array_test<half>(/*batch_size=*/2,
                                 /*q_head_num=*/8,
                                 /*kv_head_num=*/2,
                                 /*q_cache_len=*/4,
                                 /*kv_cache_len=*/8,
                                 /*head_size=*/16);
}
