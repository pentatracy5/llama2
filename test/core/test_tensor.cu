#include <vector>
#include <cstring>
#include <string>
#include <gtest/gtest.h>
#include <core/Tensor.cuh>

// ------------------------------------------------------------------
// Helper functions
// ------------------------------------------------------------------
template <typename T>
bool array_all_equal(const T *data, size_t n, T val)
{
    for (size_t i = 0; i < n; ++i)
    {
        if (data[i] != val)
            return false;
    }
    return true;
}

template <typename T>
bool array_all_zero(const T *data, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        if (data[i] != T(0))
            return false;
    }
    return true;
}

// ------------------------------------------------------------------
// Test cases
// ------------------------------------------------------------------

TEST(TensorTest, Constructor_CPU)
{
    Tensor<float> t({2, 3, 4}, CPU);
    ASSERT_EQ(t.numel(), 24);
    ASSERT_EQ(t.ndim(), 3);
    ASSERT_EQ(t.shape(), std::vector<unsigned int>({2, 3, 4}));
    ASSERT_EQ(t.dtype(), FP32);
    ASSERT_EQ(t.devtype(), CPU);
    ASSERT_FALSE(t.empty());
    ASSERT_NE(t.data(), nullptr);
    ASSERT_TRUE(array_all_zero(t.data(), t.numel()));
}

TEST(TensorTest, Constructor_Pinned)
{
    Tensor<int32_t> t({5, 5}, CPU_PINNED);
    ASSERT_EQ(t.numel(), 25);
    ASSERT_EQ(t.ndim(), 2);
    ASSERT_EQ(t.dtype(), INT32);
    ASSERT_EQ(t.devtype(), CPU_PINNED);
    ASSERT_NE(t.data(), nullptr);
    ASSERT_TRUE(array_all_zero(t.data(), t.numel()));
}

TEST(TensorTest, Constructor_GPU)
{
    Tensor<float> t({3, 3, 3}, GPU);
    ASSERT_EQ(t.numel(), 27);
    ASSERT_EQ(t.ndim(), 3);
    ASSERT_EQ(t.devtype(), GPU);
    ASSERT_NE(t.data(), nullptr);

    // Copy back to host to verify initialization to zero
    Tensor<float> h({3, 3, 3}, CPU);
    CUDA_CHECK(cudaMemcpy(h.data(), t.data(), h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(array_all_zero(h.data(), h.numel()));
}

TEST(TensorTest, EmptyTensor)
{
    // shape with explicit 0 size
    Tensor<float> t1({0}, CPU);
    ASSERT_TRUE(t1.empty());
    ASSERT_EQ(t1.numel(), 0);
    ASSERT_EQ(t1.data(), nullptr);

    // Non-empty scalar-like (empty shape vector gives size 1 in current impl)
    Tensor<float> t2({}, CPU);
    ASSERT_FALSE(t2.empty());
    ASSERT_EQ(t2.numel(), 1);
    ASSERT_EQ(t2.ndim(), 0);
    ASSERT_NE(t2.data(), nullptr);
}

TEST(TensorTest, NumelNdimShapeStrides)
{
    Tensor<float> t({2, 3, 4}, CPU);
    ASSERT_EQ(t.numel(), 24);
    ASSERT_EQ(t.ndim(), 3);
    ASSERT_EQ(t.shape()[0], 2);
    ASSERT_EQ(t.shape()[1], 3);
    ASSERT_EQ(t.shape()[2], 4);

    auto s = t.strides();
    ASSERT_EQ(s.size(), 3);
    ASSERT_EQ(s[0], 12); // 3*4
    ASSERT_EQ(s[1], 4);  // 4
    ASSERT_EQ(s[2], 1);
}

TEST(TensorTest, DataAccess)
{
    Tensor<int32_t> t({4}, CPU);
    int32_t *ptr = t.data();
    for (int i = 0; i < 4; ++i)
        ptr[i] = i * i;

    for (int i = 0; i < 4; ++i)
        ASSERT_EQ(ptr[i], i * i);
}

TEST(TensorTest, ConstantValSet_CPU)
{
    Tensor<float> t({10}, CPU);
    t.constant_val_set(3.14f);
    ASSERT_TRUE(array_all_equal(t.data(), t.numel(), 3.14f));
}

TEST(TensorTest, ConstantValSet_GPU)
{
    Tensor<float> t({100}, GPU);
    t.constant_val_set(2.718f);

    Tensor<float> h({100}, CPU);
    CUDA_CHECK(cudaMemcpy(h.data(), t.data(), h.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(array_all_equal(h.data(), h.numel(), 2.718f));
}

TEST(TensorTest, ToDevice)
{
    Tensor<float> h({5}, CPU);
    float vals[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::memcpy(h.data(), vals, sizeof(vals));

    h.to_device();
    ASSERT_EQ(h.devtype(), GPU);
    ASSERT_EQ(h.numel(), 5);

    // Copy back and verify
    Tensor<float> h2({5}, CPU);
    CUDA_CHECK(cudaMemcpy(h2.data(), h.data(), h2.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(h2.data()[i], vals[i]);
}

TEST(TensorTest, ToHost)
{
    Tensor<float> d({5}, GPU);
    float vals[5] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    CUDA_CHECK(cudaMemcpy(d.data(), vals, sizeof(vals), cudaMemcpyHostToDevice));

    d.to_host();
    ASSERT_EQ(d.devtype(), CPU);
    ASSERT_NE(d.data(), nullptr);
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(d.data()[i], vals[i]);
}

TEST(TensorTest, ToHostPinned)
{
    // CPU -> CPU_PINNED
    Tensor<int32_t> h({4}, CPU);
    for (int i = 0; i < 4; ++i)
        h.data()[i] = i + 10;

    h.to_host_pinned();
    ASSERT_EQ(h.devtype(), CPU_PINNED);
    for (int i = 0; i < 4; ++i)
        ASSERT_EQ(h.data()[i], i + 10);

    // GPU -> CPU_PINNED
    Tensor<int32_t> d({4}, GPU);
    int32_t vals[4] = {100, 200, 300, 400};
    CUDA_CHECK(cudaMemcpy(d.data(), vals, sizeof(vals), cudaMemcpyHostToDevice));

    d.to_host_pinned();
    ASSERT_EQ(d.devtype(), CPU_PINNED);
    for (int i = 0; i < 4; ++i)
        ASSERT_EQ(d.data()[i], vals[i]);
}

TEST(TensorTest, ResizeDiscard)
{
    Tensor<float> t({2, 3}, CPU);
    t.constant_val_set(1.0f);

    t.resize_discard({4, 5});
    ASSERT_EQ(t.numel(), 20);
    ASSERT_EQ(t.ndim(), 2);
    ASSERT_EQ(t.shape()[0], 4);
    ASSERT_EQ(t.shape()[1], 5);
    ASSERT_EQ(t.devtype(), CPU);
    // After resize_discard, data should be zero-initialized
    ASSERT_TRUE(array_all_zero(t.data(), t.numel()));
}

TEST(TensorTest, Swap)
{
    Tensor<float> a({2, 3}, CPU);
    a.constant_val_set(1.0f);

    Tensor<float> b({4, 5}, CPU);
    b.constant_val_set(2.0f);

    a.swap(b);
    ASSERT_EQ(a.numel(), 20);
    ASSERT_EQ(a.shape()[0], 4);
    ASSERT_TRUE(array_all_equal(a.data(), a.numel(), 2.0f));

    ASSERT_EQ(b.numel(), 6);
    ASSERT_EQ(b.shape()[0], 2);
    ASSERT_TRUE(array_all_equal(b.data(), b.numel(), 1.0f));
}

TEST(TensorTest, MoveConstructor)
{
    Tensor<float> src({3, 3}, CPU);
    src.constant_val_set(7.0f);

    Tensor<float> dst(std::move(src));
    ASSERT_EQ(dst.numel(), 9);
    ASSERT_TRUE(array_all_equal(dst.data(), dst.numel(), 7.0f));
    ASSERT_EQ(src.data(), nullptr); // moved-from
}

TEST(TensorTest, MoveAssignment)
{
    Tensor<float> src({2, 4}, CPU);
    src.constant_val_set(9.0f);

    Tensor<float> dst({1}, CPU);
    dst = std::move(src);
    ASSERT_EQ(dst.numel(), 8);
    ASSERT_TRUE(array_all_equal(dst.data(), dst.numel(), 9.0f));
    ASSERT_EQ(src.data(), nullptr);
}

TEST(TensorTest, As)
{
    Tensor<float> t({5}, CPU);
    BaseTensor *base = &t;

    Tensor<float> *p = base->as<float>();
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(p->numel(), 5);

    const BaseTensor *cbase = &t;
    const Tensor<float> *cp = cbase->as<float>();
    ASSERT_NE(cp, nullptr);
    ASSERT_EQ(cp->numel(), 5);
}

TEST(TensorTest, DTypeMapping)
{
    Tensor<float> t1({1}, CPU);
    ASSERT_EQ(t1.dtype(), FP32);

    Tensor<half> t2({1}, CPU);
    ASSERT_EQ(t2.dtype(), FP16);

    Tensor<int8_t> t3({1}, CPU);
    ASSERT_EQ(t3.dtype(), INT8);

    Tensor<int32_t> t4({1}, CPU);
    ASSERT_EQ(t4.dtype(), INT32);

    Tensor<bool> t5({1}, CPU);
    ASSERT_EQ(t5.dtype(), BOOL);

    Tensor<unsigned char> t6({1}, CPU);
    ASSERT_EQ(t6.dtype(), BYTES);
}
