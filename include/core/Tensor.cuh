#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <memory>
#include <common/types.h>
#include <common/macro.h>

template <typename T>
__global__ void fill_kernel(T *data, T val, unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    while (idx < n)
    {
        data[idx] = val;
        idx += stride;
    }
}

template <typename T>
class Tensor;

class BaseTensor
{
public:
    explicit BaseTensor(const std::vector<unsigned int> &shape,
                        const DataType dtype,
                        const DeviceType devtype)
        : shape_(shape),
          dtype_(dtype),
          devtype_(devtype)
    {
        size_ = std::accumulate(shape_.begin(), shape_.end(), static_cast<unsigned int>(1), std::multiplies<unsigned int>());
    }

    virtual ~BaseTensor() = default;

    BaseTensor(const BaseTensor &) = delete;

    BaseTensor &operator=(const BaseTensor &) = delete;

    BaseTensor(BaseTensor &&other) noexcept
        : size_(other.size_),
          shape_(std::move(other.shape_)),
          dtype_(other.dtype_),
          devtype_(other.devtype_)
    {
        other.size_ = 0;
    }

    BaseTensor &operator=(BaseTensor &&other) noexcept
    {
        if (this != &other)
        {
            size_ = other.size_;
            shape_ = std::move(other.shape_);
            dtype_ = other.dtype_;
            devtype_ = other.devtype_;
            other.size_ = 0;
        }
        return *this;
    }

    bool empty() const noexcept
    {
        return size_ == 0;
    }

    unsigned int numel() const noexcept
    {
        return size_;
    }

    int ndim() const noexcept
    {
        return static_cast<int>(shape_.size());
    }

    const std::vector<unsigned int> &shape() const noexcept
    {
        return shape_;
    }

    DataType dtype() const noexcept
    {
        return dtype_;
    }

    DeviceType devtype() const noexcept
    {
        return devtype_;
    }

    std::vector<unsigned int> strides() const
    {
        std::vector<unsigned int> s(ndim());
        unsigned int stride = 1;
        for (int i = ndim() - 1; i >= 0; --i)
        {
            s[i] = stride;
            stride *= shape_[i];
        }
        return s;
    }

    template <typename T>
    Tensor<T> *as()
    {
        assert(RealTypeToDataType<T>::dtype == dtype_ && "Type mismatch in as<T>()");
        return static_cast<Tensor<T> *>(this);
    }

    template <typename T>
    const Tensor<T> *as() const
    {
        assert(RealTypeToDataType<T>::dtype == dtype_ && "Type mismatch in as<T>()");
        return static_cast<const Tensor<T> *>(this);
    }

protected:
    unsigned int size_;
    std::vector<unsigned int> shape_;
    DataType dtype_;
    DeviceType devtype_;
};

template <typename T>
class Tensor : public BaseTensor
{
public:
    explicit Tensor(const std::vector<unsigned int> &shape,
                    const DeviceType devtype = CPU,
                    cudaStream_t stream = 0)
        : BaseTensor(shape, RealTypeToDataType<T>::dtype, devtype),
          data_(nullptr)
    {
        if (size_ == 0)
            return;

        switch (devtype)
        {
        case CPU_PINNED:
            CUDA_CHECK(cudaHostAlloc((void **)&data_, size_ * sizeof(T), cudaHostAllocDefault));
            break;
        case CPU:
            data_ = (T *)malloc(size_ * sizeof(T));
            MALLOC_CHECK(data_);
            break;
        case GPU:
            CUDA_CHECK(cudaMalloc((void **)&data_, size_ * sizeof(T)));
            break;
        default:
            data_ = nullptr;
            return;
        }
        constant_val_set(T(0), stream);
    }

    ~Tensor()
    {
        release();
    }

    Tensor(const Tensor &) = delete;

    Tensor &operator=(const Tensor &) = delete;

    Tensor(Tensor &&other) noexcept
        : BaseTensor(std::move(other)),
          data_(other.data_)
    {
        other.data_ = nullptr;
    }

    Tensor &operator=(Tensor &&other) noexcept
    {
        if (this != &other)
        {
            release();
            BaseTensor::operator=(std::move(other));
            data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }

    T *data() noexcept
    {
        return data_;
    }

    const T *data() const noexcept
    {
        return data_;
    }

    void to_device(cudaStream_t stream = 0)
    {
        if (!data_ || devtype_ == GPU)
            return;

        T *d_temp = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_temp, size_ * sizeof(T)));

        if (devtype_ == CPU)
        {
            CUDA_CHECK(cudaMemcpy(d_temp, data_, size_ * sizeof(T), cudaMemcpyHostToDevice));
        }
        else // CPU_PINNED
        {
            CUDA_CHECK(cudaMemcpyAsync(d_temp, data_, size_ * sizeof(T), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        release();
        data_ = d_temp;
        devtype_ = GPU;
    }

    void to_host()
    {
        if (!data_ || devtype_ == CPU)
            return;

        T *h_temp = (T *)malloc(size_ * sizeof(T));
        MALLOC_CHECK(h_temp);

        if (devtype_ == GPU)
        {
            CUDA_CHECK(cudaMemcpy(h_temp, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }
        else // CPU_PINNED
        {
            std::memcpy(h_temp, data_, size_ * sizeof(T));
        }

        release();
        data_ = h_temp;
        devtype_ = CPU;
    }

    void to_host_pinned(cudaStream_t stream = 0)
    {
        if (!data_ || devtype_ == CPU_PINNED)
            return;

        T *h_temp = nullptr;
        CUDA_CHECK(cudaHostAlloc((void **)&h_temp, size_ * sizeof(T), cudaHostAllocDefault));

        if (devtype_ == GPU)
        {
            CUDA_CHECK(cudaMemcpyAsync(h_temp, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        else // CPU
        {
            std::memcpy(h_temp, data_, size_ * sizeof(T));
        }

        release();
        data_ = h_temp;
        devtype_ = CPU_PINNED;
    }

    void resize_discard(const std::vector<unsigned int> &shape,
                        cudaStream_t stream = 0)
    {
        Tensor<T> temp(shape, devtype_, stream);
        swap(temp);
    }

    void swap(Tensor &other) noexcept
    {
        using std::swap;
        swap(size_, other.size_);
        swap(shape_, other.shape_);
        swap(dtype_, other.dtype_);
        swap(devtype_, other.devtype_);
        swap(data_, other.data_);
    }

    void constant_val_set(const T &val,
                          cudaStream_t stream = 0)
    {
        if (empty() || !data_)
            return;

        if (devtype_ == GPU)
        {
            dim3 n_threads{size_};
            dim3 threads_per_block{256};
            CUDA_LAUNCH_SHAREDMEM_STREAM(fill_kernel, n_threads, threads_per_block, 0, stream)(data_, val, size_);
            CUDA_KERNEL_LAUNCH_CHECK();
        }
        else
        {
            std::fill(data_, data_ + size_, val);
        }
    }

private:
    void release()
    {
        if (!data_)
            return;

        switch (devtype_)
        {
        case CPU_PINNED:
            CUDA_CHECK(cudaFreeHost(data_));
            break;
        case CPU:
            free(data_);
            break;
        case GPU:
            CUDA_CHECK(cudaFree(data_));
            break;
        default:
            break;
        }
    }

private:
    T *data_;
};

using TensorMap = std::unordered_map<std::string, std::unique_ptr<BaseTensor>>;
