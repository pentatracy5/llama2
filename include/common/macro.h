#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define NBS_PER_DIM(n_threads, threads_per_block)   ((n_threads + threads_per_block - 1) / threads_per_block)
#define N_BLOCKS(n_threads, threads_per_block)      (dim3{NBS_PER_DIM(n_threads.x, threads_per_block.x), NBS_PER_DIM(n_threads.y, threads_per_block.y), NBS_PER_DIM(n_threads.z, threads_per_block.z)})

#define CUDA_LAUNCH(kernel, n_threads, threads_per_block)										        kernel<<<N_BLOCKS(n_threads, threads_per_block), threads_per_block>>>
#define CUDA_LAUNCH_SHAREDMEM(kernel, n_threads, threads_per_block, shared_mem_bytes)				    kernel<<<N_BLOCKS(n_threads, threads_per_block), threads_per_block, shared_mem_bytes>>>
#define CUDA_LAUNCH_SHAREDMEM_STREAM(kernel, n_threads, threads_per_block, shared_mem_bytes, stream)	kernel<<<N_BLOCKS(n_threads, threads_per_block), threads_per_block, shared_mem_bytes, stream>>>

#define MALLOC_CHECK(ptr)                                                        \
    do {                                                                         \
        if ((ptr) == nullptr) {                                                  \
            std::cerr << "Host memory allocation failed"                         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUDA_KERNEL_LAUNCH_CHECK()                                               \
    do {                                                                         \
        cudaError_t err = cudaGetLastError();                                    \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS error: " << cublasGetStatusString(status)       \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUDNN_CHECK(call)                                                        \
    do {                                                                         \
        cudnnStatus_t status = call;                                             \
        if (status != CUDNN_STATUS_SUCCESS) {                                    \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(status)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CURAND_CHECK(call)                                                       \
    do {                                                                         \
        curandStatus_t status = call;                                            \
        if (status != CURAND_STATUS_SUCCESS) {                                   \
            std::cerr << "cuRAND error: "                                        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)
