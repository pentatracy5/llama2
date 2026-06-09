#pragma once

#include <cuda_runtime.h>
#include <chrono>

struct Timer
{
    Timer();
    ~Timer();
    void tic_gpu(cudaStream_t stream = 0);
    float toc_gpu(cudaStream_t stream = 0); // return milliseconds
    void tic_cpu();
    float toc_cpu(); // return milliseconds

    cudaEvent_t start, stop;
    std::chrono::high_resolution_clock::time_point begin;
};