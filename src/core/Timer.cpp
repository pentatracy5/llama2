#include <core/Timer.h>

Timer::Timer()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

Timer::~Timer()
{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void Timer::tic_gpu(cudaStream_t stream/* = 0*/)
{
    cudaEventRecord(start, stream);
}

float Timer::toc_gpu(cudaStream_t stream/* = 0*/)
{
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

void Timer::tic_cpu()
{
    begin = std::chrono::high_resolution_clock::now();
}

float Timer::toc_cpu()
{
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = finish - begin;
    return elapsed.count() * 1000.0f;
}