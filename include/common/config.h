#pragma once

constexpr unsigned int SHARED_MEM_BANK_BYTE_SIZE = 4;
constexpr unsigned int CUDA_VEC_LS_BYTE_SIZE = 16;
constexpr unsigned int Q_CACHE_LEN = 512;
constexpr unsigned int KV_CACHE_LEN = 8192;
constexpr unsigned int THREADS_PER_BLOCK = 512;
constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
constexpr unsigned int NUM_BLOCKS_X = 256;
constexpr unsigned int NUM_BLOCKS_Y = 2;
constexpr unsigned int NUM_BLOCKS_Z = 2;
constexpr float SOFTMAX_EPS = 1e-5f;
constexpr float RMSNORM_EPS = 1e-5f;
constexpr float ROPE_BASE = 10000.f;
constexpr float TRAIN_SEQ_LEN = 4096.f;
