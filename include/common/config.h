#pragma once

constexpr unsigned int THREADS_PER_BLOCK = 512;
constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
constexpr unsigned int NUM_BLOCKS = 256;
constexpr float EPS = 1e-6f;