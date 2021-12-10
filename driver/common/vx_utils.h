#pragma once

#include <cstdint>

uint64_t aligned_size(uint64_t size, uint64_t alignment);

bool is_aligned(uint64_t addr, uint64_t alignment);

#define CACHE_BLOCK_SIZE    64
#define ALLOC_BASE_ADDR     0x00000000
#define LOCAL_MEM_SIZE      4294967296     // 4 GB 