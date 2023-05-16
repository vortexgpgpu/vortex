#pragma once

#include <vortex.h>
#include <cstdint>
#include <unordered_map>

class DeviceConfig {
public:    
    void write(uint32_t addr, uint32_t value);
    uint32_t read(uint32_t addr) const;
private:
     std::unordered_map<uint32_t, uint32_t> data_;
};

int dcr_initialize(vx_device_h device);

uint64_t aligned_size(uint64_t size, uint64_t alignment);

bool is_aligned(uint64_t addr, uint64_t alignment);

void perf_add_device(vx_device_h device);

void perf_remove_device(vx_device_h device);

#define CACHE_BLOCK_SIZE    64
#define ALLOC_BASE_ADDR     CACHE_BLOCK_SIZE
#define ALLOC_MAX_ADDR      0x80000000   // 2 GB 
#define LOCAL_MEM_SIZE      0x100000000  // 4 GB 