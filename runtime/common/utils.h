// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vortex.h>
#include <cstdint>
#include <unordered_map>
#include <VX_config.h>
#include <VX_types.h>

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
#define ALLOC_MAX_ADDR      STARTUP_ADDR
#if (XLEN == 64)
#define GLOBAL_MEM_SIZE      0x200000000  // 8 GB
#else
#define GLOBAL_MEM_SIZE      0x100000000  // 4 GB
#endif