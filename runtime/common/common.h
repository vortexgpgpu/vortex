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
#include <VX_config.h>
#include <VX_types.h>
#include <callbacks.h>
#include <malloc.h>

#include <cstdint>
#include <unordered_map>
#include <array>

#define CACHE_BLOCK_SIZE  64

#define RAM_PAGE_SIZE     4096

#define ALLOC_BASE_ADDR   USER_BASE_ADDR

#if (XLEN == 64)
#define GLOBAL_MEM_SIZE    0x200000000  // 8 GB
#else
#define GLOBAL_MEM_SIZE    0x100000000  // 4 GB
#endif

#ifndef NDEBUG
#define DBGPRINT(format, ...) do { printf("[VXDRV] " format "", ##__VA_ARGS__); } while (0)
#else
#define DBGPRINT(format, ...) ((void)0)
#endif

#define CHECK_ERR(_expr, _cleanup) \
  do { \
    auto err = _expr; \
    if (err == 0) \
      break; \
    printf("[VXDRV] Error: '%s' returned %d!\n", #_expr, (int)err); \
    _cleanup \
  } while (false)

class DeviceConfig {
public:
  void write(uint32_t addr, uint32_t value) {
    store_[addr] = value;
  }
  int read(uint32_t addr, uint32_t* value) const {
    auto it = store_.find(addr);
    if (it == store_.end())
      return -1;
    *value = it->second;
    return 0;
  }
private:
  std::unordered_map<uint32_t, uint32_t> store_;
};

inline uint64_t aligned_size(uint64_t size, uint64_t alignment) {
  assert(0 == (alignment & (alignment - 1)));
  return (size + alignment - 1) & ~(alignment - 1);
}

inline bool is_aligned(uint64_t addr, uint64_t alignment) {
  assert(0 == (alignment & (alignment - 1)));
  return 0 == (addr & (alignment - 1));
}