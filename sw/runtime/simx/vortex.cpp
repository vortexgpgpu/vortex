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

#include <common.h>

#include <constants.h>
#include <mem.h>
#include <processor.h>
#include <util.h>

#include <assert.h>
#include <chrono>
#include <future>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <VX_config.h>

using namespace vortex;

class vx_device {
public:
  vx_device()
      : ram_(0, MEM_PAGE_SIZE), processor_(), global_mem_(ALLOC_BASE_ADDR, GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR, MEM_PAGE_SIZE, CACHE_BLOCK_SIZE) {
    // attach memory module
    processor_.attach_ram(&ram_);
  }

  ~vx_device() {
    if (future_.valid()) {
      future_.wait();
    }
  }

  int init() {
    return 0;
  }

  int get_caps(uint32_t caps_id, uint64_t *value) {
    uint64_t _value;
    switch (caps_id) {
    case VX_CAPS_VERSION:
      _value = IMPLEMENTATION_ID;
      break;
    case VX_CAPS_NUM_THREADS:
      _value = NUM_THREADS;
      break;
    case VX_CAPS_NUM_WARPS:
      _value = NUM_WARPS;
      break;
    case VX_CAPS_NUM_CORES:
      _value = NUM_CORES * NUM_CLUSTERS;
      break;
    case VX_CAPS_NUM_CLUSTERS:
      _value = NUM_CLUSTERS;
      break;
    case VX_CAPS_SOCKET_SIZE:
      _value = SOCKET_SIZE;
      break;
    case VX_CAPS_ISSUE_WIDTH:
      _value = ISSUE_WIDTH;
      break;
    case VX_CAPS_CACHE_LINE_SIZE:
      _value = CACHE_BLOCK_SIZE;
      break;
    case VX_CAPS_GLOBAL_MEM_SIZE:
      _value = GLOBAL_MEM_SIZE;
      break;
    case VX_CAPS_LOCAL_MEM_SIZE:
      _value = (1 << LMEM_LOG_SIZE);
      break;
    case VX_CAPS_ISA_FLAGS:
      _value = ((uint64_t(MISA_EXT)) << 32) | ((log2floor(XLEN) - 4) << 30) | MISA_STD;
      break;
    case VX_CAPS_NUM_MEM_BANKS:
      _value = PLATFORM_MEMORY_NUM_BANKS;
      break;
    case VX_CAPS_MEM_BANK_SIZE:
      _value = 1ull << (MEM_ADDR_WIDTH / PLATFORM_MEMORY_NUM_BANKS);
      break;
    case VX_CAPS_CLOCK_RATE:
      _value = 0;
      break;
    case VX_CAPS_PEAK_MEM_BW:
      _value = PLATFORM_MEMORY_PEAK_BW;
      break;
    default:
      std::cout << "invalid caps id: " << caps_id << std::endl;
      std::abort();
      return -1;
    }
    *value = _value;
    return 0;
  }

  int mem_alloc(uint64_t size, int flags, uint64_t *dev_addr) {
    uint64_t asize = size;
    uint64_t addr = 0;

    DBGPRINT("[RT:mem_alloc] size: 0x%lx, asize, 0x%lx,flag : 0x%d\n", size, asize, flags);
    CHECK_ERR(global_mem_.allocate(asize, &addr), {
      return err;
    });
    CHECK_ERR(this->mem_access(addr, asize, flags), {
      global_mem_.release(addr);
      return err;
    });
    *dev_addr = addr;
    return 0;
  }

  int mem_reserve(uint64_t dev_addr, uint64_t size, int flags) {
    uint64_t asize = size;
    CHECK_ERR(global_mem_.reserve(dev_addr, asize), {
      return err;
    });
    DBGPRINT("[RT:mem_reserve] addr: 0x%lx, asize:0x%lx, size: 0x%lx\n", dev_addr, asize, size);
    CHECK_ERR(this->mem_access(dev_addr, asize, flags), {
      global_mem_.release(dev_addr);
      return err;
    });
    return 0;
  }

  int mem_free(uint64_t dev_addr) {
    return global_mem_.release(dev_addr);
  }

  int mem_access(uint64_t dev_addr, uint64_t size, int flags) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (dev_addr + asize > GLOBAL_MEM_SIZE)
      return -1;

    ram_.set_acl(dev_addr, size, flags);
    return 0;
  }

  int mem_info(uint64_t *mem_free, uint64_t *mem_used) const {
    if (mem_free)
      *mem_free = global_mem_.free();
    if (mem_used)
      *mem_used = global_mem_.allocated();
    return 0;
  }

  int copy(uint64_t dest_addr, uint64_t src_addr, uint64_t size) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (src_addr + asize > GLOBAL_MEM_SIZE || dest_addr + asize > GLOBAL_MEM_SIZE)
      return -1;
    ram_.enable_acl(false);
    ram_.copy(dest_addr, src_addr, size);
    ram_.enable_acl(true);
    return 0;
  }

  int upload(uint64_t dest_addr, const void *src, uint64_t size) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (dest_addr + asize > GLOBAL_MEM_SIZE)
      return -1;

    ram_.enable_acl(false);
    ram_.write((const uint8_t *)src, dest_addr, size);
    ram_.enable_acl(true);

    /*
    DBGPRINT("upload %ld bytes to 0x%lx\n", size, dest_addr);
    for (uint64_t i = 0; i < size && i < 1024; i += 4) {
        DBGPRINT("  0x%lx <- 0x%x\n", dest_addr + i, *(uint32_t*)((uint8_t*)src + i));
    }*/

    return 0;
  }

  int download(void *dest, uint64_t src_addr, uint64_t size) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (src_addr + asize > GLOBAL_MEM_SIZE)
      return -1;

    // flush GPU caches before reading back results
    {
      uint32_t dummy;
      for (uint32_t cid = 0; cid < NUM_CORES * NUM_CLUSTERS; ++cid) {
        this->dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy);
      }
    }

    ram_.enable_acl(false);
    ram_.read((uint8_t *)dest, src_addr, size);
    ram_.enable_acl(true);

    /*DBGPRINT("download %ld bytes from 0x%lx\n", size, src_addr);
    for (uint64_t i = 0; i < size && i < 1024; i += 4) {
        DBGPRINT("  0x%lx -> 0x%x\n", src_addr + i, *(uint32_t*)((uint8_t*)dest + i));
    }*/

    return 0;
  }

  int start() {
    // DCRs already written by stub; just trigger execution
    future_ = std::async(std::launch::async, [&] { processor_.run(); });
    return 0;
  }

  int ready_wait(uint64_t timeout) {
    if (!future_.valid())
      return 0;
    uint64_t timeout_sec = timeout / 1000;
    std::chrono::seconds wait_time(1);
    for (;;) {
      // wait for 1 sec and check status
      auto status = future_.wait_for(wait_time);
      if (status == std::future_status::ready)
        break;
      if (0 == timeout_sec--)
        return -1;
    }
    return 0;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    if (future_.valid()) {
      future_.wait(); // ensure prior run completed
    }
    return processor_.dcr_write(addr, value);
  }

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t *value) {
    if (future_.valid()) {
      future_.wait(); // ensure prior run completed
    }
    return processor_.dcr_read(addr, tag, value);
  }

private:
  RAM ram_;
  Processor processor_;
  MemoryAllocator global_mem_;
  std::future<void> future_;
};

#include <callbacks.inc>
