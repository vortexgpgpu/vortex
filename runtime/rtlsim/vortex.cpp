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

#include <mem.h>
#include <util.h>
#include <processor.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <future>
#include <list>
#include <chrono>

using namespace vortex;

class vx_device {
public:
  vx_device()
    : ram_(0, RAM_PAGE_SIZE)
    , global_mem_(ALLOC_BASE_ADDR,
                  GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR,
                  RAM_PAGE_SIZE,
                  CACHE_BLOCK_SIZE)
  {
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
      _value = ((uint64_t(MISA_EXT))<<32) | ((log2floor(XLEN)-4) << 30) | MISA_STD;
      break;
    case VX_CAPS_NUM_MEM_BANKS:
      _value = PLATFORM_MEMORY_NUM_BANKS;
      break;
    case VX_CAPS_MEM_BANK_SIZE:
      _value = 1ull << (MEM_ADDR_WIDTH / PLATFORM_MEMORY_NUM_BANKS);
      break;
    case VX_CAPS_CLOCK_RATE:
      _value = PLATFORM_CLOCK_RATE;
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

  int mem_alloc(uint64_t size, int flags, uint64_t* dev_addr) {
    uint64_t addr;
    CHECK_ERR(global_mem_.allocate(size, &addr), {
      return err;
    });
    CHECK_ERR(this->mem_access(addr, size, flags), {
      global_mem_.release(addr);
      return err;
    });
    *dev_addr = addr;
    return 0;
  }

  int mem_reserve(uint64_t dev_addr, uint64_t size, int flags) {
    CHECK_ERR(global_mem_.reserve(dev_addr, size), {
      return err;
    });
    CHECK_ERR(this->mem_access(dev_addr, size, flags), {
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

    if (flags | VX_MEM_WRITE) {
      flags |= VX_MEM_READ; // ensure caches can handle fill requests
    }

    ram_.set_acl(dev_addr, size, flags);

    return 0;
  }

  int mem_info(uint64_t* mem_free, uint64_t* mem_used) const {
    if (mem_free)
      *mem_free = global_mem_.free();
    if (mem_used)
      *mem_used = global_mem_.allocated();
    return 0;
  }

  int upload(uint64_t dest_addr, const void* src, uint64_t size) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (dest_addr + asize > GLOBAL_MEM_SIZE)
      return -1;

    ram_.enable_acl(false);
    ram_.write((const uint8_t*)src, dest_addr, size);
    ram_.enable_acl(true);

    /*printf("VXDRV: upload %ld bytes from 0x%lx:", size, uintptr_t((uint8_t*)src));
    for (int i = 0;  i < (asize / CACHE_BLOCK_SIZE); ++i) {
      printf("\n0x%08lx=", dest_addr + i * CACHE_BLOCK_SIZE);
      for (int j = 0;  j < CACHE_BLOCK_SIZE; ++j) {
        printf("%02x", *((uint8_t*)src + i * CACHE_BLOCK_SIZE + CACHE_BLOCK_SIZE - 1 - j));
      }
    }
    printf("\n");*/

    return 0;
  }

  int download(void* dest, uint64_t src_addr, uint64_t size) {
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
    ram_.read((uint8_t*)dest, src_addr, size);
    ram_.enable_acl(true);

    /*printf("VXDRV: download %ld bytes to 0x%lx:", size, uintptr_t((uint8_t*)dest));
    for (int i = 0;  i < (asize / CACHE_BLOCK_SIZE); ++i) {
      printf("\n0x%08lx=", src_addr + i * CACHE_BLOCK_SIZE);
      for (int j = 0;  j < CACHE_BLOCK_SIZE; ++j) {
        printf("%02x", *((uint8_t*)dest + i * CACHE_BLOCK_SIZE + CACHE_BLOCK_SIZE - 1 - j));
      }
    }
    printf("\n");*/

    return 0;
  }

  int start_wg(uint64_t krnl_addr, uint64_t args_addr, uint32_t dimension,
              const uint32_t* grid_dim, const uint32_t* block_dim, uint32_t lmem_size) {
     // ensure prior run completed
    if (future_.valid()) {
      future_.wait();
    }

    // setup kernel launch parameters
    uint32_t block_size, warp_step_x, warp_step_y, warp_step_z;
    prepare_kernel_launch_params(this, dimension, block_dim,
        &block_size, &warp_step_x, &warp_step_y, &warp_step_z);

    // configure kernel launch
    this->dcr_write(VX_DCR_KMU_STARTUP_ADDR0, krnl_addr & 0xffffffff);
    this->dcr_write(VX_DCR_KMU_STARTUP_ADDR1, krnl_addr >> 32);
    this->dcr_write(VX_DCR_KMU_STARTUP_ARG0, args_addr & 0xffffffff);
    this->dcr_write(VX_DCR_KMU_STARTUP_ARG1, args_addr >> 32);
    uint32_t grid_regs[] = {VX_DCR_KMU_GRID_DIM_X, VX_DCR_KMU_GRID_DIM_Y, VX_DCR_KMU_GRID_DIM_Z};
    uint32_t block_regs[] = {VX_DCR_KMU_BLOCK_DIM_X, VX_DCR_KMU_BLOCK_DIM_Y, VX_DCR_KMU_BLOCK_DIM_Z};
    for (uint32_t i = 0; i < 3; ++i) {
      this->dcr_write(grid_regs[i], (i < dimension) ? grid_dim[i] : 1);
      this->dcr_write(block_regs[i], (i < dimension && block_dim) ? block_dim[i] : 1);
    }
    this->dcr_write(VX_DCR_KMU_LMEM_SIZE, lmem_size);
    this->dcr_write(VX_DCR_KMU_BLOCK_SIZE, block_size);
    this->dcr_write(VX_DCR_KMU_WARP_STEP_X, warp_step_x);
    this->dcr_write(VX_DCR_KMU_WARP_STEP_Y, warp_step_y);
    this->dcr_write(VX_DCR_KMU_WARP_STEP_Z, warp_step_z);

    // start new run
    future_ = std::async(std::launch::async, [&]{
      processor_.run();
    });

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

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
    if (future_.valid()) {
      future_.wait(); // ensure prior run completed
    }
    return processor_.dcr_read(addr, tag, value);
  }


private:

  RAM                 ram_;
  Processor           processor_;
  MemoryAllocator     global_mem_;
  std::future<void>   future_;
};

#include <callbacks.inc>