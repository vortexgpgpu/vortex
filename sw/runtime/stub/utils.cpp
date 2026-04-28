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

#include <iostream>
#include <fstream>
#include <list>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <vortex.h>
#include <assert.h>

extern int vx_upload_kernel_bytes(vx_device_h hdevice, const void* content, uint64_t size, vx_buffer_h* hbuffer) {
  if (nullptr == hdevice || nullptr == content || size <= 8 || nullptr == hbuffer)
    return -1;

  auto bytes = reinterpret_cast<const uint64_t*>(content);

  auto min_vma = *bytes++;
  auto max_vma = *bytes++;
  auto bin_size = size - 2 * 8;
  auto runtime_size = (max_vma - min_vma);

  vx_buffer_h _hbuffer;
  CHECK_ERR(vx_mem_reserve(hdevice, min_vma, runtime_size, 0, &_hbuffer), {
    return err;
  });

  // mask binary region as read-only
  CHECK_ERR(vx_mem_access(_hbuffer, 0, bin_size, VX_MEM_READ), {
    vx_mem_free(_hbuffer);
    return err;
  });

  // mark global variables region as read-write
  CHECK_ERR(vx_mem_access(_hbuffer, bin_size, runtime_size - bin_size, VX_MEM_READ_WRITE), {
    vx_mem_free(_hbuffer);
    return err;
  });

  CHECK_ERR(vx_copy_to_dev(_hbuffer, bytes, 0, bin_size), {
    vx_mem_free(_hbuffer);
    return err;
  });

  *hbuffer = _hbuffer;

  return 0;
}

extern int vx_upload_kernel_file(vx_device_h hdevice, const char* filename, vx_buffer_h* hbuffer) {
  if (nullptr == hdevice || nullptr == filename || nullptr == hbuffer)
    return -1;

  std::ifstream ifs(filename);
  if (!ifs) {
    std::cerr << "Error: " << filename << " not found" << std::endl;
    return -1;
  }

  // read file content
  ifs.seekg(0, ifs.end);
  auto size = ifs.tellg();
  std::vector<char> content(size);
  ifs.seekg(0, ifs.beg);
  ifs.read(content.data(), size);

  // upload buffer
  CHECK_ERR(vx_upload_kernel_bytes(hdevice, content.data(), size, hbuffer), {
    return err;
  });

  return 0;
}

extern int vx_upload_bytes(vx_device_h hdevice, const void* content, uint64_t size, vx_buffer_h* hbuffer) {
  if (nullptr == hdevice || nullptr == content || 0 == size || nullptr == hbuffer)
    return -1;

  vx_buffer_h _hbuffer;

  CHECK_ERR(vx_mem_alloc(hdevice, size, VX_MEM_READ, &_hbuffer), {
    return err;
  });

  CHECK_ERR(vx_copy_to_dev(_hbuffer, content, 0, size), {
    vx_mem_free(_hbuffer);
    return err;
  });

  *hbuffer = _hbuffer;

  return 0;
}

extern int vx_upload_file(vx_device_h hdevice, const char* filename, vx_buffer_h* hbuffer) {
  if (nullptr == hdevice || nullptr == filename || nullptr == hbuffer)
    return -1;

  std::ifstream ifs(filename);
  if (!ifs) {
    std::cerr << "Error: " << filename << " not found" << std::endl;
    return -1;
  }

  // read file content
  ifs.seekg(0, ifs.end);
  auto size = ifs.tellg();
  std::vector<char> content(size);
  ifs.seekg(0, ifs.beg);
  ifs.read(content.data(), size);

  // upload buffer
  CHECK_ERR(vx_upload_bytes(hdevice, content.data(), size, hbuffer), {
    return err;
  });

  return 0;
}

int vx_check_occupancy(vx_device_h hdevice, uint32_t block_size, uint32_t* max_localmem) {
   // check block size
  uint64_t warps_per_core, threads_per_warp;
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_WARPS, &warps_per_core), {
    return err;
  });
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_THREADS, &threads_per_warp), {
    return err;
  });
  uint32_t threads_per_core = warps_per_core * threads_per_warp;
  if (block_size > threads_per_core) {
    printf("Error: cannot schedule kernel with block_size > threads_per_core (%d,%d)\n", block_size, threads_per_core);
    return -1;
  }

  // calculate blocks occupancy
  int warps_per_block = (block_size + threads_per_warp-1) / threads_per_warp;
  int blocks_per_core = warps_per_core / warps_per_block;

  // Compute per-block local-memory budget. Treated as an out-only field:
  // some external callers (POCL pocl-vortex.c) pass uninitialized memory
  // here, so reading it as a user-supplied requirement is unsafe.
  if (max_localmem) {
    uint64_t local_mem_size;
    CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size), {
      return err;
    });
    *max_localmem = local_mem_size / blocks_per_core;
  }

  return 0;
}

extern int vx_mpm_query(vx_device_h hdevice, uint32_t mpm_class, uint32_t addr, uint32_t core_id, uint64_t* value) {
  if (!(addr >= VX_CSR_MPM_BASE && addr < (VX_CSR_MPM_BASE + 32))) {
     printf("Error: invalid MPM CSR address: 0x%x\n", addr);
     return -1;
  }

  auto read_one = [&](uint32_t cid, uint64_t* out) -> int {
    uint32_t lo, hi;
    uint32_t csr_id = addr - VX_CSR_MPM_BASE;
    uint32_t clss_sh= mpm_class << (16+6);
    uint32_t tag_lo = clss_sh | ((csr_id << 16) | cid);
    uint32_t tag_hi = clss_sh | (((csr_id + 32) << 16) | cid);
    CHECK_ERR(vx_dcr_read(hdevice, VX_DCR_BASE_MPM_VALUE, tag_lo, &lo), { return err; });
    CHECK_ERR(vx_dcr_read(hdevice, VX_DCR_BASE_MPM_VALUE, tag_hi, &hi), { return err; });
    *out = ((uint64_t)hi << 32) | lo;
    return 0;
  };

  if (core_id == 0xffffffff) {
    uint64_t num_cores;
    CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_CORES, &num_cores), { return err; });
    uint64_t sum = 0;
    for (uint32_t i = 0; i < (uint32_t)num_cores; ++i) {
      uint64_t cur;
      CHECK_ERR(read_one(i, &cur), { return err; });
      sum += cur;
    }
    *value = sum;
  } else {
    CHECK_ERR(read_one(core_id, value), { return err; });
  }
  return 0;
}