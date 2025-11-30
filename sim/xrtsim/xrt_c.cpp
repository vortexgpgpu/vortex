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

#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <assert.h>
#include "xrt_c.h"
#include "xrt_sim.h"
#include <VX_config.h>
#include <util.h>

using namespace vortex;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t   size;
  xrt_sim* sim;
  uint32_t bank;
  uint64_t addr;
} buffer_t;

extern xrtDeviceHandle xrtDeviceOpen(unsigned int index) {
  if (index != 0)
    return nullptr;
  auto sim = new xrt_sim();
  int ret = sim->init();
  if (ret != 0) {
    delete sim;
    return nullptr;
  }
  return sim;
}

extern int xrtXclbinGetXSAName(xrtDeviceHandle /*dhdl*/, char* name, int size, int* ret_size) {
  static const char* deviceName = "vortex_xrtsim";
  if (name) {
    if (size < strlen(deviceName) + 1)
      return -1;
    memcpy(name, deviceName, size);
  }
  if (ret_size) {
    *ret_size = strlen(deviceName) + 1;
  }
  return 0;
}

extern int xrtDeviceClose(xrtDeviceHandle dhdl) {
  if (dhdl == nullptr)
    return -1;
  auto sim = reinterpret_cast<xrt_sim*>(dhdl);
  delete sim;
  return 0;
}

extern int xrtKernelClose(xrtKernelHandle /*kernelHandle*/) {
  return 0;
}

extern xrtBufferHandle xrtBOAlloc(xrtDeviceHandle dhdl, size_t size, xrtBufferFlags flags, xrtMemoryGroup grp) {
  auto sim = reinterpret_cast<xrt_sim*>(dhdl);
  uint64_t addr;
  int err = sim->mem_alloc(size, grp, &addr);
  if (err != 0)
    return nullptr;
  auto buffer   = new buffer_t();
  buffer->size  = size;
  buffer->bank  = grp;
  buffer->sim   = sim;
  buffer->addr  = addr;
  return buffer;
}

extern int xrtBOFree(xrtBufferHandle bhdl) {
  if (bhdl == nullptr)
    return -1;
  auto buffer = reinterpret_cast<buffer_t*>(bhdl);
  return buffer->sim->mem_free(buffer->bank, buffer->addr);
}

extern int xrtBOWrite(xrtBufferHandle bhdl, const void* src, size_t size, size_t offset) {
  if (bhdl == nullptr)
    return -1;
  auto buffer = reinterpret_cast<buffer_t*>(bhdl);
  return buffer->sim->mem_write(buffer->bank, buffer->addr + offset, size, src);
}

extern int xrtBORead(xrtBufferHandle bhdl, void* dst, size_t size, size_t offset) {
  if (bhdl == nullptr)
    return -1;
  auto buffer = reinterpret_cast<buffer_t*>(bhdl);
  return buffer->sim->mem_read(buffer->bank, buffer->addr + offset, size, dst);
}

extern int xrtBOSync(xrtBufferHandle bhdl, enum xclBOSyncDirection dir, size_t size, size_t offset) {
  return 0;
}

extern int xrtKernelWriteRegister(xrtKernelHandle kernelHandle, uint32_t offset, uint32_t data) {
  if (kernelHandle == nullptr)
    return -1;
  auto sim = reinterpret_cast<xrt_sim*>(kernelHandle);
  return sim->register_write(offset, data);
}

extern int xrtKernelReadRegister(xrtKernelHandle kernelHandle, uint32_t offset, uint32_t* data) {
  if (kernelHandle == nullptr)
    return -1;
  auto sim = reinterpret_cast<xrt_sim*>(kernelHandle);
  return sim->register_read(offset, data);
}

extern int xrtErrorGetString(xrtDeviceHandle, xrtErrorCode error, char* out, size_t len, size_t* out_len) {
  return 0;
}

#ifdef __cplusplus
}
#endif
