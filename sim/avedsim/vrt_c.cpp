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

#include "vrt_c.h"
#include "xrt_sim.h"

using namespace vortex;

#ifdef __cplusplus
extern "C" {
#endif

// The kernel handle aliases the device handle: a single AFU owns the whole
// AXI-Lite space, so there is nothing to resolve by name.

extern vrtDeviceHandle vrtDeviceOpen(const char* /*bdf*/, const char* /*vbin_path*/) {
  auto sim = new xrt_sim();
  if (sim->init() != 0) {
    delete sim;
    return nullptr;
  }
  return sim;
}

extern int vrtDeviceClose(vrtDeviceHandle dhdl) {
  if (dhdl == nullptr) {
    return -1;
  }
  delete reinterpret_cast<xrt_sim*>(dhdl);
  return 0;
}

extern vrtKernelHandle vrtKernelOpen(vrtDeviceHandle dhdl, const char* /*name*/) {
  return dhdl;
}

extern int vrtKernelClose(vrtKernelHandle /*khdl*/) {
  return 0;
}

extern int vrtKernelWriteRegister(vrtKernelHandle khdl, uint32_t offset, uint32_t data) {
  if (khdl == nullptr) {
    return -1;
  }
  return reinterpret_cast<xrt_sim*>(khdl)->register_write(offset, data);
}

extern int vrtKernelReadRegister(vrtKernelHandle khdl, uint32_t offset, uint32_t* data) {
  if (khdl == nullptr) {
    return -1;
  }
  return reinterpret_cast<xrt_sim*>(khdl)->register_read(offset, data);
}

#ifdef __cplusplus
}
#endif
