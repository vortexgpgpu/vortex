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

#include <unistd.h>
#include <string.h>
#include <string>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>

///////////////////////////////////////////////////////////////////////////////

static callbacks_t g_callbacks;
static void* g_drv_handle = nullptr;

typedef int (*vx_dev_init_t)(callbacks_t*);

extern int vx_dev_open(vx_device_h* hdevice) {
  {
    const char* driverName = getenv("VORTEX_DRIVER");
    if (driverName == nullptr) {
      driverName = "simx";
    }
    std::string driverName_s(driverName);
    std::string libName = "libvortex-" + driverName_s + ".so";
    auto handle = dlopen(libName.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
      std::cerr << "Cannot open library: " << dlerror() << std::endl;
      return 1;
    }

    auto vx_dev_init = (vx_dev_init_t)dlsym(handle, "vx_dev_init");
    auto dlsym_error = dlerror();
    if (dlsym_error) {
      std::cerr << "Cannot load symbol 'vx_init': " << dlsym_error << std::endl;
      dlclose(handle);
      return 1;
    }

    vx_dev_init(&g_callbacks);
    g_drv_handle = handle;
  }

  vx_device_h _hdevice;

  CHECK_ERR((g_callbacks.dev_open)(&_hdevice), {
    return err;
  });

  *hdevice = _hdevice;

  return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
  vx_dump_perf(hdevice, stdout);
  int ret = (g_callbacks.dev_close)(hdevice);
  dlclose(g_drv_handle);
  return ret;
}

extern int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t* value) {
  return (g_callbacks.dev_caps)(hdevice, caps_id, value);
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int flags, vx_buffer_h* hbuffer) {
  return (g_callbacks.mem_alloc)(hdevice, size, flags, hbuffer);
}

extern int vx_mem_reserve(vx_device_h hdevice, uint64_t address, uint64_t size, int flags, vx_buffer_h* hbuffer) {
  return (g_callbacks.mem_reserve)(hdevice, address, size, flags, hbuffer);
}

extern int vx_mem_free(vx_buffer_h hbuffer) {
  return (g_callbacks.mem_free)(hbuffer);
}

extern int vx_mem_access(vx_buffer_h hbuffer, uint64_t offset, uint64_t size, int flags) {
  return (g_callbacks.mem_access)(hbuffer, offset, size, flags);
}

extern int vx_mem_address(vx_buffer_h hbuffer, uint64_t* address) {
  return (g_callbacks.mem_address)(hbuffer, address);
}

extern int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_used) {
  return (g_callbacks.mem_info)(hdevice, mem_free, mem_used);
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr, uint64_t dst_offset, uint64_t size) {
  return (g_callbacks.copy_to_dev)(hbuffer, host_ptr, dst_offset, size);
}

extern int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer, uint64_t src_offset, uint64_t size) {
  return (g_callbacks.copy_from_dev)(host_ptr, hbuffer, src_offset, size);
}

extern int vx_copy_dev_to_dev(vx_buffer_h hdest_buffer, uint64_t dest_offset, vx_buffer_h hsrc_buffer, uint64_t src_offset, uint64_t size) {
  return (g_callbacks.copy_dev_to_dev)(hdest_buffer, dest_offset, hsrc_buffer, src_offset, size);
}

extern int vx_start(vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h harguments) {
  // schedule a CTA on each core
  uint64_t num_cores;
  CHECK_ERR ((g_callbacks.dev_caps)(hdevice, VX_CAPS_NUM_CORES, &num_cores), { return err; });
  return (g_callbacks.start_wg)(hdevice, hkernel, harguments, 1, (const uint32_t*)&num_cores, nullptr, 0);
}

extern int vx_start_wg(vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h harguments,
                       uint32_t dimension, const uint32_t* grid_dim, const uint32_t * block_dim, uint32_t lmem_size) {
  uint32_t block_size = 1;
  if (block_dim) {
    for (uint32_t i = 0; i < dimension; ++i) {
      block_size *= block_dim[i];
  }
  }
  CHECK_ERR(vx_check_occupancy(hdevice, block_size, &lmem_size), { return err; });
  return (g_callbacks.start_wg)(hdevice, hkernel, harguments, dimension, grid_dim, block_dim, lmem_size);
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
  return (g_callbacks.ready_wait)(hdevice, timeout);
}

extern int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint32_t value) {
  return (g_callbacks.dcr_write)(hdevice, addr, value);
}

extern int vx_dcr_read(vx_device_h hdevice, uint32_t addr, uint32_t tag, uint32_t* value) {
  return (g_callbacks.dcr_read)(hdevice, addr, tag, value);
}