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

// ============================================================================
// V80 backend transport HAL. Exposes:
//   * device lifecycle       — init() / ~vx_device()
//   * CP register channel    — cp_reg_read / cp_reg_write
//   * CP-visible host memory — host_mem_alloc / host_mem_free
//
// Device-memory allocation, DMA and capability decoding all live in the
// common core; the Command Processor is the sole memory engine. Host memory
// is the command ring + DMA staging; it must stay coherent with the device's
// view of it without an explicit sync, which plain process memory satisfies
// while the model runs in-process.
// ============================================================================

#include <common.h>

#ifdef SCOPE
#include "scope.h"
#endif

#ifdef AVEDSIM
#include <vrt_c.h>
#else
#include <vrt/device.hpp>
#include <vrt/kernel.hpp>
#endif

#include <cstdlib>
#include <exception>
#include <stdio.h>
#include <string>
#include <util.h>

using namespace vortex;

#ifndef AVEDSIM
#define CPP_API
#endif

#define MMIO_CTL_ADDR 0x00
#define MMIO_SCP_ADDR 0x28

#define CTL_AP_IDLE  (1 << 2)
#define CTL_AP_RESET (1 << 4)

// ----- Command Processor regfile -----
// Host addresses 0x1000..0x1FFF reach the CP regfile, which sees them as its
// native 0x000-based 12-bit space. Callers pass the CP-internal offset;
// cp_reg_* add this base.
#define CP_BASE 0x1000

#define DEFAULT_DEVICE_BDF "0000:11:00"
#define DEFAULT_VBIN_PATH  "vortex_afu.vbin"
#define KERNEL_NAME        "vortex_afu_0"

#ifdef CPP_API
typedef vrt::Device vrt_device_t;
typedef vrt::Kernel vrt_kernel_t;
#else
typedef vrtDeviceHandle vrt_device_t;
typedef vrtKernelHandle vrt_kernel_t;
#endif

#define CHECK_HANDLE(handle, _expr, _cleanup)                                  \
  auto handle = _expr;                                                         \
  if (handle == nullptr) {                                                     \
    printf("[VXDRV] Error: '%s' returned NULL!\n", #_expr);                    \
    _cleanup                                                                   \
  }

// VRT throws on error; propagating across the extern "C" callbacks_t
// boundary is UB, so every VRT-touching member wraps in a try/catch and
// returns -1 on exception. Errors are logged once so a missing vbin or an
// AXI-Lite timeout produces a diagnostic, not a silent SEGV from
// libstdc++'s unhandled-exception terminate path.
#define VRT_TRY() try {
#define VRT_CATCH(_ret) }                                                      \
  catch (const std::exception& _e) {                                           \
    fprintf(stderr, "[VXDRV] VRT exception: %s\n", _e.what());                 \
    return _ret;                                                               \
  } catch (...) {                                                              \
    fprintf(stderr, "[VXDRV] VRT exception (unknown)\n");                      \
    return _ret;                                                               \
  }

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
  vx_device()
#ifndef CPP_API
    : vrtDevice_(nullptr), vrtKernel_(nullptr)
#endif
  {}

  ~vx_device() {
  #ifdef SCOPE
    vx_scope_stop(this);
  #endif
  #ifndef CPP_API
    if (vrtKernel_) {
      vrtKernelClose(vrtKernel_);
    }
    if (vrtDevice_) {
      vrtDeviceClose(vrtDevice_);
    }
  #endif
  }

  int init() {
    // An empty value is treated as unset: the test harness exports these
    // unconditionally, so they arrive empty rather than absent.
    const char* bdf = getenv("VRT_DEVICE_BDF");
    if (bdf == nullptr || bdf[0] == '\0') {
      bdf = DEFAULT_DEVICE_BDF;
    }

    const char* vbin_path = getenv("VRT_VBIN_PATH");
    if (vbin_path == nullptr || vbin_path[0] == '\0') {
      vbin_path = DEFAULT_VBIN_PATH;
    }

  #ifdef CPP_API

    VRT_TRY()
      vrtDevice_ = vrt::Device(bdf, vbin_path);
      vrtKernel_ = vrt::Kernel(vrtDevice_, KERNEL_NAME);
    VRT_CATCH(-1)

  #else

    CHECK_HANDLE(vrtDevice, vrtDeviceOpen(bdf, vbin_path), {
      return -1;
    });

    CHECK_HANDLE(vrtKernel, vrtKernelOpen(vrtDevice, KERNEL_NAME), {
      vrtDeviceClose(vrtDevice);
      return -1;
    });

    vrtDevice_ = vrtDevice;
    vrtKernel_ = vrtKernel;

  #endif

    CHECK_ERR(this->write_register(MMIO_CTL_ADDR, CTL_AP_RESET), {
      return err;
    });

    // wait for the reset sequence to complete (ap_idle deasserts while the
    // device reset is in flight)
    {
      uint32_t ctl = 0;
      for (int retry = 0; retry < 1000; ++retry) {
        CHECK_ERR(this->read_register(MMIO_CTL_ADDR, &ctl), {
          return err;
        });
        if (ctl & CTL_AP_IDLE) {
          break;
        }
      }
      if ((ctl & CTL_AP_IDLE) == 0) {
        printf("[VXDRV] Error: device reset timeout!\n");
        return -1;
      }
    }

  #ifdef SCOPE
    {
      scope_callback_t callback;
      callback.registerWrite = [](vx_device_h hdevice, uint64_t value) -> int {
        auto device = (vx_device *)hdevice;
        uint32_t value_lo = (uint32_t)(value);
        uint32_t value_hi = (uint32_t)(value >> 32);
        CHECK_ERR(device->write_register(MMIO_SCP_ADDR, value_lo), {
          return err;
        });
        CHECK_ERR(device->write_register(MMIO_SCP_ADDR + 4, value_hi), {
          return err;
        });
        return 0;
      };
      callback.registerRead = [](vx_device_h hdevice, uint64_t *value) -> int {
        auto device = (vx_device *)hdevice;
        uint32_t value_lo, value_hi;
        CHECK_ERR(device->read_register(MMIO_SCP_ADDR, &value_lo), {
          return err;
        });
        CHECK_ERR(device->read_register(MMIO_SCP_ADDR + 4, &value_hi), {
          return err;
        });
        *value = (((uint64_t)value_hi) << 32) | value_lo;
        return 0;
      };
      CHECK_ERR(vx_scope_start(&callback, this, -1, -1), {
        return err;
      });
    }
  #endif

    return 0;
  }

  // ----- CP register channel -----
  int cp_reg_write(uint32_t off, uint32_t value) {
    return this->write_register(CP_BASE + off, value);
  }

  int cp_reg_read(uint32_t off, uint32_t *value) {
    return this->read_register(CP_BASE + off, value);
  }

  // ----- CP-visible host memory (command ring + DMA staging) -----
  int host_mem_alloc(uint64_t size, void **host_ptr, uint64_t *cp_addr) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    void *ptr = aligned_alloc(CACHE_BLOCK_SIZE, asize);
    if (ptr == nullptr)
      return -1;
    *host_ptr = ptr;
    *cp_addr  = reinterpret_cast<uint64_t>(ptr);
    return 0;
  }

  int host_mem_free(uint64_t cp_addr) {
    free(reinterpret_cast<void *>(cp_addr));
    return 0;
  }

private:

  int write_register(uint32_t addr, uint32_t value) {
  #ifdef CPP_API
    VRT_TRY()
      vrtKernel_.write(addr, value);
    VRT_CATCH(-1)
  #else
    CHECK_ERR(vrtKernelWriteRegister(vrtKernel_, addr, value), {
      return err;
    });
  #endif
    return 0;
  }

  int read_register(uint32_t addr, uint32_t *value) {
  #ifdef CPP_API
    VRT_TRY()
      *value = vrtKernel_.read(addr);
    VRT_CATCH(-1)
  #else
    CHECK_ERR(vrtKernelReadRegister(vrtKernel_, addr, value), {
      return err;
    });
  #endif
    return 0;
  }

  vrt_device_t vrtDevice_;
  vrt_kernel_t vrtKernel_;
};

#include <callbacks.inc>
