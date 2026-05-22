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
// XRT backend — a pure transport HAL (see callbacks.h). It exposes only:
//   * device lifecycle      — init() / ~vx_device()
//   * CP register channel   — cp_reg_read / cp_reg_write
//   * CP-visible host memory — host_mem_alloc / host_mem_free
//
// Device-memory allocation, DMA and capability decoding all live in the
// common core; the Command Processor is the sole memory engine. Host memory
// is the command ring + DMA staging, reachable by the CP's m_axi_host master
// (a host-only XRT BO on hardware; plain process memory under xrtsim, where
// the sim runs in-process and the m_axi_host model dereferences it directly).
// ============================================================================

#include <common.h>

#ifdef SCOPE
#include "scope.h"
#endif

// XRT includes
#ifdef XRTSIM
#include <xrt_c.h>
#else
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_error.h"
#include "experimental/xrt_ip.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_xclbin.h"
#endif

#include <cstdlib>
#include <limits>
#include <map>
#include <stdarg.h>
#include <string>
#include <util.h>
#include <vector>

using namespace vortex;

#ifndef XRTSIM
#define CPP_API
#endif

#define MMIO_CTL_ADDR 0x00
#define MMIO_SCP_ADDR 0x28

#define CTL_AP_START (1 << 0)
#define CTL_AP_DONE (1 << 1)
#define CTL_AP_IDLE (1 << 2)
#define CTL_AP_READY (1 << 3)
#define CTL_AP_RESET (1 << 4)
#define CTL_AP_RESTART (1 << 7)

// ----- Command Processor regfile -----
// The AXI-Lite demux in VX_afu_wrap routes host addresses 0x1000..0x1FFF
// to the CP regfile (mapped to CP's native 0x000-based 12-bit address
// space). Callers pass the CP-internal offset; cp_reg_* add this base.
#define CP_BASE              0x1000     // host-side base of CP regfile

#ifdef CPP_API
typedef xrt::device xrt_device_t;
typedef xrt::ip     xrt_kernel_t;
#else
typedef xrtDeviceHandle xrt_device_t;
typedef xrtKernelHandle xrt_kernel_t;
#endif

#define DEFAULT_DEVICE_INDEX 0
#define DEFAULT_XCLBIN_PATH "vortex_afu.xclbin"
#define KERNEL_NAME "vortex_afu"

#define CHECK_HANDLE(handle, _expr, _cleanup)                                  \
  auto handle = _expr;                                                         \
  if (handle == nullptr) {                                                     \
    printf("[VXDRV] Error: '%s' returned NULL!\n", #_expr);                    \
    _cleanup                                                                   \
  }

#ifndef CPP_API
static void dump_xrt_error(xrtDeviceHandle xrtDevice, xrtErrorCode err) {
  size_t len = 0;
  xrtErrorGetString(xrtDevice, err, nullptr, 0, &len);
  std::vector<char> buf(len);
  xrtErrorGetString(xrtDevice, err, buf.data(), buf.size(), nullptr);
  printf("[VXDRV] detail: %s!\n", buf.data());
}
#endif

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
  vx_device()
#ifndef CPP_API
    : xrtDevice_(nullptr), xrtKernel_(nullptr)
#endif
  {}

  ~vx_device() {
  #ifdef SCOPE
    vx_scope_stop(this);
  #endif
  #ifdef CPP_API
    // xrt::device / xrt::ip / xrt::bo are RAII.
    host_bos_.clear();
  #else
    if (xrtKernel_) xrtKernelClose(xrtKernel_);
    if (xrtDevice_) xrtDeviceClose(xrtDevice_);
  #endif
  }

  int init() {
    int device_index = DEFAULT_DEVICE_INDEX;
    const char *device_index_s = getenv("XRT_DEVICE_INDEX");
    if (device_index_s != nullptr) {
      device_index = atoi(device_index_s);
    }

    const char *xlbin_path_s = getenv("XRT_XCLBIN_PATH");
    if (xlbin_path_s == nullptr) {
      xlbin_path_s = DEFAULT_XCLBIN_PATH;
    }

  #ifdef CPP_API

    auto xrtDevice = xrt::device(device_index);
    auto uuid = xrtDevice.load_xclbin(xlbin_path_s);
    auto xrtKernel = xrt::ip(xrtDevice, uuid, KERNEL_NAME);

  #else

    CHECK_HANDLE(xrtDevice, xrtDeviceOpen(device_index), {
      return -1;
    });

  #ifndef XRTSIM
    CHECK_ERR(xrtDeviceLoadXclbinFile(xrtDevice, xlbin_path_s), {
      dump_xrt_error(xrtDevice, err);
      xrtDeviceClose(xrtDevice);
      return err;
    });

    xuid_t uuid;
    CHECK_ERR(xrtDeviceGetXclbinUUID(xrtDevice, uuid), {
      dump_xrt_error(xrtDevice, err);
      xrtDeviceClose(xrtDevice);
      return err;
    });

    CHECK_HANDLE(xrtKernel, xrtPLKernelOpenExclusive(xrtDevice, uuid, KERNEL_NAME), {
      xrtDeviceClose(xrtDevice);
      return -1;
    });
  #else
    xrtKernelHandle xrtKernel = xrtDevice;
  #endif

  #endif

    xrtDevice_ = xrtDevice;
    xrtKernel_ = xrtKernel;

    CHECK_ERR(this->write_register(MMIO_CTL_ADDR, CTL_AP_RESET), {
      return err;
    });

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

  #ifdef CHIPSCOPE
    std::cout << "\nPress ENTER to continue after setting up ILA trigger..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  #endif

    return 0;
  }

  // ----- CP register channel -----
  // VX_afu_wrap demuxes host AXI-Lite addresses 0x1000..0x1FFF to the CP
  // regfile (CP-internal 0x000-based offsets). Callers pass the CP-internal
  // offset; we add the AFU base here.
  int cp_reg_write(uint32_t off, uint32_t value) {
    return this->write_register(CP_BASE + off, value);
  }

  int cp_reg_read(uint32_t off, uint32_t *value) {
    return this->read_register(CP_BASE + off, value);
  }

  // ----- CP-visible host memory (command ring + DMA staging) -----
  // Reached by the CP's m_axi_host master. On hardware: a host-only XRT BO.
  // Under xrtsim: plain process memory — the sim runs in-process and the
  // m_axi_host model dereferences the address directly (see xrt_sim.cpp).
  int host_mem_alloc(uint64_t size, void **host_ptr, uint64_t *cp_addr) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
  #ifdef CPP_API
    xrt::bo bo(xrtDevice_, asize, xrt::bo::flags::host_only, 0);
    void *ptr     = bo.map<uint8_t *>();
    uint64_t addr = bo.address();
    host_bos_.emplace(addr, host_bo_t{std::move(bo), ptr});
    *host_ptr = ptr;
    *cp_addr  = addr;
  #else
    void *ptr = aligned_alloc(CACHE_BLOCK_SIZE, asize);
    if (ptr == nullptr)
      return -1;
    *host_ptr = ptr;
    *cp_addr  = reinterpret_cast<uint64_t>(ptr);
  #endif
    return 0;
  }

  int host_mem_free(uint64_t cp_addr) {
  #ifdef CPP_API
    auto it = host_bos_.find(cp_addr);
    if (it == host_bos_.end())
      return -1;
    host_bos_.erase(it);   // xrt::bo RAII releases the BO
  #else
    free(reinterpret_cast<void *>(cp_addr));
  #endif
    return 0;
  }

private:

  int write_register(uint32_t addr, uint32_t value) {
  #ifdef CPP_API
    xrtKernel_.write_register(addr, value);
  #else
    CHECK_ERR(xrtKernelWriteRegister(xrtKernel_, addr, value), {
      dump_xrt_error(xrtDevice_, err);
      return err;
    });
  #endif
    return 0;
  }

  int read_register(uint32_t addr, uint32_t *value) {
  #ifdef CPP_API
    *value = xrtKernel_.read_register(addr);
  #else
    CHECK_ERR(xrtKernelReadRegister(xrtKernel_, addr, value), {
      dump_xrt_error(xrtDevice_, err);
      return err;
    });
  #endif
    return 0;
  }

  xrt_device_t xrtDevice_;
  xrt_kernel_t xrtKernel_;

#ifdef CPP_API
  // Host-only BOs (CP-visible host memory), keyed by kernel-visible address.
  struct host_bo_t {
    xrt::bo bo;
    void   *ptr;
  };
  std::map<uint64_t, host_bo_t> host_bos_;
#endif
};

#include <callbacks.inc>
