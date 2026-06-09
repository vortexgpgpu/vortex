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
// OPAE backend — a pure transport HAL (see callbacks.h). It exposes only:
//   * device lifecycle      — init() / ~vx_device()
//   * CP register channel   — cp_reg_read / cp_reg_write
//   * CP-visible host memory — host_mem_alloc / host_mem_free
//
// Device-memory allocation, DMA and capability decoding all live in the
// common core; the Command Processor is the sole memory engine. Host memory
// is the command ring + DMA staging — a CCI-P-shared buffer (fpgaPrepareBuffer)
// reached by the CP's CCI-P host bridge.
// ============================================================================

#include <common.h>

#include "driver.h"

#include <vortex_opae.h>

#ifdef SCOPE
#include "scope.h"
#endif

#include <cstdlib>
#include <cstring>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <uuid/uuid.h>

using namespace vortex;

#define MMIO_SCOPE_READ  (AFU_IMAGE_MMIO_SCOPE_READ * 4)
#define MMIO_SCOPE_WRITE (AFU_IMAGE_MMIO_SCOPE_WRITE * 4)

// ----- Command Processor regfile (host byte addresses) -----
// The AFU's MMIO demux routes byte addresses 0x1000..0x1FFF to the CP
// regfile (mapped to CP's native 0x000-based 12-bit address space).
// Callers pass the CP-internal offset; cp_reg_* add this base.
#define CP_BASE              0x1000

#define CHECK_FPGA_ERR(_expr, _cleanup)                                        \
  do {                                                                         \
    auto err = _expr;                                                          \
    if (err == 0)                                                              \
      break;                                                                   \
    printf("[VXDRV] Error: '%s' returned %d, %s!\n", #_expr, (int)err,         \
           api_.fpgaErrStr(err));                                              \
    _cleanup                                                                   \
  } while (false)

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
  vx_device()
    : fpga_(nullptr)
  {}

  ~vx_device() {
  #ifdef SCOPE
    vx_scope_stop(this);
  #endif
    if (fpga_ != nullptr) {
      for (auto& kv : host_bos_)
        api_.fpgaReleaseBuffer(fpga_, kv.second.wsid);
      host_bos_.clear();
      api_.fpgaClose(fpga_);
    }
    drv_close();
  }

  int init() {
    fpga_token accel_token;
    fpga_properties filter;
    fpga_guid guid;
    uint32_t num_matches;

    memset(&api_, 0, sizeof(opae_drv_api_t));
    if (drv_init(&api_) != 0) {
      return -1;
    }

    // Set up a filter that will search for an accelerator
    CHECK_FPGA_ERR(api_.fpgaGetProperties(nullptr, &filter), {
      return -1;
    });

    CHECK_FPGA_ERR(api_.fpgaPropertiesSetObjectType(filter, FPGA_ACCELERATOR), {
      api_.fpgaDestroyProperties(&filter);
      return -1;
    });

    // Add the desired UUID to the filter
    uuid_parse(AFU_ACCEL_UUID_S, guid);
    CHECK_FPGA_ERR(api_.fpgaPropertiesSetGUID(filter, guid), {
      api_.fpgaDestroyProperties(&filter);
      return -1;
    });

    // Do the search across the available FPGA contexts
    CHECK_FPGA_ERR(api_.fpgaEnumerate(&filter, 1, &accel_token, 1, &num_matches), {
      api_.fpgaDestroyProperties(&filter);
      return -1;
    });

    // Not needed anymore
    CHECK_FPGA_ERR(api_.fpgaDestroyProperties(&filter), {
      api_.fpgaDestroyToken(&accel_token);
      return -1;
    });

    if (num_matches < 1) {
      fprintf(stderr, "[VXDRV] Error: accelerator %s not found!\n", AFU_ACCEL_UUID_S);
      api_.fpgaDestroyToken(&accel_token);
      return -1;
    }

    // Open accelerator
    CHECK_FPGA_ERR(api_.fpgaOpen(accel_token, &fpga_, 0), {
      api_.fpgaDestroyToken(&accel_token);
      return -1;
    });

    // Done with token
    CHECK_FPGA_ERR(api_.fpgaDestroyToken(&accel_token), {
      api_.fpgaClose(fpga_);
      return -1;
    });

  #ifdef SCOPE
    {
      scope_callback_t callback;
      callback.registerWrite = [](vx_device_h hdevice, uint64_t value) -> int {
        auto device = (vx_device *)hdevice;
        return device->api_.fpgaWriteMMIO64(device->fpga_, 0, MMIO_SCOPE_WRITE, value);
      };

      callback.registerRead = [](vx_device_h hdevice, uint64_t *value) -> int {
        auto device = (vx_device *)hdevice;
        return device->api_.fpgaReadMMIO64(device->fpga_, 0, MMIO_SCOPE_READ, value);
      };

      CHECK_ERR(vx_scope_start(&callback, this, -1, -1), {
        api_.fpgaClose(fpga_);
        return err;
      });
    }
  #endif

    return 0;
  }

  // ----- CP register channel -----
  // The AFU's MMIO demux routes host byte offsets 0x1000..0x1FFF to the CP
  // regfile (CP-internal 0x000-based offsets). Callers pass the CP-internal
  // offset; we add the AFU base here.
  int cp_reg_write(uint32_t off, uint32_t value) {
    CHECK_FPGA_ERR(api_.fpgaWriteMMIO64(fpga_, 0, CP_BASE + off, value), {
      return -1;
    });
    return 0;
  }

  int cp_reg_read(uint32_t off, uint32_t* value) {
    uint64_t v = 0;
    CHECK_FPGA_ERR(api_.fpgaReadMMIO64(fpga_, 0, CP_BASE + off, &v), {
      return -1;
    });
    *value = uint32_t(v);
    return 0;
  }

  // ----- CP-visible host memory (command ring + DMA staging) -----
  // A CCI-P-shared host buffer reached by the CP's CCI-P host bridge.
  // fpgaPrepareBuffer hands back both the host VA and the IO address.
  int host_mem_alloc(uint64_t size, void** host_ptr, uint64_t* cp_addr) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    void*    ptr   = nullptr;
    uint64_t wsid  = 0, ioaddr = 0;
    CHECK_FPGA_ERR(api_.fpgaPrepareBuffer(fpga_, asize, &ptr, &wsid, 0), {
      return -1;
    });
    CHECK_FPGA_ERR(api_.fpgaGetIOAddress(fpga_, wsid, &ioaddr), {
      api_.fpgaReleaseBuffer(fpga_, wsid);
      return -1;
    });
    host_bos_[ioaddr] = host_bo_t{ ptr, wsid };
    *host_ptr = ptr;
    *cp_addr  = ioaddr;
    return 0;
  }

  int host_mem_free(uint64_t cp_addr) {
    auto it = host_bos_.find(cp_addr);
    if (it == host_bos_.end())
      return -1;
    api_.fpgaReleaseBuffer(fpga_, it->second.wsid);
    host_bos_.erase(it);
    return 0;
  }

private:

  // CCI-P-shared host buffers (CP-visible host memory), keyed by IO address.
  struct host_bo_t {
    void*    ptr;     // host-side mapping
    uint64_t wsid;    // OPAE workspace id
  };
  std::map<uint64_t, host_bo_t> host_bos_;

  opae_drv_api_t api_;
  fpga_handle    fpga_;
};

#include <callbacks.inc>
