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
// callbacks.h — runtime dispatcher contract between libvortex.so and each
// backend's libvortex-<NAME>.so.
//
// At vx_dev_open time, the dispatcher (sw/runtime/stub/vortex.cpp) dlopens
// the backend library named by $VORTEX_DRIVER, resolves vx_dev_init, and
// calls it to populate a callbacks_t with the backend's implementations.
// All subsequent vortex.h / vortex2.h calls in libvortex.so flow through
// the function pointers in callbacks_t.
//
// The fields below are intentionally Platform-shaped: they operate on
// opaque void* device contexts and raw uint64_t device addresses. The
// dispatcher wraps these primitives into refcounted vx::Device /
// vx::Buffer / vx::Queue / vx::Event objects on top.
// ============================================================================

#ifndef CALLBACKS_H
#define CALLBACKS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {

  // ----- Device lifecycle -----
  // dev_open creates a backend-private device context (returned as void*).
  // The dispatcher wraps it in a vx::Device on its side.
  int (*dev_open)  (void** out_dev_ctx);
  int (*dev_close) (void*  dev_ctx);

  // ----- Capability + heap queries -----
  int (*query_caps)  (void* dev_ctx, uint32_t caps_id, uint64_t* out_value);
  int (*memory_info) (void* dev_ctx, uint64_t* out_free, uint64_t* out_used);

  // ----- Device memory (raw uint64_t addresses; dispatcher wraps in
  //                     vx::Buffer) -----
  int (*mem_alloc)   (void* dev_ctx, uint64_t size, uint32_t flags,
                      uint64_t* out_dev_addr);
  int (*mem_reserve) (void* dev_ctx, uint64_t dev_addr, uint64_t size,
                      uint32_t flags);
  int (*mem_free)    (void* dev_ctx, uint64_t dev_addr);
  int (*mem_access)  (void* dev_ctx, uint64_t dev_addr, uint64_t size,
                      uint32_t flags);

  // ----- DMA primitives (sync; the dispatcher's vx::Queue layer adds the
  //                      async event wrapping on top) -----
  int (*mem_upload)  (void* dev_ctx, uint64_t dst_dev_addr, const void* src,
                      uint64_t size);
  int (*mem_download)(void* dev_ctx, void* dst, uint64_t src_dev_addr,
                      uint64_t size);
  int (*mem_copy)    (void* dev_ctx, uint64_t dst_dev_addr,
                      uint64_t src_dev_addr, uint64_t size);

  // ----- Command Processor control plane (sole control path) -----
  // The `off` argument is the CP-internal regfile offset (matches the
  // VX_cp_axil_regfile address map: globals at 0x000..0xFF, queue 0
  // at 0x100..0x13F). xrt/opae backends translate to their host-side
  // MMIO offset by adding 0x1000 (per the AFU's bit-12 demux split).
  // simx/rtlsim forward directly to a sim/common/CommandProcessor.
  //
  // All kernel launches and DCR ops flow through the dispatcher's
  // CP submission path (sw/runtime/common/vx_device.cpp) which builds
  // CMD_* descriptors, mem_uploads them into the ring, commits Q_TAIL
  // via cp_mmio_write, and polls Q_SEQNUM / Q_LAST_DCR_RSP via
  // cp_mmio_read. Backends have no per-command implementation work.
  int (*cp_mmio_write)(void* dev_ctx, uint32_t off, uint32_t value);
  int (*cp_mmio_read) (void* dev_ctx, uint32_t off, uint32_t* out_value);

} callbacks_t;

// Each backend's vortex.cpp implements this function (typically via the
// shared template in <callbacks.inc>) to populate the table.
int vx_dev_init(callbacks_t* callbacks);

#ifdef __cplusplus
}
#endif

#endif // CALLBACKS_H
