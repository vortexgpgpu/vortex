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
// At vx_device_open time, the dispatcher (sw/runtime/stub/vortex.cpp) dlopens
// the backend library named by $VORTEX_DRIVER, resolves vx_dev_init, and
// calls it to populate a callbacks_t with the backend's implementations.
//
// The backend is a pure TRANSPORT HAL. Once the Command Processor is the
// sole command + DMA engine, the backend does no memory management, no DMA
// and no capability decoding — those are the CP's job (CMD_* descriptors)
// or generic common-core code (the device-memory allocator, caps decode).
// The backend provides exactly three things:
//
//   * device lifecycle           — dev_open / dev_close
//   * a register channel to CP   — cp_reg_read / cp_reg_write
//   * CP-visible host memory     — host_mem_alloc / host_mem_free
//
// See docs/proposals/cp_pure_v2_callbacks_proposal.md (Addendum) for the
// rationale. All return values are 0 on success, non-zero on failure.
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
  // The dispatcher wraps it in a vx::Device. All one-time platform setup
  // (device handle, CP-reachable device-memory aperture) happens here.
  int (*dev_open)  (void** out_dev_ctx);
  int (*dev_close) (void*  dev_ctx);

  // ----- CP register channel -----
  // A 32-bit read/write window into the Command Processor regfile — the
  // entire control plane (doorbell via Q_TAIL, status via Q_SEQNUM, and
  // the device/ISA caps window). `off` is the CP-internal regfile offset
  // (matches hw/rtl/cp/VX_cp_axil_regfile.sv); the backend forwards it to
  // its physical MMIO mechanism, applying any platform base itself.
  int (*cp_reg_write)(void* dev_ctx, uint32_t off, uint32_t value);
  int (*cp_reg_read) (void* dev_ctx, uint32_t off, uint32_t* out_value);

  // ----- CP-visible host memory -----
  // Allocates host-resident memory the CP's m_axi_host master can DMA.
  // Returns BOTH a CPU-addressable pointer (out_host_ptr — the runtime
  // memcpy's the command ring / DMA staging through it) and the device-
  // side address the CP uses to reach the same bytes (out_cp_addr).
  // host_mem_free is keyed by that cp_addr. The region must be coherent
  // with the CP's m_axi_host view (no explicit sync callback).
  int (*host_mem_alloc)(void* dev_ctx, uint64_t size,
                        void** out_host_ptr, uint64_t* out_cp_addr);
  int (*host_mem_free) (void* dev_ctx, uint64_t cp_addr);

} callbacks_t;

// Each backend's vortex.cpp implements this function (typically via the
// shared template in <callbacks.inc>) to populate the table.
int vx_dev_init(callbacks_t* callbacks);

#ifdef __cplusplus
}
#endif

#endif // CALLBACKS_H
