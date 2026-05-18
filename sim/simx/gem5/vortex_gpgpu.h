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

// libvortex-gem5 — C ABI for the gem5 VortexGPGPU SimObject.
//
// The gem5 device (sim/simx/gem5/<simobject>.cc, installed into a pinned
// gem5 tree by sim/simx/gem5/install.sh) loads this shared library and
// drives it through this C ABI. Keeping the ABI in C — not C++ — means
// the gem5 side does not depend on SimX's C++ types and can be rebuilt
// against a new gem5 release without touching anything Vortex-side.
//
// Concurrency: the gem5 device serializes calls on its event-loop thread;
// no internal locking. Re-entrancy: completion callbacks (currently
// unused — the DMA path is fully synchronous on the gem5 side per Phase
// 2) may be added later as Phase 3 wires up async DMA.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle. The library owns a vortex::Processor + RAM behind it.
typedef struct vortex_gem5_device_s* vortex_gem5_handle_t;

// Returns a printable description of the build config (cores, warps,
// threads, XLEN). Returned pointer is static; do not free.
const char* vortex_gem5_build_info(void);

// Construct a Vortex device instance. Returns NULL on failure.
// VRAM is allocated lazily; no kernel is loaded until
// vortex_gem5_load_kernel is called.
vortex_gem5_handle_t vortex_gem5_create(void);

// Destroy the device. Safe to call with NULL.
void vortex_gem5_destroy(vortex_gem5_handle_t h);

// Load a kernel image into VRAM. Accepts .vxbin / .bin / .hex (same
// shape as sim/simx/main.cpp:120). Primes the KMU DCRs for a 1x1x1
// CTA at STARTUP_ADDR (same as sim/simx/main.cpp:101-116) so a
// subsequent cycle() loop launches the kernel.
//
// In the Phase-2 in-process smoke driver this is how kernels reach
// the device. The Phase-4 runtime will instead upload kernels via
// the staging-buffer DMA path (vortex_gem5_vram_write + the OPAE MMIO
// commands), and Phase 3's gem5 SimObject can optionally call this
// at boot via a Python `kernel=...` parameter for one-shot smoke
// tests.
//
// Returns 0 on success, -1 on file-not-found or unsupported format.
int vortex_gem5_load_kernel(vortex_gem5_handle_t h, const char* path);

// Advance the simulator by one cycle. Returns true while work
// remains (clusters running or channels carrying packets); false once
// the program has finished. Mirrors vortex::Processor::cycle().
bool vortex_gem5_tick(vortex_gem5_handle_t h);

// MMIO (PIO) accessed by the simulated host CPU via the gem5 SimObject's
// read()/write() callbacks. Offsets are byte addresses inside the
// device's PIO range. See sw/runtime/opae/vortex.cpp for the OPAE MMIO
// layout this protocol mirrors.
uint64_t vortex_gem5_mmio_read64(vortex_gem5_handle_t h, uint64_t offset);
void vortex_gem5_mmio_write64(vortex_gem5_handle_t h, uint64_t offset, uint64_t value);

// VRAM access. The gem5 device DMAs to/from the host's staging buffer
// using its own DmaPort; once the bytes are in a local scratch, it
// calls these to copy into/out of the device VRAM. Bytes here cross
// only the C ABI boundary — they do not re-enter gem5's DMA system.
//
// Bounds-checked against the RAM image; on overflow the call is a
// no-op and (in debug builds) logs to stderr.
void vortex_gem5_vram_write(vortex_gem5_handle_t h, uint64_t dev_addr, const uint8_t* src, uint32_t size);
void vortex_gem5_vram_read(vortex_gem5_handle_t h, uint64_t dev_addr, uint8_t* dst, uint32_t size);

// DCR write/read passthrough. The DCR-read path also handles the
// cache-flush DCR (VX_DCR_BASE_CACHE_FLUSH), which drains dirty cache
// lines all the way to VRAM — required before a host read-back per
// B9 in docs/proposals/gem5_simx_v3_proposal.md §2.2.
int vortex_gem5_dcr_write(vortex_gem5_handle_t h, uint32_t addr, uint32_t value);
int vortex_gem5_dcr_read(vortex_gem5_handle_t h, uint32_t addr, uint32_t tag, uint32_t* value);

// Protocol state introspection for the gem5 SimObject. The library
// owns the OPAE state machine (cmd_args + busy bit + cmd_type +
// dcr_rsp); the gem5 SimObject calls these to drive DMA for the
// async CMD_MEM_{READ,WRITE} commands.
//
// pop_pending_cmd returns the CMD_* constant of an async command
// the SimObject must service (CMD_RUN, CMD_MEM_WRITE, CMD_MEM_READ),
// or 0 if no command is pending. Synchronous commands (CMD_DCR_*)
// are handled inside mmio_write64 and never surface here.
uint64_t vortex_gem5_pop_pending_cmd(vortex_gem5_handle_t h);
uint64_t vortex_gem5_get_cmd_arg(vortex_gem5_handle_t h, int which);
void     vortex_gem5_set_busy(vortex_gem5_handle_t h, bool busy);

#ifdef __cplusplus
} // extern "C"
#endif
