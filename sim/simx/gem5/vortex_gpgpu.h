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
// Per gem5_v2_cp_migration_proposal §5.1 the device library hosts a
// vortex::Processor + vortex::CommandProcessor pair, exposes a 32-bit
// CP MMIO regfile (PIO_BASE_ADDR + 0x0 .. + 0x1FF), and provides two
// independently-tickable engines so the SimObject can drive CP and
// Vortex as separate gem5 event chains:
//
//     cpTickEvent_      -> vortex_gem5_cp_tick()
//     vortexTickEvent_  -> vortex_gem5_vortex_tick()
//
// Both engines self-report whether they still have work via
// vortex_gem5_cp_has_work() / vortex_gem5_vortex_busy(); the SimObject
// uses those to decide whether to reschedule. The CP's vortex_start
// hook calls back into the SimObject via the start-handler registered
// at construction so a CMD_LAUNCH retirement schedules vortexTickEvent_
// from inside cpTickEvent_'s execution.
//
// The ABI is C — not C++ — so the gem5 side does not depend on SimX's
// internal types and can be rebuilt against a new gem5 release without
// touching anything Vortex-side.
//
// Concurrency: all calls are serialized on the gem5 event-loop thread.
// No internal locking. No re-entrancy.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle. Owns a vortex::Processor, RAM, MemoryAllocator, and
// vortex::CommandProcessor.
typedef struct vortex_gem5_device_s* vortex_gem5_handle_t;

// Returns a printable description of the build config (cores, warps,
// threads, VX_CFG_XLEN). Returned pointer is static; do not free.
const char* vortex_gem5_build_info(void);

// Construct a Vortex device instance. Returns NULL on failure.
// VRAM is allocated lazily; no kernel is loaded until
// vortex_gem5_load_kernel is called.
vortex_gem5_handle_t vortex_gem5_create(void);

// Destroy the device. Safe to call with NULL.
void vortex_gem5_destroy(vortex_gem5_handle_t h);

// Register a callback the device library invokes from inside its CP
// vortex_start hook. The SimObject uses this to schedule its Vortex
// tick event when the CP launches a kernel. Pass NULL to clear.
// `ctx` is forwarded back unchanged.
typedef void (*vortex_gem5_start_handler_t)(void* ctx);
void vortex_gem5_set_start_handler(vortex_gem5_handle_t h,
                                   vortex_gem5_start_handler_t fn,
                                   void* ctx);

// Load a kernel image into VRAM. Accepts .vxbin / .bin / .hex (same
// shape as sim/simx/main.cpp). Primes the KMU DCRs for a 1×1×1 CTA
// at VX_CFG_STARTUP_ADDR for the Phase 3 standalone test path (in hosted
// mode the dispatcher uploads kernels via mem_upload + programs KMU
// DCRs via CMD_DCR_WRITE through the CP).
//
// Returns 0 on success, -1 on file-not-found or unsupported format.
int vortex_gem5_load_kernel(vortex_gem5_handle_t h, const char* path);

// CP regfile MMIO. `off` is the CP-internal byte offset (0..0x13F for
// queue 0; see sim/common/cmd_processor.h §address map). All
// accesses are 32-bit. The SimObject translates a PIO packet at
// `PIO_BASE_ADDR + off` into one of these calls; the host runtime's
// cp_mmio_{write,read} translates `cp_mmio_write(off, v)` to one of
// these via a 32-bit PIO write at `PIO_BASE_ADDR + off` (no AFU bit-12
// split — the gem5 device's PIO range IS the CP regfile).
void     vortex_gem5_cp_mmio_write(vortex_gem5_handle_t h,
                                   uint32_t off, uint32_t value);
uint32_t vortex_gem5_cp_mmio_read (vortex_gem5_handle_t h, uint32_t off);

// Advance the embedded CommandProcessor by one functional cycle.
// Returns true if the CP has more work (ring non-empty, command in
// flight) and should be ticked again.
bool vortex_gem5_cp_tick(vortex_gem5_handle_t h);

// True iff the CP would benefit from being ticked: enabled and busy.
// The SimObject uses this from PIO write handlers (after a CP regfile
// update may have armed work) to decide whether to schedule
// cpTickEvent_.
bool vortex_gem5_cp_has_work(vortex_gem5_handle_t h);

// Advance the Vortex Processor by one cycle. Returns true while the
// processor is still running (clusters active or channels carrying
// packets); the SimObject's vortexTickEvent_ reschedules itself while
// this returns true and stops otherwise.
bool vortex_gem5_vortex_tick(vortex_gem5_handle_t h);

// True iff Vortex is currently executing a kernel (any cluster
// running, any in-flight memory transactions). Used by the CP's
// vortex_busy hook to know when to retire a CMD_LAUNCH.
bool vortex_gem5_vortex_busy(vortex_gem5_handle_t h);

// Direct device-VRAM access used by the SimObject's DMA-path scratch
// buffers in v1 (a peer of the host runtime, ACL-bypassed). v2 will
// route both Vortex memory and CP DMA through gem5's memory hierarchy
// via the same DevMemAccessor interface.
void vortex_gem5_vram_write(vortex_gem5_handle_t h,
                            uint64_t dev_addr, const uint8_t* src,
                            uint32_t size);
void vortex_gem5_vram_read (vortex_gem5_handle_t h,
                            uint64_t dev_addr, uint8_t* dst,
                            uint32_t size);

#ifdef __cplusplus
} // extern "C"
#endif
