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
// vortex2.h — minimal async runtime for the Vortex Command Processor.
//
// Canonical Vortex runtime API. Provides device/queue/buffer/event handles
// with refcounted lifecycle, asynchronous command submission, OpenCL-shaped
// events with wait lists, and per-command profiling timestamps.
//
// All upper-layer translators (POCL, chipStar, future Vulkan/CUDA/HIP/
// Metal/OpenGL) should target this API directly. The legacy synchronous
// API is a thin wrapper layered on top of these entry points.
// ============================================================================

#ifndef __VX_VORTEX2_H__
#define __VX_VORTEX2_H__

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

// This public header is deliberately self-contained: it includes
// standard C headers ONLY — never VX_config.h or any other Vortex
// build-time header. Hardware configuration is discovered at runtime
// via vx_device_query(); nothing here depends on the build config.

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Opaque handles
// ============================================================================

// Device + buffer handles are void* for ABI stability across runtime builds.
typedef void* vx_device_h;
typedef void* vx_buffer_h;

typedef struct vx_queue*  vx_queue_h;
typedef struct vx_event*  vx_event_h;
typedef struct vx_module* vx_module_h;
typedef struct vx_kernel* vx_kernel_h;

// ============================================================================
// Device capability IDs  (vx_device_query)
// ============================================================================

#define VX_CAPS_VERSION             0x0   // implementation version
#define VX_CAPS_NUM_THREADS         0x1   // number of threads per warp
#define VX_CAPS_NUM_WARPS           0x2   // number of warps per core
#define VX_CAPS_NUM_CORES           0x3   // number of total cores
#define VX_CAPS_CACHE_LINE_SIZE     0x4   // cache line size in bytes
#define VX_CAPS_GLOBAL_MEM_SIZE     0x5   // global memory size in bytes
#define VX_CAPS_LOCAL_MEM_SIZE      0x6   // local memory size per core in bytes
#define VX_CAPS_ISA_FLAGS           0x7   // device ISA flags
#define VX_CAPS_NUM_MEM_BANKS       0x8   // number of memory banks
#define VX_CAPS_MEM_BANK_SIZE       0x9   // memory bank size in bytes
#define VX_CAPS_NUM_CLUSTERS        0xA   // number of clusters
#define VX_CAPS_SOCKET_SIZE         0xB   // number of cores per socket
#define VX_CAPS_ISSUE_WIDTH         0xC   // issue width per core
#define VX_CAPS_CLOCK_RATE          0xD   // pipeline clock rate in MHz
#define VX_CAPS_PEAK_MEM_BW         0xE   // peak memory bandwidth (MB/s)
#define VX_CAPS_VM_SUPPORT          0xF   // 1 if the device has an MMU (VM), else 0
#define VX_CAPS_VM_PINNED_SIZE      0x10  // pinned-region total size (bytes); 0 if disabled
#define VX_CAPS_VM_PINNED_FREE      0x11  // pinned-region free  size (bytes); 0 if disabled

// ============================================================================
// Device ISA flags  (decode a VX_CAPS_ISA_FLAGS query result)
// ============================================================================

// Standard-extension flags — bit positions are the RISC-V `misa`
// register layout, fixed by the ISA spec. Inlined as literals so this
// public header stays self-contained (must match VX_CFG_MISA_STD in
// VX_config.toml).
#define VX_ISA_STD_A                (1ull << 0)
#define VX_ISA_STD_C                (1ull << 2)
#define VX_ISA_STD_D                (1ull << 3)
#define VX_ISA_STD_E                (1ull << 4)
#define VX_ISA_STD_F                (1ull << 5)
#define VX_ISA_STD_H                (1ull << 7)
#define VX_ISA_STD_I                (1ull << 8)
#define VX_ISA_STD_N                (1ull << 13)
#define VX_ISA_STD_Q                (1ull << 16)
#define VX_ISA_STD_S                (1ull << 18)
#define VX_ISA_STD_V                (1ull << 21)
#define VX_ISA_ARCH(flags)          (1ull << (((flags >> 30) & 0x3) + 4))
// Custom-extension flags — Vortex `misa` custom field, bits 32+ (must
// match VX_CFG_MISA_EXT in VX_config.toml).
#define VX_ISA_EXT_ICACHE           (1ull << (32 + 0))
#define VX_ISA_EXT_DCACHE           (1ull << (32 + 1))
#define VX_ISA_EXT_L2CACHE          (1ull << (32 + 2))
#define VX_ISA_EXT_L3CACHE          (1ull << (32 + 3))
#define VX_ISA_EXT_LMEM             (1ull << (32 + 4))
#define VX_ISA_EXT_ZICOND           (1ull << (32 + 5))
#define VX_ISA_EXT_TEX              (1ull << (32 + 6))
#define VX_ISA_EXT_RASTER           (1ull << (32 + 7))
#define VX_ISA_EXT_OM               (1ull << (32 + 8))
#define VX_ISA_EXT_TCU              (1ull << (32 + 9))
#define VX_ISA_EXT_DXA              (1ull << (32 + 10))
#define VX_ISA_EXT_RTU              (1ull << (32 + 11))

// ============================================================================
// Device memory access flags  (vx_buffer_create / vx_buffer_access)
// ============================================================================

#define VX_MEM_READ                 0x1
#define VX_MEM_WRITE                0x2
#define VX_MEM_READ_WRITE           0x3
#define VX_MEM_PIN_MEMORY           0x4
// Allocation returns a PHYSICAL device address (no VA translation).
#define VX_MEM_PHYS                 0x8
// Allocate the buffer in host memory (the platform slave-bridge / Host
// Memory Access aperture) so the Command Processor's host-memory master
// (m_axi_host) can reach it — used for the CP command ring and for
// host<->device DMA staging. Backends without a real host/device memory
// split (simx, rtlsim, gem5) ignore this flag.
#define VX_MEM_HOST                 0x10

// ============================================================================
// Result type
// ============================================================================

typedef enum {
    VX_SUCCESS                  = 0,
    VX_ERR_INVALID_HANDLE       = 1,
    VX_ERR_INVALID_INFO         = 2,
    VX_ERR_INVALID_VALUE        = 3,
    VX_ERR_OUT_OF_HOST_MEMORY   = 4,
    VX_ERR_OUT_OF_DEVICE_MEMORY = 5,
    VX_ERR_DEVICE_LOST          = 6,
    VX_ERR_TIMEOUT              = 7,
    VX_ERR_EVENT_FAILED         = 8,
    VX_ERR_NOT_SUPPORTED        = 9,
    VX_ERR_INTERNAL             = 10
} vx_result_t;

const char* vx_result_string(vx_result_t r);

// ============================================================================
// Enums
// ============================================================================

typedef enum {
    VX_QUEUE_PRIORITY_LOW    = 0,
    VX_QUEUE_PRIORITY_NORMAL = 1,
    VX_QUEUE_PRIORITY_HIGH   = 2
} vx_queue_priority_e;

// ============================================================================
// Macros
// ============================================================================

#define VX_QUEUE_PROFILING_ENABLE  (1u << 0)

// Timeout sentinel — wait forever.
#define VX_TIMEOUT_INFINITE        ((uint64_t)-1)

// ============================================================================
// Versioned create-info structs
// ============================================================================

typedef struct {
    size_t              struct_size;
    const void*         next;
    vx_queue_priority_e priority;
    uint32_t            flags;
} vx_queue_info_t;

typedef struct {
    size_t       struct_size;
    const void*  next;
    // Kernel entry point (vx_module_get_kernel). NULL is the legacy escape
    // hatch — the caller is expected to have programmed the KMU PC DCRs
    // itself via prior vx_dcr_write calls.
    vx_kernel_h  kernel;
    // Kernel argument block as a host-side blob. The runtime stages it into
    // a device-side scratch slot at launch time and programs the KMU ARG
    // pointer — callers no longer allocate/upload/free an args device
    // buffer. Buffers passed as kernel args appear as their uint64_t
    // device addresses inline in the blob (see vx_buffer_address).
    //
    // args_host may be NULL (args_size 0) — the legacy escape hatch: the
    // caller is expected to have programmed the ARG DCRs itself via prior
    // vx_dcr_write calls (matches the ndim==0 convention).
    const void*  args_host;
    size_t       args_size;
    uint32_t     ndim;            // 1, 2, or 3 (0 = legacy escape hatch)
    uint32_t     grid_dim [3];
    uint32_t     block_dim[3];
    uint32_t     lmem_size;
    uint32_t     cluster_dim[3];
} vx_launch_info_t;

// 3D-strided DMA descriptor for the rect enqueue ops. Mirrors OpenCL 1.2's
// clEnqueue{Read,Write,Copy}BufferRect parameter block.
//
//   region[0]       bytes per row (the contiguous run)
//   region[1]       rows per slice
//   region[2]       slices
//   *_origin        start position: [0] in bytes, [1] in rows, [2] in slices
//   *_row_pitch     byte stride between rows   (0 -> region[0])
//   *_slice_pitch   byte stride between slices (0 -> region[1] * row_pitch)
//
// For read_rect / write_rect the `buffer_*` fields describe the device
// buffer and the `host_*` fields the host memory. For copy_rect the
// `buffer_*` fields describe the destination buffer and the `host_*`
// fields the source buffer.
typedef struct {
    size_t       struct_size;
    const void*  next;
    size_t       buffer_origin[3];
    size_t       host_origin  [3];
    size_t       region       [3];
    size_t       buffer_row_pitch;
    size_t       buffer_slice_pitch;
    size_t       host_row_pitch;
    size_t       host_slice_pitch;
} vx_rect_info_t;

typedef struct {
    uint64_t queued_ns;
    uint64_t submit_ns;
    uint64_t start_ns;
    uint64_t end_ns;
} vx_profile_info_t;

// ============================================================================
// Device
// ============================================================================

vx_result_t vx_device_count       (uint32_t* out_count);
vx_result_t vx_device_open        (uint32_t index, vx_device_h* out);
vx_result_t vx_device_retain      (vx_device_h dev);
vx_result_t vx_device_release     (vx_device_h dev);
vx_result_t vx_device_query       (vx_device_h dev, uint32_t caps_id,
                                   uint64_t* out_value);
vx_result_t vx_device_memory_info (vx_device_h dev,
                                   uint64_t* free, uint64_t* used);

// Dump the formatted MPM performance-counter report (per core / cluster /
// cache) to `stream` (NULL -> stdout). Controlled by the VORTEX_PROFILING
// environment variable, same as the legacy vx_dump_perf.
vx_result_t vx_device_dump_perf   (vx_device_h dev, FILE* stream);

// Compute the maximum-occupancy block / grid for `global_dim` work
// items on this device. block[i] = device's natural per-warp / per-
// core dimension (num_threads, num_warps, 1); grid[i] = ceil(global / block).
// `block_out` and `grid_out` must both be at least `ndim` elements.
vx_result_t vx_device_max_occupancy_grid (vx_device_h dev, uint32_t ndim,
                                          const uint32_t* global_dim,
                                          uint32_t* grid_out,
                                          uint32_t* block_out);

// ============================================================================
// Buffer
// ============================================================================

vx_result_t vx_buffer_create  (vx_device_h dev, uint64_t size, uint32_t flags,
                               vx_buffer_h* out);
vx_result_t vx_buffer_reserve (vx_device_h dev, uint64_t address,
                               uint64_t size, uint32_t flags,
                               vx_buffer_h* out);

vx_result_t vx_buffer_retain  (vx_buffer_h buf);
vx_result_t vx_buffer_release (vx_buffer_h buf);
vx_result_t vx_buffer_address (vx_buffer_h buf, uint64_t* out_addr);
vx_result_t vx_buffer_access  (vx_buffer_h buf, uint64_t offset,
                               uint64_t size, uint32_t flags);
vx_result_t vx_buffer_map     (vx_buffer_h buf, uint64_t offset, uint64_t size,
                               uint32_t flags, void** out_host_ptr);
vx_result_t vx_buffer_unmap   (vx_buffer_h buf, void* host_ptr);

// ============================================================================
// Module + kernel
//
// Load a .vxbin as a module, then resolve named entry points to vx_kernel_h
// handles. Matches CUDA cuModule/cuFunction, HIP hipModule/hipFunction,
// Metal MTLLibrary/MTLFunction, Vulkan VkShaderModule + entry name.
//
// A vx_kernel_h is the only handle accepted by vx_launch_info_t.kernel.
// ============================================================================

vx_result_t vx_module_load_file  (vx_device_h dev, const char* path,
                                  vx_module_h* out);
vx_result_t vx_module_load_bytes (vx_device_h dev, const void* bytes,
                                  size_t size, vx_module_h* out);
vx_result_t vx_module_retain     (vx_module_h mod);
vx_result_t vx_module_release    (vx_module_h mod);

vx_result_t vx_module_get_kernel (vx_module_h mod, const char* name,
                                  vx_kernel_h* out);
vx_result_t vx_kernel_retain     (vx_kernel_h k);
vx_result_t vx_kernel_release    (vx_kernel_h k);

// Returns the device's natural block dims as a starting point.
// Per-kernel compiler metadata in the .vxbin symbol footer will refine
// this when available.
vx_result_t vx_kernel_get_max_block_size (vx_kernel_h k,
                                          uint32_t* x, uint32_t* y,
                                          uint32_t* z);

// ============================================================================
// Queue
// ============================================================================

vx_result_t vx_queue_create   (vx_device_h dev, const vx_queue_info_t* info,
                               vx_queue_h* out);
vx_result_t vx_queue_retain   (vx_queue_h q);
vx_result_t vx_queue_release  (vx_queue_h q);
vx_result_t vx_queue_flush    (vx_queue_h q);
vx_result_t vx_queue_finish   (vx_queue_h q, uint64_t timeout_ns);

// ============================================================================
// Async enqueue
//
// Every enqueue takes a wait-list and returns an event for the work just
// submitted. out_event may be NULL if the caller does not need to observe
// completion of this particular command.
// ============================================================================

vx_result_t vx_enqueue_launch    (vx_queue_h q,
                                  const vx_launch_info_t* info,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_copy      (vx_queue_h q,
                                  vx_buffer_h dst, uint64_t dst_off,
                                  vx_buffer_h src, uint64_t src_off,
                                  uint64_t    size,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_read      (vx_queue_h q,
                                  void* host_dst,
                                  vx_buffer_h src, uint64_t src_off,
                                  uint64_t    size,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_write     (vx_queue_h q,
                                  vx_buffer_h dst, uint64_t dst_off,
                                  const void* host_src,
                                  uint64_t    size,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

// Strided 3D DMA. Decomposes the rect into per-row linear transfers
// (a single transfer when the rect is fully contiguous).
vx_result_t vx_enqueue_read_rect (vx_queue_h q,
                                  void* host_dst,
                                  vx_buffer_h src,
                                  const vx_rect_info_t* info,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_write_rect(vx_queue_h q,
                                  vx_buffer_h dst,
                                  const void* host_src,
                                  const vx_rect_info_t* info,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_copy_rect (vx_queue_h q,
                                  vx_buffer_h dst,
                                  vx_buffer_h src,
                                  const vx_rect_info_t* info,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

// Fill `size` bytes of `dst` at `offset` with `pattern` (pattern_size
// bytes) repeated. `size` must be a whole multiple of `pattern_size`.
vx_result_t vx_enqueue_fill_buffer(vx_queue_h q,
                                   vx_buffer_h dst,
                                   uint64_t offset, uint64_t size,
                                   const void* pattern, size_t pattern_size,
                                   uint32_t          n_wait_events,
                                   const vx_event_h* wait_events,
                                   vx_event_h*       out_event);

// Async buffer map / unmap. vx_enqueue_map allocates a host-accessible
// staging region and returns its pointer synchronously in *out_host_ptr;
// once the wait list resolves the worker populates it from the device
// (for VX_MEM_READ maps). The pointer is valid for access only after
// out_event signals. vx_enqueue_unmap flushes a VX_MEM_WRITE mapping back
// to the device and frees the staging region.
vx_result_t vx_enqueue_map       (vx_queue_h q, vx_buffer_h buf,
                                  uint64_t offset, uint64_t size,
                                  uint32_t flags,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event,
                                  void**            out_host_ptr);

vx_result_t vx_enqueue_unmap     (vx_queue_h q, vx_buffer_h buf,
                                  void* host_ptr,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_barrier   (vx_queue_h q,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_dcr_write (vx_queue_h q,
                                  uint32_t addr, uint32_t value,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_dcr_read  (vx_queue_h q,
                                  uint32_t addr, uint32_t* host_dst,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

// ============================================================================
// Events
//
// Counter-based timeline events (Vulkan VkTimelineSemaphore / Metal
// MTLSharedEvent.value / CUDA cuStreamWaitValue64 shape). Each event carries
// a monotonically-increasing uint64_t value; signal advances it, wait blocks
// until value >= target. The binary case (signal once, wait once) is just
// "signal value=1, wait value=1" — which is what every vx_enqueue_* op
// returns: a fresh event the worker advances to 1 on completion.
// ============================================================================

// Create a timeline event. The counter starts at 0.
vx_result_t vx_event_create        (vx_device_h dev, vx_event_h* out);

// Host-side signal — advance the event's counter to max(current, value).
// Idempotent; never decrements.
vx_result_t vx_event_signal        (vx_event_h ev, uint64_t value);

// Host-side queries.
vx_result_t vx_event_get_value     (vx_event_h ev, uint64_t* out_value);
vx_result_t vx_event_wait_value    (vx_event_h ev, uint64_t value,
                                    uint64_t timeout_ns);
vx_result_t vx_event_wait_values   (uint32_t n,
                                    const vx_event_h* evs,
                                    const uint64_t*   values,
                                    uint64_t timeout_ns);

vx_result_t vx_event_retain        (vx_event_h ev);
vx_result_t vx_event_release       (vx_event_h ev);

vx_result_t vx_event_get_profiling (vx_event_h ev, vx_profile_info_t* out);

// ============================================================================
// Queue-ordered timeline signal / wait
//
// vx_enqueue_signal:    after the queue's prior work completes (including
//                       wait_events), advance `ev`'s counter to `value`.
// vx_enqueue_wait_value: subsequent queue ops block until `ev`'s counter
//                       reaches `value`.
//
// Both return their own completion event (out_event), which signals when
// the signal/wait op itself retires.
// ============================================================================

vx_result_t vx_enqueue_signal      (vx_queue_h q, vx_event_h ev,
                                    uint64_t value,
                                    uint32_t n_wait_events,
                                    const vx_event_h* wait_events,
                                    vx_event_h* out_event);

vx_result_t vx_enqueue_wait_value  (vx_queue_h q, vx_event_h ev,
                                    uint64_t value,
                                    uint32_t n_wait_events,
                                    const vx_event_h* wait_events,
                                    vx_event_h* out_event);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __VX_VORTEX2_H__
