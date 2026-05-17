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
// Legacy synchronous vortex.h is implemented as a thin wrapper over the
// entry points here (see common/vortex_legacy_wrapper.cpp). All upper-layer
// translators (POCL, chipStar, future Vulkan/CUDA/HIP/Metal/OpenGL) should
// target vortex2.h directly.
//
// See docs/proposals/command_processor_proposal.md §8 for the architectural
// design and docs/proposals/cp_runtime_impl_proposal.md for the
// implementation plan.
// ============================================================================

#ifndef __VX_VORTEX2_H__
#define __VX_VORTEX2_H__

#include <vortex.h>      // inherit vx_device_h, vx_buffer_h, VX_CAPS_*, VX_MEM_*
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Opaque handles introduced by vortex2.h
// ============================================================================

typedef struct vx_queue* vx_queue_h;
typedef struct vx_event* vx_event_h;

// (vx_device_h, vx_buffer_h inherited from vortex.h as void* for ABI compat.)

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

typedef enum {
    VX_EVENT_STATUS_QUEUED    = 0,
    VX_EVENT_STATUS_SUBMITTED = 1,
    VX_EVENT_STATUS_RUNNING   = 2,
    VX_EVENT_STATUS_COMPLETE  = 3,
    VX_EVENT_STATUS_ERROR     = 4
} vx_event_status_e;

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
    vx_buffer_h  kernel;          // loaded ELF; entry PC = buffer base
    vx_buffer_h  args;            // kernel argument block
    uint32_t     ndim;            // 1, 2, or 3
    uint32_t     grid_dim [3];
    uint32_t     block_dim[3];
    uint32_t     lmem_size;
} vx_launch_info_t;

typedef struct {
    uint64_t queued_ns;
    uint64_t submit_ns;
    uint64_t start_ns;
    uint64_t end_ns;
} vx_profile_info_t;

// ============================================================================
// Device  (6 functions)
// ============================================================================

vx_result_t vx_device_count       (uint32_t* out_count);
vx_result_t vx_device_open        (uint32_t index, vx_device_h* out);
vx_result_t vx_device_retain      (vx_device_h dev);
vx_result_t vx_device_release     (vx_device_h dev);
vx_result_t vx_device_query       (vx_device_h dev, uint32_t caps_id,
                                   uint64_t* out_value);
vx_result_t vx_device_memory_info (vx_device_h dev,
                                   uint64_t* free, uint64_t* used);

// ============================================================================
// Buffer  (8 functions)
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
// Queue  (5 functions)
// ============================================================================

vx_result_t vx_queue_create   (vx_device_h dev, const vx_queue_info_t* info,
                               vx_queue_h* out);
vx_result_t vx_queue_retain   (vx_queue_h q);
vx_result_t vx_queue_release  (vx_queue_h q);
vx_result_t vx_queue_flush    (vx_queue_h q);
vx_result_t vx_queue_finish   (vx_queue_h q, uint64_t timeout_ns);

// ============================================================================
// Async enqueue  (7 functions)
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
// Events  (7 functions)
// ============================================================================

vx_result_t vx_user_event_create   (vx_device_h dev, vx_event_h* out);
vx_result_t vx_user_event_signal   (vx_event_h ev, vx_result_t status);

vx_result_t vx_event_retain        (vx_event_h ev);
vx_result_t vx_event_release       (vx_event_h ev);

vx_result_t vx_event_status        (vx_event_h ev, vx_event_status_e* out);
vx_result_t vx_event_wait_all      (uint32_t n, const vx_event_h* evs,
                                    uint64_t timeout_ns);
vx_result_t vx_event_get_profiling (vx_event_h ev, vx_profile_info_t* out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __VX_VORTEX2_H__
