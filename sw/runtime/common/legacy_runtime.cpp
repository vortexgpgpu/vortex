// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// legacy_runtime.cpp — vortex.h entry points as thin wrappers over vortex2.h.
// These wrappers never touch callbacks_t directly; they delegate to vortex2.h
// C entry points, which dispatch to the loaded backend via CallbacksAdapter.
// ============================================================================

#include "vortex2_internal.h"
#include "common.h"

#include <VX_types.h>

using namespace vx;

namespace {

inline int to_int(vx_result_t r) {
    return (r == VX_SUCCESS) ? 0 : -1;
}

// Helper: enqueue an operation that produces an event, then wait on it
// synchronously and release the event.
template <typename Fn>
vx_result_t enqueue_and_wait(Device* dev, Fn&& fn) {
    Queue* q = dev->legacy_default_queue();
    if (!q) return VX_ERR_OUT_OF_HOST_MEMORY;
    vx_event_h ev = nullptr;
    auto r = fn(to_handle(q), &ev);
    if (r != VX_SUCCESS) return r;
    if (ev) {
        r = vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
        vx_event_release(ev);
    }
    return r;
}

} // anonymous namespace

// ============================================================================
// Device lifecycle
// ============================================================================

extern "C" int vx_dev_open(vx_device_h* hdevice) {
    if (!hdevice) return -1;
    return to_int(vx_device_open(0, hdevice));
}

extern "C" int vx_dev_close(vx_device_h hdevice) {
    if (!hdevice) return -1;
    // Drain any in-flight legacy launch first so the worker thread does not
    // outlive the device.
    Device* dev = to_device(hdevice);
    if (Event* last = dev->legacy_take_last_event()) {
        last->wait(VX_TIMEOUT_INFINITE);
        last->release();
    }
    return to_int(vx_device_release(hdevice));
}

extern "C" int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id,
                           uint64_t* value) {
    return to_int(vx_device_query(hdevice, caps_id, value));
}

// ============================================================================
// Memory  (vx_mem_* → vx_buffer_* / vx_device_memory_info)
// ============================================================================

extern "C" int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int flags,
                            vx_buffer_h* hbuffer) {
    return to_int(vx_buffer_create(hdevice, size, (uint32_t)flags, hbuffer));
}

extern "C" int vx_mem_reserve(vx_device_h hdevice, uint64_t address,
                              uint64_t size, int flags, vx_buffer_h* hbuffer) {
    return to_int(vx_buffer_reserve(hdevice, address, size,
                                    (uint32_t)flags, hbuffer));
}

extern "C" int vx_mem_free(vx_buffer_h hbuffer) {
    return to_int(vx_buffer_release(hbuffer));
}

extern "C" int vx_mem_access(vx_buffer_h hbuffer, uint64_t offset,
                             uint64_t size, int flags) {
    return to_int(vx_buffer_access(hbuffer, offset, size, (uint32_t)flags));
}

extern "C" int vx_mem_address(vx_buffer_h hbuffer, uint64_t* address) {
    return to_int(vx_buffer_address(hbuffer, address));
}

extern "C" int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free,
                           uint64_t* mem_used) {
    return to_int(vx_device_memory_info(hdevice, mem_free, mem_used));
}

// ============================================================================
// Synchronous DMA  (vx_copy_* → enqueue + wait on default queue)
// ============================================================================

extern "C" int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr,
                              uint64_t dst_offset, uint64_t size) {
    if (!hbuffer) return -1;
    Buffer* buf = to_buffer(hbuffer);
    return to_int(enqueue_and_wait(buf->device(),
        [&](vx_queue_h q, vx_event_h* ev) {
            return vx_enqueue_write(q, hbuffer, dst_offset, host_ptr, size,
                                    0, nullptr, ev);
        }));
}

extern "C" int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer,
                                uint64_t src_offset, uint64_t size) {
    if (!hbuffer) return -1;
    Buffer* buf = to_buffer(hbuffer);
    return to_int(enqueue_and_wait(buf->device(),
        [&](vx_queue_h q, vx_event_h* ev) {
            return vx_enqueue_read(q, host_ptr, hbuffer, src_offset, size,
                                   0, nullptr, ev);
        }));
}

extern "C" int vx_copy_dev_to_dev(vx_buffer_h hdest_buffer, uint64_t dest_offset,
                                  vx_buffer_h hsrc_buffer, uint64_t src_offset,
                                  uint64_t size) {
    if (!hdest_buffer) return -1;
    Buffer* dst = to_buffer(hdest_buffer);
    return to_int(enqueue_and_wait(dst->device(),
        [&](vx_queue_h q, vx_event_h* ev) {
            return vx_enqueue_copy(q, hdest_buffer, dest_offset,
                                   hsrc_buffer, src_offset, size,
                                   0, nullptr, ev);
        }));
}

// ============================================================================
// Kernel launch  (vx_start → vx_enqueue_launch on default queue, async)
//
// vx_start programs the full KMU descriptor and enqueues an async launch
// (grid = num_cores, block = auto warp width). vx_ready_wait blocks on the
// stored event and releases it.
// ============================================================================

extern "C" int vx_start(vx_device_h hdevice, vx_buffer_h hkernel,
                        vx_buffer_h harguments) {
    if (!hdevice || !hkernel || !harguments) return -1;

    Device* dev = to_device(hdevice);
    Buffer* kernel = to_buffer(hkernel);
    Buffer* args   = to_buffer(harguments);

    // Drain any prior in-flight legacy launch (legacy vx_start can be
    // called back-to-back without an interleaved vx_ready_wait).
    if (Event* prev = dev->legacy_take_last_event()) {
        prev->wait(VX_TIMEOUT_INFINITE);
        prev->release();
    }

    uint64_t num_cores = 0, num_threads = 0, num_warps = 0;
    if (vx_device_query(hdevice, VX_CAPS_NUM_CORES,   &num_cores)   != VX_SUCCESS) return -1;
    if (vx_device_query(hdevice, VX_CAPS_NUM_THREADS, &num_threads) != VX_SUCCESS) return -1;
    if (vx_device_query(hdevice, VX_CAPS_NUM_WARPS,   &num_warps)   != VX_SUCCESS) return -1;

    uint32_t eff_block_dim[3];
    uint32_t block_size = 0;
    uint32_t warp_step_x = 0, warp_step_y = 0, warp_step_z = 0;
    prepare_kernel_launch_params((uint32_t)num_threads, (uint32_t)num_warps,
                                 /*ndim=*/1, /*block_dim=*/nullptr, eff_block_dim,
                                 &block_size, &warp_step_x, &warp_step_y, &warp_step_z);

    uint32_t full_grid[3]  = {(uint32_t)num_cores, 1, 1};
    uint32_t full_block[3] = {eff_block_dim[0], 1, 1};
    uint32_t lmem_size     = 0;

    Queue* q = dev->legacy_default_queue();
    if (!q) return -1;

    // DCR writes are fire-and-forget (queue is a strict FIFO); the launch
    // sits behind them and executes in order.
    uint64_t pc   = kernel->dev_address();
    uint64_t argp = args->dev_address();
    struct { uint32_t addr; uint32_t value; } kmu_writes[] = {
        { VX_DCR_KMU_STARTUP_ADDR0, (uint32_t)(pc & 0xffffffffu) },
        { VX_DCR_KMU_STARTUP_ADDR1, (uint32_t)(pc >> 32)         },
        { VX_DCR_KMU_STARTUP_ARG0,  (uint32_t)(argp & 0xffffffffu) },
        { VX_DCR_KMU_STARTUP_ARG1,  (uint32_t)(argp >> 32)        },
        { VX_DCR_KMU_BLOCK_DIM_X,   full_block[0] },
        { VX_DCR_KMU_BLOCK_DIM_Y,   full_block[1] },
        { VX_DCR_KMU_BLOCK_DIM_Z,   full_block[2] },
        { VX_DCR_KMU_GRID_DIM_X,    full_grid[0]  },
        { VX_DCR_KMU_GRID_DIM_Y,    full_grid[1]  },
        { VX_DCR_KMU_GRID_DIM_Z,    full_grid[2]  },
        { VX_DCR_KMU_LMEM_SIZE,     lmem_size     },
        { VX_DCR_KMU_BLOCK_SIZE,    block_size    },
        { VX_DCR_KMU_WARP_STEP_X,   warp_step_x   },
        { VX_DCR_KMU_WARP_STEP_Y,   warp_step_y   },
        { VX_DCR_KMU_WARP_STEP_Z,   warp_step_z   },
        { VX_DCR_KMU_CLUSTER_DIM_X, 1 },
        { VX_DCR_KMU_CLUSTER_DIM_Y, 1 },
        { VX_DCR_KMU_CLUSTER_DIM_Z, 1 },
    };
    for (auto& w : kmu_writes) {
        auto r = vx_enqueue_dcr_write(to_handle(q), w.addr, w.value,
                                      0, nullptr, /*out_event=*/nullptr);
        if (r != VX_SUCCESS) return -1;
    }

    // All KMU DCRs already programmed above; enqueue_launch just triggers CMD_LAUNCH.
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel      = nullptr;
    li.args_host   = nullptr;
    li.args_size   = 0;
    li.ndim        = 0;
    vx_event_h ev = nullptr;
    auto r = vx_enqueue_launch(to_handle(q), &li, 0, nullptr, &ev);
    if (r != VX_SUCCESS) return -1;
    dev->legacy_remember_last_event(to_event(ev));
    return 0;
}

extern "C" int vx_ready_wait(vx_device_h hdevice, uint64_t timeout_ms) {
    if (!hdevice) return -1;
    Device* dev = to_device(hdevice);
    Event* ev = dev->legacy_take_last_event();
    if (!ev) return 0;   // nothing pending
    uint64_t timeout_ns = (timeout_ms == (uint64_t)-1)
                            ? VX_TIMEOUT_INFINITE
                            : timeout_ms * 1'000'000ull;
    auto r = ev->wait(timeout_ns);
    ev->release();
    return to_int(r);
}

// ============================================================================
// DCR  (vx_dcr_* → vx_enqueue_dcr_* on default queue + wait)
// ============================================================================

extern "C" int vx_dcr_write(vx_device_h hdevice, uint32_t addr,
                            uint32_t value) {
    if (!hdevice) return -1;
    Device* dev = to_device(hdevice);
    return to_int(enqueue_and_wait(dev,
        [&](vx_queue_h q, vx_event_h* ev) {
            return vx_enqueue_dcr_write(q, addr, value, 0, nullptr, ev);
        }));
}

extern "C" int vx_dcr_read(vx_device_h hdevice, uint32_t addr, uint32_t tag,
                           uint32_t* value) {
    if (!hdevice) return -1;
    // `tag` packs mpm_class+csr_id+core_id onto the DCR bus; vortex2's
    // enqueue_dcr_read does not surface it, so submit directly via the CP.
    Device* dev = to_device(hdevice);
    return to_int(dev->cp_submit_dcr_read(addr, tag, value));
}
