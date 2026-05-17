// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "vortex2_internal.h"

#include <VX_config.h>
#include <VX_types.h>

#include <thread>

namespace vx {

Queue::Queue(Device* dev, const vx_queue_info_t& info)
    : device_(dev),
      priority_(static_cast<uint32_t>(info.priority)),
      flags_(info.flags) {
    device_->retain();
    device_->register_queue(this);
}

Queue::~Queue() {
    if (device_) {
        device_->unregister_queue(this);
        device_->release();
    }
}

vx_result_t Queue::create(Device* dev, const vx_queue_info_t* info,
                          Queue** out) {
    if (!dev || !out) return VX_ERR_INVALID_VALUE;
    vx_queue_info_t default_info = {};
    default_info.struct_size = sizeof(default_info);
    default_info.priority    = VX_QUEUE_PRIORITY_NORMAL;
    default_info.flags       = 0;
    if (!info) info = &default_info;
    if (info->struct_size < sizeof(vx_queue_info_t)) return VX_ERR_INVALID_INFO;
    *out = new Queue(dev, *info);
    return VX_SUCCESS;
}

vx_result_t Queue::wait_on_externals(uint32_t nw, const vx_event_h* w) {
    if (nw != 0 && !w) return VX_ERR_INVALID_VALUE;
    for (uint32_t i = 0; i < nw; ++i) {
        if (!w[i]) return VX_ERR_INVALID_HANDLE;
        auto r = to_event(w[i])->wait(VX_TIMEOUT_INFINITE);
        if (r != VX_SUCCESS) return r;
    }
    return VX_SUCCESS;
}

Event* Queue::bind_event(uint64_t queued_ns, uint64_t submit_ns,
                         uint64_t start_ns, uint64_t end_ns) {
    // Synchronous (non-launch) enqueue: the work has already completed by
    // the time bind_event is called. Create an internal event, fill its
    // profile, and mark it complete immediately.
    Event* ev = nullptr;
    if (Event::create(device_, &ev) != VX_SUCCESS) return nullptr;
    if (profiling_enabled()) {
        ev->set_profile(queued_ns, submit_ns, start_ns, end_ns);
    }
    ev->complete(VX_SUCCESS);
    return ev;
}

vx_result_t Queue::flush() {
    // No-op in v1 pre-CP — every enqueue completes synchronously, so the
    // doorbell pattern doesn't apply yet.
    return VX_SUCCESS;
}

vx_result_t Queue::finish(uint64_t timeout_ns) {
    // No-op in v1 pre-CP — every enqueue is already complete on return.
    (void)timeout_ns;
    return VX_SUCCESS;
}

vx_result_t Queue::enqueue_write(Buffer* dst, uint64_t off, const void* host,
                                 uint64_t sz, uint32_t nw,
                                 const vx_event_h* w, vx_event_h* out) {
    if (!dst || (!host && sz != 0)) return VX_ERR_INVALID_VALUE;
    if (off + sz > dst->size())     return VX_ERR_INVALID_VALUE;

    uint64_t queued_ns = now_ns();
    auto r = wait_on_externals(nw, w);
    if (r != VX_SUCCESS) return r;

    uint64_t submit_ns = now_ns();
    uint64_t start_ns  = submit_ns;
    {
        std::lock_guard<std::mutex> g(enqueue_mu_);
        r = device_->platform()->mem_upload(dst->dev_address() + off,
                                            host, sz);
    }
    if (r != VX_SUCCESS) return r;
    uint64_t end_ns = now_ns();

    if (out) {
        Event* ev = bind_event(queued_ns, submit_ns, start_ns, end_ns);
        if (!ev) return VX_ERR_OUT_OF_HOST_MEMORY;
        *out = to_handle(ev);
    }
    return VX_SUCCESS;
}

vx_result_t Queue::enqueue_read(void* host, Buffer* src, uint64_t so,
                                uint64_t sz, uint32_t nw,
                                const vx_event_h* w, vx_event_h* out) {
    if (!src || (!host && sz != 0)) return VX_ERR_INVALID_VALUE;
    if (so + sz > src->size())      return VX_ERR_INVALID_VALUE;

    uint64_t queued_ns = now_ns();
    auto r = wait_on_externals(nw, w);
    if (r != VX_SUCCESS) return r;

    uint64_t submit_ns = now_ns();
    uint64_t start_ns  = submit_ns;
    {
        std::lock_guard<std::mutex> g(enqueue_mu_);
        r = device_->platform()->mem_download(host,
                                              src->dev_address() + so, sz);
    }
    if (r != VX_SUCCESS) return r;
    uint64_t end_ns = now_ns();

    if (out) {
        Event* ev = bind_event(queued_ns, submit_ns, start_ns, end_ns);
        if (!ev) return VX_ERR_OUT_OF_HOST_MEMORY;
        *out = to_handle(ev);
    }
    return VX_SUCCESS;
}

vx_result_t Queue::enqueue_copy(Buffer* dst, uint64_t do_, Buffer* src,
                                uint64_t so, uint64_t sz, uint32_t nw,
                                const vx_event_h* w, vx_event_h* out) {
    if (!dst || !src)               return VX_ERR_INVALID_VALUE;
    if (do_ + sz > dst->size())     return VX_ERR_INVALID_VALUE;
    if (so + sz > src->size())      return VX_ERR_INVALID_VALUE;

    uint64_t queued_ns = now_ns();
    auto r = wait_on_externals(nw, w);
    if (r != VX_SUCCESS) return r;

    uint64_t submit_ns = now_ns();
    uint64_t start_ns  = submit_ns;
    {
        std::lock_guard<std::mutex> g(enqueue_mu_);
        r = device_->platform()->mem_copy(dst->dev_address() + do_,
                                          src->dev_address() + so, sz);
    }
    if (r != VX_SUCCESS) return r;
    uint64_t end_ns = now_ns();

    if (out) {
        Event* ev = bind_event(queued_ns, submit_ns, start_ns, end_ns);
        if (!ev) return VX_ERR_OUT_OF_HOST_MEMORY;
        *out = to_handle(ev);
    }
    return VX_SUCCESS;
}

vx_result_t Queue::enqueue_launch(const vx_launch_info_t* info,
                                  uint32_t nw, const vx_event_h* w,
                                  vx_event_h* out) {
    if (!info || !info->kernel || !info->args) return VX_ERR_INVALID_VALUE;
    if (info->struct_size < sizeof(vx_launch_info_t))
        return VX_ERR_INVALID_INFO;
    // ndim==0 is the legacy "use prior DCRs, just trigger launch" escape
    // hatch for vx_start (see common/vortex_legacy_wrapper.cpp). The CP-aware
    // v2 path uses ndim in [1, 3] and programs grid/block DCRs here.
    if (info->ndim > 3) return VX_ERR_INVALID_VALUE;

    uint64_t queued_ns = now_ns();
    auto r = wait_on_externals(nw, w);
    if (r != VX_SUCCESS) return r;

    Buffer* kernel = to_buffer(info->kernel);
    Buffer* args   = to_buffer(info->args);

    uint64_t submit_ns = now_ns();
    Platform* p = device_->platform();

    // Program legacy startup DCRs (PC + args). Even when ndim==0 (legacy
    // path), the kernel/args pointers still need to be programmed unless
    // the caller has already done so via prior vx_dcr_write calls — but
    // setting them again is idempotent and harmless.
    {
        std::lock_guard<std::mutex> g(enqueue_mu_);

        uint64_t pc   = kernel->dev_address();
        uint64_t argp = args->dev_address();
        r = p->dcr_write(VX_DCR_KMU_STARTUP_ADDR0,
                         (uint32_t)(pc & 0xffffffff));
        if (r != VX_SUCCESS) return r;
        r = p->dcr_write(VX_DCR_KMU_STARTUP_ADDR1,
                         (uint32_t)(pc >> 32));
        if (r != VX_SUCCESS) return r;
        r = p->dcr_write(VX_DCR_KMU_STARTUP_ARG0,
                         (uint32_t)(argp & 0xffffffff));
        if (r != VX_SUCCESS) return r;
        r = p->dcr_write(VX_DCR_KMU_STARTUP_ARG1,
                         (uint32_t)(argp >> 32));
        if (r != VX_SUCCESS) return r;

        // TODO(commit 1c+): when ndim > 0, program KMU grid/block/lmem DCRs
        // here from info->grid_dim / block_dim / lmem_size. v1 pre-CP path
        // requires the caller to set these via prior vx_dcr_write calls
        // (matching legacy vx_start semantics).
        (void)kernel; (void)args;

        r = p->launch_start();
        if (r != VX_SUCCESS) return r;
    }   // release enqueue_mu_ before async wait

    // Async: spawn a background thread to wait for launch completion and
    // signal the returned event. Retain the device so it cannot be
    // destroyed before the thread completes; retain the event so the
    // caller releasing it doesn't free it out from under us.
    Event* ev = nullptr;
    if (out) {
        if (Event::create(device_, &ev) != VX_SUCCESS)
            return VX_ERR_OUT_OF_HOST_MEMORY;
        ev->retain();   // for the worker thread
        *out = to_handle(ev);
    }

    Device* dev = device_;
    dev->retain();   // for the worker thread
    bool prof = profiling_enabled();
    std::thread([dev, ev, prof, queued_ns, submit_ns]() {
        uint64_t start_ns = now_ns();
        auto r = dev->platform()->launch_wait(VX_TIMEOUT_INFINITE);
        uint64_t end_ns = now_ns();
        if (ev) {
            if (prof) ev->set_profile(queued_ns, submit_ns, start_ns, end_ns);
            ev->complete(r);
            ev->release();
        }
        dev->release();
    }).detach();

    return VX_SUCCESS;
}

vx_result_t Queue::enqueue_barrier(uint32_t nw, const vx_event_h* w,
                                   vx_event_h* out) {
    uint64_t queued_ns = now_ns();
    auto r = wait_on_externals(nw, w);
    if (r != VX_SUCCESS) return r;
    uint64_t end_ns = now_ns();
    if (out) {
        Event* ev = bind_event(queued_ns, queued_ns, queued_ns, end_ns);
        if (!ev) return VX_ERR_OUT_OF_HOST_MEMORY;
        *out = to_handle(ev);
    }
    return VX_SUCCESS;
}

vx_result_t Queue::enqueue_dcr_write(uint32_t addr, uint32_t value,
                                     uint32_t nw, const vx_event_h* w,
                                     vx_event_h* out) {
    uint64_t queued_ns = now_ns();
    auto r = wait_on_externals(nw, w);
    if (r != VX_SUCCESS) return r;

    uint64_t submit_ns = now_ns();
    uint64_t start_ns  = submit_ns;
    {
        std::lock_guard<std::mutex> g(enqueue_mu_);
        r = device_->platform()->dcr_write(addr, value);
    }
    if (r != VX_SUCCESS) return r;
    uint64_t end_ns = now_ns();

    if (out) {
        Event* ev = bind_event(queued_ns, submit_ns, start_ns, end_ns);
        if (!ev) return VX_ERR_OUT_OF_HOST_MEMORY;
        *out = to_handle(ev);
    }
    return VX_SUCCESS;
}

vx_result_t Queue::enqueue_dcr_read(uint32_t addr, uint32_t* host_dst,
                                    uint32_t nw, const vx_event_h* w,
                                    vx_event_h* out) {
    if (!host_dst) return VX_ERR_INVALID_VALUE;
    uint64_t queued_ns = now_ns();
    auto r = wait_on_externals(nw, w);
    if (r != VX_SUCCESS) return r;

    uint64_t submit_ns = now_ns();
    uint64_t start_ns  = submit_ns;
    {
        std::lock_guard<std::mutex> g(enqueue_mu_);
        r = device_->platform()->dcr_read(addr, /*tag=*/0, host_dst);
    }
    if (r != VX_SUCCESS) return r;
    uint64_t end_ns = now_ns();

    if (out) {
        Event* ev = bind_event(queued_ns, submit_ns, start_ns, end_ns);
        if (!ev) return VX_ERR_OUT_OF_HOST_MEMORY;
        *out = to_handle(ev);
    }
    return VX_SUCCESS;
}

} // namespace vx

// ============================================================================
// C entry points
// ============================================================================

using namespace vx;

extern "C" vx_result_t vx_queue_create(vx_device_h dev,
                                       const vx_queue_info_t* info,
                                       vx_queue_h* out) {
    if (!dev || !out) return VX_ERR_INVALID_VALUE;
    Queue* q = nullptr;
    auto r = Queue::create(to_device(dev), info, &q);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(q);
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_queue_retain(vx_queue_h q) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    to_queue(q)->retain();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_queue_release(vx_queue_h q) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    to_queue(q)->release();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_queue_flush(vx_queue_h q) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->flush();
}

extern "C" vx_result_t vx_queue_finish(vx_queue_h q, uint64_t timeout_ns) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->finish(timeout_ns);
}

extern "C" vx_result_t vx_enqueue_launch(vx_queue_h q,
                                         const vx_launch_info_t* info,
                                         uint32_t nw, const vx_event_h* w,
                                         vx_event_h* out) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_launch(info, nw, w, out);
}

extern "C" vx_result_t vx_enqueue_copy(vx_queue_h q,
                                       vx_buffer_h dst, uint64_t do_,
                                       vx_buffer_h src, uint64_t so,
                                       uint64_t sz, uint32_t nw,
                                       const vx_event_h* w, vx_event_h* out) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_copy(to_buffer(dst), do_, to_buffer(src), so,
                                     sz, nw, w, out);
}

extern "C" vx_result_t vx_enqueue_read(vx_queue_h q, void* host_dst,
                                       vx_buffer_h src, uint64_t so,
                                       uint64_t sz, uint32_t nw,
                                       const vx_event_h* w, vx_event_h* out) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_read(host_dst, to_buffer(src), so, sz, nw,
                                     w, out);
}

extern "C" vx_result_t vx_enqueue_write(vx_queue_h q,
                                        vx_buffer_h dst, uint64_t off,
                                        const void* host_src, uint64_t sz,
                                        uint32_t nw, const vx_event_h* w,
                                        vx_event_h* out) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_write(to_buffer(dst), off, host_src, sz, nw,
                                      w, out);
}

extern "C" vx_result_t vx_enqueue_barrier(vx_queue_h q, uint32_t nw,
                                          const vx_event_h* w,
                                          vx_event_h* out) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_barrier(nw, w, out);
}

extern "C" vx_result_t vx_enqueue_dcr_write(vx_queue_h q,
                                            uint32_t addr, uint32_t value,
                                            uint32_t nw, const vx_event_h* w,
                                            vx_event_h* out) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_dcr_write(addr, value, nw, w, out);
}

extern "C" vx_result_t vx_enqueue_dcr_read(vx_queue_h q,
                                           uint32_t addr, uint32_t* host_dst,
                                           uint32_t nw, const vx_event_h* w,
                                           vx_event_h* out) {
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_dcr_read(addr, host_dst, nw, w, out);
}
