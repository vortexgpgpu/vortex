// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "vortex2_internal.h"

#include <VX_config.h>
#include <VX_types.h>

#include <array>

namespace vx {

// ============================================================================
// Construction / destruction
// ============================================================================

Queue::Queue(Device* dev, const vx_queue_info_t& info)
    : device_(dev),
      priority_(static_cast<uint32_t>(info.priority)),
      flags_(info.flags) {
    device_->retain();
    device_->register_queue(this);
    worker_ = std::thread([this]{ this->worker_loop(); });
}

Queue::~Queue() {
    // Drain + stop the worker. Push a shutdown flag and wake the worker;
    // it will finish any commands already in the FIFO and then return.
    {
        std::lock_guard<std::mutex> g(cmd_mu_);
        shutdown_ = true;
    }
    cmd_cv_.notify_all();
    if (worker_.joinable()) worker_.join();

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

// ============================================================================
// Worker loop — processes commands strictly in FIFO order.
//
// Each command may have a wait-list of events that must complete before its
// work runs. The waits happen on the worker thread, so an enqueue gated on
// an unsignaled user event does not block the caller. In-order queue
// semantics are preserved because there is exactly one worker per Queue.
// ============================================================================

void Queue::worker_loop() {
    while (true) {
        Command cmd;
        {
            std::unique_lock<std::mutex> lk(cmd_mu_);
            cmd_cv_.wait(lk, [&]{ return shutdown_ || !commands_.empty(); });
            if (commands_.empty()) return;   // shutdown with empty queue
            cmd = std::move(commands_.front());
            commands_.pop_front();
        }

        // Wait for each external dependency. wait() blocks the worker but
        // not the caller; if a wait fails (event errored), short-circuit
        // the command's work and propagate the failure into completion.
        vx_result_t r = VX_SUCCESS;
        for (Event* dep : cmd.waits) {
            if (r == VX_SUCCESS) r = dep->wait(VX_TIMEOUT_INFINITE);
            dep->release();
        }

        uint64_t submit_ns = now_ns();
        uint64_t start_ns  = submit_ns;
        uint64_t end_ns    = submit_ns;

        if (r == VX_SUCCESS && cmd.work) {
            r = cmd.work(&start_ns, &end_ns);
        }

        if (cmd.completion) {
            if (profiling_enabled()) {
                cmd.completion->set_profile(cmd.queued_ns, submit_ns,
                                            start_ns, end_ns);
            }
            cmd.completion->complete(r);
            cmd.completion->release();
        }
    }
}

// ============================================================================
// enqueue() — common builder: capture waits, allocate completion event,
// stuff the command into the FIFO, notify the worker.
// ============================================================================

vx_result_t Queue::enqueue(Command&& cmd, uint32_t nw, const vx_event_h* w,
                           vx_event_h* out) {
    if (nw != 0 && !w) return VX_ERR_INVALID_VALUE;

    // Retain each wait event so the caller can release them immediately
    // after enqueue returns. The worker releases them in turn after each
    // wait completes.
    cmd.waits.reserve(nw);
    for (uint32_t i = 0; i < nw; ++i) {
        if (!w[i]) return VX_ERR_INVALID_HANDLE;
        Event* e = to_event(w[i]);
        e->retain();
        cmd.waits.push_back(e);
    }

    // Completion event — created in QUEUED state. The worker will mark it
    // COMPLETE (or set ERROR status) once cmd.work runs. We hand the
    // caller one ref and the worker holds one ref.
    Event* completion = nullptr;
    auto r = Event::create(device_, &completion);
    if (r != VX_SUCCESS) {
        for (Event* e : cmd.waits) e->release();
        return r;
    }
    completion->retain();           // for the worker
    cmd.completion = completion;

    if (out) *out = to_handle(completion);
    else     completion->release(); // caller doesn't want it — drop caller's ref

    {
        std::lock_guard<std::mutex> g(cmd_mu_);
        commands_.push_back(std::move(cmd));
    }
    cmd_cv_.notify_one();
    return VX_SUCCESS;
}

// ============================================================================
// flush / finish
// ============================================================================

vx_result_t Queue::flush() {
    // The worker is already woken on each enqueue, so this is effectively
    // a no-op sync point for higher layers.
    cmd_cv_.notify_one();
    return VX_SUCCESS;
}

vx_result_t Queue::finish(uint64_t timeout_ns) {
    // Enqueue a sentinel barrier and wait for its completion event. This
    // is the in-order-queue contract: after finish returns, every
    // previously enqueued command has completed (the barrier sits behind
    // them in FIFO order).
    vx_event_h ev = nullptr;
    auto r = this->enqueue_barrier(0, nullptr, &ev);
    if (r != VX_SUCCESS) return r;
    r = to_event(ev)->wait(timeout_ns);
    to_event(ev)->release();
    return r;
}

// ============================================================================
// Enqueue primitives — each wraps a Platform call into a Command lambda.
// ============================================================================

vx_result_t Queue::enqueue_write(Buffer* dst, uint64_t off, const void* host,
                                 uint64_t sz, uint32_t nw,
                                 const vx_event_h* w, vx_event_h* out) {
    if (!dst || (!host && sz != 0)) return VX_ERR_INVALID_VALUE;
    if (off + sz > dst->size())     return VX_ERR_INVALID_VALUE;

    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, dst, off, host, sz](uint64_t* s, uint64_t* e) {
        *s = now_ns();
        std::lock_guard<std::mutex> g(enqueue_mu_);
        auto r = device_->platform()->mem_upload(dst->dev_address() + off,
                                                 host, sz);
        *e = now_ns();
        return r;
    };
    return this->enqueue(std::move(cmd), nw, w, out);
}

vx_result_t Queue::enqueue_read(void* host, Buffer* src, uint64_t so,
                                uint64_t sz, uint32_t nw,
                                const vx_event_h* w, vx_event_h* out) {
    if (!src || (!host && sz != 0)) return VX_ERR_INVALID_VALUE;
    if (so + sz > src->size())      return VX_ERR_INVALID_VALUE;

    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, host, src, so, sz](uint64_t* s, uint64_t* e) {
        *s = now_ns();
        std::lock_guard<std::mutex> g(enqueue_mu_);
        auto r = device_->platform()->mem_download(host,
                                                   src->dev_address() + so, sz);
        *e = now_ns();
        return r;
    };
    return this->enqueue(std::move(cmd), nw, w, out);
}

vx_result_t Queue::enqueue_copy(Buffer* dst, uint64_t do_, Buffer* src,
                                uint64_t so, uint64_t sz, uint32_t nw,
                                const vx_event_h* w, vx_event_h* out) {
    if (!dst || !src)               return VX_ERR_INVALID_VALUE;
    if (do_ + sz > dst->size())     return VX_ERR_INVALID_VALUE;
    if (so + sz > src->size())      return VX_ERR_INVALID_VALUE;

    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, dst, do_, src, so, sz](uint64_t* s, uint64_t* e) {
        *s = now_ns();
        std::lock_guard<std::mutex> g(enqueue_mu_);
        auto r = device_->platform()->mem_copy(dst->dev_address() + do_,
                                               src->dev_address() + so, sz);
        *e = now_ns();
        return r;
    };
    return this->enqueue(std::move(cmd), nw, w, out);
}

vx_result_t Queue::enqueue_launch(const vx_launch_info_t* info,
                                  uint32_t nw, const vx_event_h* w,
                                  vx_event_h* out) {
    if (!info || !info->kernel) return VX_ERR_INVALID_VALUE;
    if (info->struct_size < sizeof(vx_launch_info_t))
        return VX_ERR_INVALID_INFO;
    if (info->ndim > 3) return VX_ERR_INVALID_VALUE;
    if (info->args_size != 0 && !info->args_host) return VX_ERR_INVALID_VALUE;

    // Resolve the kernel handle. The launch kernel slot accepts either a
    // Buffer (legacy: image base = entry PC) or a Kernel (new in Phase 1b:
    // named entry point inside a Module). Disambiguated by the tag at
    // offset 0 of the LaunchKernelHandle base.
    LaunchKernelHandle* lkh = to_lkh(info->kernel);
    const uint64_t kernel_pc = (lkh->launch_kind() == LaunchKernelHandle::Kind::Kernel)
                                 ? static_cast<Kernel*>(lkh)->pc()
                                 : static_cast<Buffer*>(lkh)->dev_address();

    // Phase 2: the args block arrives as a host blob. Copy it into the
    // command now so the caller can free/reuse `info` (and the memory it
    // points at) the instant enqueue returns. An empty blob is the legacy
    // escape hatch — the caller pre-programmed the ARG DCRs itself.
    std::vector<uint8_t> args_blob;
    if (info->args_host && info->args_size > 0) {
        const uint8_t* p = static_cast<const uint8_t*>(info->args_host);
        args_blob.assign(p, p + info->args_size);
    }

    // Capture the launch descriptor by value into the work lambda so the
    // caller can free/reuse `info` immediately after enqueue returns.
    // ndim==0 is the legacy escape hatch — only PC (+ staged arg ptr) are
    // programmed and the host is expected to have set the rest via prior
    // vx_dcr_write calls (matches legacy vx_start semantics).
    const uint32_t ndim      = info->ndim;
    const uint32_t lmem_size = info->lmem_size;
    std::array<uint32_t, 3> grid_in  = {1, 1, 1};
    std::array<uint32_t, 3> block_in = {1, 1, 1};
    for (uint32_t i = 0; i < ndim; ++i) {
        grid_in [i] = info->grid_dim [i];
        block_in[i] = info->block_dim[i];
    }

    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, kernel_pc, ndim, lmem_size, grid_in, block_in,
                args_blob = std::move(args_blob)](uint64_t* s, uint64_t* e) {
        Platform* p = device_->platform();

        // ---- Compute the full KMU descriptor (block_size, warp_step).
        uint64_t num_threads = 0, num_warps = 0;
        if (ndim > 0) {
            auto r = p->query_caps(VX_CAPS_NUM_THREADS, &num_threads);
            if (r != VX_SUCCESS) { *s = *e = now_ns(); return r; }
            r = p->query_caps(VX_CAPS_NUM_WARPS, &num_warps);
            if (r != VX_SUCCESS) { *s = *e = now_ns(); return r; }
        }
        uint32_t eff_block[3] = {1, 1, 1};
        for (uint32_t i = 0; i < ndim; ++i) eff_block[i] = block_in[i];
        uint32_t block_size = 1;
        for (uint32_t i = 0; i < ndim; ++i) block_size *= eff_block[i];
        const uint32_t tpw = (uint32_t)num_threads;
        const uint32_t ws_x = (ndim >= 1 && eff_block[0]) ?
                                tpw % eff_block[0] : 0;
        const uint32_t ws_y = (ndim >= 2 && eff_block[1]) ?
                                (tpw / eff_block[0]) % eff_block[1] : 0;
        const uint32_t ws_z = (ndim >= 3 && eff_block[2]) ?
                                (tpw / (eff_block[0] * eff_block[1]))
                                  % eff_block[2] : 0;

        // ---- Stage the kernel-args blob into a device scratch slot.
        // Empty blob → caller pre-programmed the ARG DCRs (legacy path).
        uint64_t args_addr   = 0;
        bool     args_pooled = false;
        bool     args_staged = !args_blob.empty();
        if (args_staged) {
            auto r = device_->args_slot_acquire(args_blob.size(),
                                                &args_addr, &args_pooled);
            if (r != VX_SUCCESS) { *s = *e = now_ns(); return r; }
            r = p->mem_upload(args_addr, args_blob.data(), args_blob.size());
            if (r != VX_SUCCESS) {
                device_->args_slot_release(args_addr, args_pooled);
                *s = *e = now_ns();
                return r;
            }
        }

        vx_result_t r;
        {
            std::lock_guard<std::mutex> g(enqueue_mu_);

            const uint64_t pc = kernel_pc;

            // Program the KMU DCRs via CMD_DCR_WRITE descriptors through
            // the CP ring.
            #define WR(addr, val) do {                                        \
                auto _r = device_->cp_submit_dcr_write((addr), (uint32_t)(val)); \
                if (_r != VX_SUCCESS) {                                       \
                    if (args_staged)                                          \
                        device_->args_slot_release(args_addr, args_pooled);   \
                    *s = *e = now_ns();                                       \
                    return _r;                                                \
                }                                                             \
            } while (0)
            WR(VX_DCR_KMU_STARTUP_ADDR0, pc & 0xffffffffu);
            WR(VX_DCR_KMU_STARTUP_ADDR1, pc >> 32);
            if (args_staged) {
                WR(VX_DCR_KMU_STARTUP_ARG0, args_addr & 0xffffffffu);
                WR(VX_DCR_KMU_STARTUP_ARG1, args_addr >> 32);
            }

            if (ndim > 0) {
                WR(VX_DCR_KMU_BLOCK_DIM_X, eff_block[0]);
                WR(VX_DCR_KMU_BLOCK_DIM_Y, eff_block[1]);
                WR(VX_DCR_KMU_BLOCK_DIM_Z, eff_block[2]);
                WR(VX_DCR_KMU_GRID_DIM_X,  grid_in[0]);
                WR(VX_DCR_KMU_GRID_DIM_Y,  ndim >= 2 ? grid_in[1] : 1);
                WR(VX_DCR_KMU_GRID_DIM_Z,  ndim >= 3 ? grid_in[2] : 1);
                WR(VX_DCR_KMU_LMEM_SIZE,   lmem_size);
                WR(VX_DCR_KMU_BLOCK_SIZE,  block_size);
                WR(VX_DCR_KMU_WARP_STEP_X, ws_x);
                WR(VX_DCR_KMU_WARP_STEP_Y, ws_y);
                WR(VX_DCR_KMU_WARP_STEP_Z, ws_z);
            }
            #undef WR

            *s = now_ns();
            // cp_submit_launch posts CMD_LAUNCH and polls Q_SEQNUM until
            // the engine retires (the engine retires only after Vortex
            // signals done, so Q_SEQNUM advance means the kernel
            // finished).
            r = device_->cp_submit_launch();
            *e = now_ns();
        }

        // Launch retired — the kernel has consumed its args. Return the
        // scratch slot to the pool for the next launch.
        if (args_staged)
            device_->args_slot_release(args_addr, args_pooled);
        return r;
    };
    return this->enqueue(std::move(cmd), nw, w, out);
}

vx_result_t Queue::enqueue_barrier(uint32_t nw, const vx_event_h* w,
                                   vx_event_h* out) {
    // A barrier is a no-op work item; its purpose is to introduce a
    // synchronization point that completes only after all waits resolve.
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [](uint64_t* s, uint64_t* e) {
        uint64_t t = now_ns();
        *s = t; *e = t;
        return VX_SUCCESS;
    };
    return this->enqueue(std::move(cmd), nw, w, out);
}

vx_result_t Queue::enqueue_dcr_write(uint32_t addr, uint32_t value,
                                     uint32_t nw, const vx_event_h* w,
                                     vx_event_h* out) {
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, addr, value](uint64_t* s, uint64_t* e) {
        *s = now_ns();
        std::lock_guard<std::mutex> g(enqueue_mu_);
        auto r = device_->cp_submit_dcr_write(addr, value);
        *e = now_ns();
        return r;
    };
    return this->enqueue(std::move(cmd), nw, w, out);
}

vx_result_t Queue::enqueue_dcr_read(uint32_t addr, uint32_t* host_dst,
                                    uint32_t nw, const vx_event_h* w,
                                    vx_event_h* out) {
    if (!host_dst) return VX_ERR_INVALID_VALUE;

    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, addr, host_dst](uint64_t* s, uint64_t* e) {
        *s = now_ns();
        std::lock_guard<std::mutex> g(enqueue_mu_);
        auto r = device_->cp_submit_dcr_read(addr, /*tag=*/0, host_dst);
        *e = now_ns();
        return r;
    };
    return this->enqueue(std::move(cmd), nw, w, out);
}

vx_result_t Queue::enqueue_signal(Event* ev, uint64_t value,
                                  uint32_t nw, const vx_event_h* w,
                                  vx_event_h* out) {
    if (!ev) return VX_ERR_INVALID_HANDLE;
    // Eagerly allocate the CP-side counter slot now (cheap if already done)
    // so all downstream worker-side and CP-side signals see a consistent
    // device-resident value. The slot doubles as the mirror that
    // Event::signal upcalls write into via mem_upload.
    ev->cp_slot();
    // Retain the target event for the duration the work item lives in the
    // queue; release in the work lambda after signaling.
    ev->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, ev, value](uint64_t* s, uint64_t* e) {
        *s = now_ns();
        // 1) Host-side signal FIRST. Event::signal() mirrors the value
        //    to the device-side counter slot via mem_upload, which
        //    bypasses cp_mu_. A CP-side CMD_EVENT_WAIT spinning on the
        //    same slot will see the update and retire — without this,
        //    a WAIT that grabbed cp_mu_ first would deadlock against a
        //    SIGNAL waiting on cp_mu_.
        ev->signal(value);
        // 2) CP-side SIGNAL — enforces queue-internal ordering against
        //    subsequent ops on this queue that depend on the slot being
        //    set. The CP write of the same value is redundant but
        //    idempotent; the cost is one ring slot + one round-trip.
        vx_result_t r = VX_SUCCESS;
        uint64_t slot = ev->cp_slot();
        if (slot != 0) {
            std::lock_guard<std::mutex> g(enqueue_mu_);
            r = device_->cp_submit_event_signal(slot, value);
        }
        ev->release();
        *e = now_ns();
        return r;
    };
    return this->enqueue(std::move(cmd), nw, w, out);
}

vx_result_t Queue::enqueue_wait_value(Event* ev, uint64_t value,
                                      uint32_t nw, const vx_event_h* w,
                                      vx_event_h* out) {
    if (!ev) return VX_ERR_INVALID_HANDLE;
    ev->cp_slot();
    ev->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, ev, value](uint64_t* s, uint64_t* e) {
        *s = now_ns();
        vx_result_t r = VX_SUCCESS;
        // Route through the CP if the event has a device-side slot —
        // VX_cp_event_unit spin-polls device-side. The CP retire
        // serializes the wait against subsequent queue ops without a
        // host round-trip per check. cp_submit_cl_ releases cp_mu_
        // during the SEQNUM poll so a concurrent SIGNAL on another
        // queue can post into the same ring.
        uint64_t slot = ev->cp_slot();
        if (slot != 0) {
            std::lock_guard<std::mutex> g(enqueue_mu_);
            r = device_->cp_submit_event_wait(slot, value);
        } else {
            // Fallback: host-side wait. Used only if slot allocation
            // failed (out of device memory) — preserves correctness.
            r = ev->wait_value(value, VX_TIMEOUT_INFINITE);
        }
        ev->release();
        *e = now_ns();
        return r;
    };
    return this->enqueue(std::move(cmd), nw, w, out);
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

extern "C" vx_result_t vx_enqueue_signal(vx_queue_h q, vx_event_h ev,
                                         uint64_t value,
                                         uint32_t nw, const vx_event_h* w,
                                         vx_event_h* out) {
    if (!q || !ev) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_signal(to_event(ev), value, nw, w, out);
}

extern "C" vx_result_t vx_enqueue_wait_value(vx_queue_h q, vx_event_h ev,
                                             uint64_t value,
                                             uint32_t nw, const vx_event_h* w,
                                             vx_event_h* out) {
    if (!q || !ev) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_wait_value(to_event(ev), value, nw, w, out);
}
