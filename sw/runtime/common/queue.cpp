// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "vortex2_internal.h"

#include <VX_types.h>

#include <algorithm>
#include <array>
#include <cstring>

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

    // Retain dst for the worker's lifetime — caller may release the buffer
    // immediately after enqueue returns (matches OpenCL retain semantics).
    dst->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, dst, off, host, sz](uint64_t* s, uint64_t* e) {
        vx_result_t r;
        {
            *s = now_ns();
            std::lock_guard<std::mutex> g(enqueue_mu_);
            // Host->device through the CP's DMA engine (CMD_MEM_WRITE).
            r = device_->cp_submit_mem_write(dst->dev_address() + off,
                                             host, sz);
            *e = now_ns();
        }
        dst->release();
        return r;
    };
    auto r = this->enqueue(std::move(cmd), nw, w, out);
    if (r != VX_SUCCESS) dst->release();
    return r;
}

vx_result_t Queue::enqueue_read(void* host, Buffer* src, uint64_t so,
                                uint64_t sz, uint32_t nw,
                                const vx_event_h* w, vx_event_h* out) {
    if (!src || (!host && sz != 0)) return VX_ERR_INVALID_VALUE;
    if (so + sz > src->size())      return VX_ERR_INVALID_VALUE;

    src->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, host, src, so, sz](uint64_t* s, uint64_t* e) {
        vx_result_t r;
        {
            *s = now_ns();
            std::lock_guard<std::mutex> g(enqueue_mu_);
            // Device->host through the CP's DMA engine (CMD_MEM_READ).
            r = device_->cp_submit_mem_read(host,
                                            src->dev_address() + so, sz);
            *e = now_ns();
        }
        src->release();
        return r;
    };
    auto r = this->enqueue(std::move(cmd), nw, w, out);
    if (r != VX_SUCCESS) src->release();
    return r;
}

vx_result_t Queue::enqueue_copy(Buffer* dst, uint64_t do_, Buffer* src,
                                uint64_t so, uint64_t sz, uint32_t nw,
                                const vx_event_h* w, vx_event_h* out) {
    if (!dst || !src)               return VX_ERR_INVALID_VALUE;
    if (do_ + sz > dst->size())     return VX_ERR_INVALID_VALUE;
    if (so + sz > src->size())      return VX_ERR_INVALID_VALUE;

    dst->retain();
    src->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, dst, do_, src, so, sz](uint64_t* s, uint64_t* e) {
        vx_result_t r;
        {
            *s = now_ns();
            std::lock_guard<std::mutex> g(enqueue_mu_);
            // Device->device through the CP's DMA engine (CMD_MEM_COPY).
            r = device_->cp_submit_mem_copy(dst->dev_address() + do_,
                                            src->dev_address() + so, sz);
            *e = now_ns();
        }
        src->release();
        dst->release();
        return r;
    };
    auto r = this->enqueue(std::move(cmd), nw, w, out);
    if (r != VX_SUCCESS) { src->release(); dst->release(); }
    return r;
}

vx_result_t Queue::enqueue_launch(const vx_launch_info_t* info,
                                  uint32_t nw, const vx_event_h* w,
                                  vx_event_h* out) {
    if (!info) return VX_ERR_INVALID_VALUE;
    if (info->struct_size < sizeof(vx_launch_info_t))
        return VX_ERR_INVALID_INFO;
    if (info->ndim > 3) return VX_ERR_INVALID_VALUE;
    if (info->args_size != 0 && !info->args_host) return VX_ERR_INVALID_VALUE;

    // info->kernel is a vx_kernel_h (vx_module_get_kernel). NULL is the legacy
    // escape hatch — the caller pre-programmed the PC DCRs itself. Retain the
    // kernel for the worker's lifetime so the underlying module image
    // (Kernel → Module → image Buffer) stays alive until the launch retires,
    // even if the caller releases the kernel immediately. The launch PCs are
    // derived from this handle inside the work lambda, so it is the only
    // kernel-state we capture.
    Kernel* kernel = (info->kernel != nullptr) ? to_kernel(info->kernel) : nullptr;
    if (kernel) kernel->retain();

    // Copy the args block now so the caller can free/reuse `info` (and the
    // memory it points at) the instant enqueue returns. An empty blob is the
    // legacy escape hatch — the caller pre-programmed the ARG DCRs itself.
    std::vector<uint8_t> args_blob;
    if (info->args_host && info->args_size > 0) {
        const uint8_t* p = static_cast<const uint8_t*>(info->args_host);
        args_blob.assign(p, p + info->args_size);
    }

    // Capture the launch descriptor by value into the work lambda so the
    // caller can free/reuse `info` immediately after enqueue returns.
    // ndim==0 is the legacy escape hatch — grid/block DCRs are left to the
    // host's prior vx_dcr_write calls (matches legacy vx_start semantics);
    // kernel==NULL and args_host==NULL are the analogous PC / ARG hatches.
    const uint32_t ndim      = info->ndim;
    const uint32_t lmem_size = info->lmem_size;
    std::array<uint32_t, 3> grid_in  = {1, 1, 1};
    std::array<uint32_t, 3> block_in = {1, 1, 1};
    for (uint32_t i = 0; i < ndim; ++i) {
        grid_in [i] = info->grid_dim [i];
        block_in[i] = info->block_dim[i];
    }
    // Cluster shape (CTAs guaranteed co-resident on a core). Normalise
    // zero entries to 1 (no grouping). Validate that grid_dim is a whole
    // multiple of cluster_dim along each in-use axis.
    std::array<uint32_t, 3> lg_in = {1, 1, 1};
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t lg = info->cluster_dim[i];
        if (lg == 0) lg = 1;
        if (grid_in[i] % lg != 0) {
            if (kernel) kernel->release();
            return VX_ERR_INVALID_VALUE;
        }
        lg_in[i] = lg;
    }

    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, kernel, ndim, lmem_size, grid_in, block_in, lg_in,
                args_blob = std::move(args_blob)](uint64_t* s, uint64_t* e) {
        // Launch PCs, derived from the retained kernel handle. has_kernel is
        // false on the legacy escape hatch (caller pre-programmed the DCRs).
        // kernel_pc  → KERNEL_ENTRY: the selected kernel's function entry,
        //              read back per-CTA via VX_CSR_CTA_ENTRY.
        // program_pc → STARTUP_ADDR: the program image base where every warp
        //              begins executing (__vx_cta_entry is linked there).
        // Both are XLEN-wide (64-bit on build64) and split across the
        // paired *_ADDR0/1 / *_ENTRY0/1 DCRs below.
        const bool     has_kernel = (kernel != nullptr);
        const uint64_t kernel_pc  = has_kernel ? kernel->pc() : 0;
        const uint64_t program_pc = has_kernel ? kernel->module()->base_address() : 0;
        // ---- Compute the full KMU descriptor (block_size, warp_step).
        uint64_t num_threads = 0, num_warps = 0;
        if (ndim > 0) {
            auto r = device_->query_caps(VX_CAPS_NUM_THREADS, &num_threads);
            if (r != VX_SUCCESS) {
                if (kernel) kernel->release();
                *s = *e = now_ns(); return r;
            }
            r = device_->query_caps(VX_CAPS_NUM_WARPS, &num_warps);
            if (r != VX_SUCCESS) {
                if (kernel) kernel->release();
                *s = *e = now_ns(); return r;
            }
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
            if (r != VX_SUCCESS) {
                if (kernel) kernel->release();
                *s = *e = now_ns(); return r;
            }
            r = device_->dev_write(args_addr, args_blob.data(),
                                   args_blob.size());
            if (r != VX_SUCCESS) {
                device_->args_slot_release(args_addr, args_pooled);
                if (kernel) kernel->release();
                *s = *e = now_ns();
                return r;
            }
        }

        vx_result_t r;
        {
            std::lock_guard<std::mutex> g(enqueue_mu_);

            // Program the KMU DCRs via CMD_DCR_WRITE descriptors through
            // the CP ring. has_kernel / args_staged false → caller
            // pre-programmed those DCRs (legacy escape hatch).
            #define WR(addr, val) do {                                        \
                auto _r = device_->cp_submit_dcr_write((addr), (uint32_t)(val)); \
                if (_r != VX_SUCCESS) {                                       \
                    if (args_staged)                                          \
                        device_->args_slot_release(args_addr, args_pooled);   \
                    if (kernel) kernel->release();                            \
                    *s = *e = now_ns();                                       \
                    return _r;                                                \
                }                                                             \
            } while (0)
            if (has_kernel) {
                WR(VX_DCR_KMU_STARTUP_ADDR0, program_pc & 0xffffffffu);
                WR(VX_DCR_KMU_STARTUP_ADDR1, program_pc >> 32);
                WR(VX_DCR_KMU_KERNEL_ENTRY0, kernel_pc & 0xffffffffu);
                WR(VX_DCR_KMU_KERNEL_ENTRY1, kernel_pc >> 32);
            }
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
                WR(VX_DCR_KMU_CLUSTER_DIM_X, lg_in[0]);
                WR(VX_DCR_KMU_CLUSTER_DIM_Y, lg_in[1]);
                WR(VX_DCR_KMU_CLUSTER_DIM_Z, lg_in[2]);
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
        if (kernel) kernel->release();
        return r;
    };
    auto r = this->enqueue(std::move(cmd), nw, w, out);
    if (r != VX_SUCCESS && kernel) kernel->release();
    return r;
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

// ============================================================================
// Rect-DMA helpers — software fallback for the *_rect enqueues. Each rect is
// decomposed into linear transfers; a future CP DMA descriptor with native
// 3D stride can replace the per-row loop without an API change.
// ============================================================================

namespace {

struct ResolvedRect {
    size_t region[3];
    size_t buffer_origin[3];
    size_t host_origin[3];
    size_t buffer_row, buffer_slice;
    size_t host_row, host_slice;
};

// Apply OpenCL pitch defaults (0 row pitch -> region[0]; 0 slice pitch ->
// region[1] * row_pitch) and reject a degenerate region or a pitch too
// small to hold its row / slice.
vx_result_t resolve_rect(const vx_rect_info_t& in, ResolvedRect* out) {
    for (int i = 0; i < 3; ++i) {
        out->region[i]        = in.region[i];
        out->buffer_origin[i] = in.buffer_origin[i];
        out->host_origin[i]   = in.host_origin[i];
    }
    if (in.region[0] == 0 || in.region[1] == 0 || in.region[2] == 0)
        return VX_ERR_INVALID_VALUE;
    out->buffer_row   = in.buffer_row_pitch   ? in.buffer_row_pitch
                                              : in.region[0];
    out->buffer_slice = in.buffer_slice_pitch ? in.buffer_slice_pitch
                                              : out->buffer_row * in.region[1];
    out->host_row     = in.host_row_pitch     ? in.host_row_pitch
                                              : in.region[0];
    out->host_slice   = in.host_slice_pitch   ? in.host_slice_pitch
                                              : out->host_row * in.region[1];
    if (out->buffer_row < in.region[0] || out->host_row < in.region[0])
        return VX_ERR_INVALID_VALUE;
    if (out->buffer_slice < out->buffer_row * in.region[1] ||
        out->host_slice   < out->host_row   * in.region[1])
        return VX_ERR_INVALID_VALUE;
    return VX_SUCCESS;
}

// Byte offset of row `row` of slice `sl` within one side of the rect.
uint64_t rect_off(const size_t origin[3], size_t row_pitch, size_t slice_pitch,
                  size_t sl, size_t row) {
    return (uint64_t)(origin[2] + sl)  * slice_pitch
         + (uint64_t)(origin[1] + row) * row_pitch
         + origin[0];
}

// Highest byte (exclusive) one side of the rect touches — for bounds checks.
uint64_t rect_span(const size_t region[3], const size_t origin[3],
                   size_t row_pitch, size_t slice_pitch) {
    return rect_off(origin, row_pitch, slice_pitch,
                    region[2] - 1, region[1] - 1) + region[0];
}

// Walk the rect, invoking row_fn(buffer_off, host_off, len) for each
// contiguous run — once for the whole rect when it is fully contiguous,
// otherwise once per row. Stops at the first failing transfer.
template <class Fn>
vx_result_t rect_for_each(const ResolvedRect& r, Fn&& row_fn) {
    const bool contig =
        r.buffer_row   == r.region[0] &&
        r.host_row     == r.region[0] &&
        r.buffer_slice == r.region[1] * r.region[0] &&
        r.host_slice   == r.region[1] * r.region[0];
    if (contig) {
        return row_fn(
            rect_off(r.buffer_origin, r.buffer_row, r.buffer_slice, 0, 0),
            rect_off(r.host_origin,   r.host_row,   r.host_slice,   0, 0),
            (uint64_t)r.region[0] * r.region[1] * r.region[2]);
    }
    for (size_t sl = 0; sl < r.region[2]; ++sl) {
        for (size_t row = 0; row < r.region[1]; ++row) {
            auto rc = row_fn(
                rect_off(r.buffer_origin, r.buffer_row, r.buffer_slice, sl, row),
                rect_off(r.host_origin,   r.host_row,   r.host_slice,   sl, row),
                (uint64_t)r.region[0]);
            if (rc != VX_SUCCESS) return rc;
        }
    }
    return VX_SUCCESS;
}

} // namespace

vx_result_t Queue::enqueue_read_rect(void* host_dst, Buffer* src,
                                     const vx_rect_info_t* info,
                                     uint32_t nw, const vx_event_h* w,
                                     vx_event_h* out) {
    if (!src || !host_dst || !info)                  return VX_ERR_INVALID_VALUE;
    if (info->struct_size < sizeof(vx_rect_info_t))  return VX_ERR_INVALID_INFO;
    ResolvedRect rr;
    auto r = resolve_rect(*info, &rr);
    if (r != VX_SUCCESS) return r;
    if (rect_span(rr.region, rr.buffer_origin, rr.buffer_row, rr.buffer_slice)
        > src->size())
        return VX_ERR_INVALID_VALUE;

    src->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, host_dst, src, rr](uint64_t* s, uint64_t* e) {
        vx_result_t rc;
        {
            *s = now_ns();
            std::lock_guard<std::mutex> g(enqueue_mu_);
            rc = rect_for_each(rr, [&](uint64_t bo, uint64_t ho, uint64_t len) {
                return device_->dev_read((uint8_t*)host_dst + ho,
                                         src->dev_address() + bo, len);
            });
            *e = now_ns();
        }
        src->release();
        return rc;
    };
    auto rr_enq = this->enqueue(std::move(cmd), nw, w, out);
    if (rr_enq != VX_SUCCESS) src->release();
    return rr_enq;
}

vx_result_t Queue::enqueue_write_rect(Buffer* dst, const void* host_src,
                                      const vx_rect_info_t* info,
                                      uint32_t nw, const vx_event_h* w,
                                      vx_event_h* out) {
    if (!dst || !host_src || !info)                  return VX_ERR_INVALID_VALUE;
    if (info->struct_size < sizeof(vx_rect_info_t))  return VX_ERR_INVALID_INFO;
    ResolvedRect rr;
    auto r = resolve_rect(*info, &rr);
    if (r != VX_SUCCESS) return r;
    if (rect_span(rr.region, rr.buffer_origin, rr.buffer_row, rr.buffer_slice)
        > dst->size())
        return VX_ERR_INVALID_VALUE;

    dst->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, dst, host_src, rr](uint64_t* s, uint64_t* e) {
        vx_result_t rc;
        {
            *s = now_ns();
            std::lock_guard<std::mutex> g(enqueue_mu_);
            rc = rect_for_each(rr, [&](uint64_t bo, uint64_t ho, uint64_t len) {
                return device_->dev_write(dst->dev_address() + bo,
                                          (const uint8_t*)host_src + ho, len);
            });
            *e = now_ns();
        }
        dst->release();
        return rc;
    };
    auto rr_enq = this->enqueue(std::move(cmd), nw, w, out);
    if (rr_enq != VX_SUCCESS) dst->release();
    return rr_enq;
}

vx_result_t Queue::enqueue_copy_rect(Buffer* dst, Buffer* src,
                                     const vx_rect_info_t* info,
                                     uint32_t nw, const vx_event_h* w,
                                     vx_event_h* out) {
    if (!dst || !src || !info)                       return VX_ERR_INVALID_VALUE;
    if (info->struct_size < sizeof(vx_rect_info_t))  return VX_ERR_INVALID_INFO;
    ResolvedRect rr;
    auto r = resolve_rect(*info, &rr);
    if (r != VX_SUCCESS) return r;
    // buffer_* describes the destination, host_* the source.
    if (rect_span(rr.region, rr.buffer_origin, rr.buffer_row, rr.buffer_slice)
        > dst->size())
        return VX_ERR_INVALID_VALUE;
    if (rect_span(rr.region, rr.host_origin, rr.host_row, rr.host_slice)
        > src->size())
        return VX_ERR_INVALID_VALUE;

    dst->retain();
    src->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, dst, src, rr](uint64_t* s, uint64_t* e) {
        vx_result_t rc;
        {
            *s = now_ns();
            std::lock_guard<std::mutex> g(enqueue_mu_);
            rc = rect_for_each(rr, [&](uint64_t do_, uint64_t so, uint64_t len) {
                return device_->dev_copy(dst->dev_address() + do_,
                                         src->dev_address() + so, len);
            });
            *e = now_ns();
        }
        src->release();
        dst->release();
        return rc;
    };
    auto rr_enq = this->enqueue(std::move(cmd), nw, w, out);
    if (rr_enq != VX_SUCCESS) { src->release(); dst->release(); }
    return rr_enq;
}

vx_result_t Queue::enqueue_fill_buffer(Buffer* dst, uint64_t offset,
                                       uint64_t size, const void* pattern,
                                       size_t pattern_size, uint32_t nw,
                                       const vx_event_h* w, vx_event_h* out) {
    if (!dst || !pattern)             return VX_ERR_INVALID_VALUE;
    if (pattern_size == 0)            return VX_ERR_INVALID_VALUE;
    if (size % pattern_size != 0)     return VX_ERR_INVALID_VALUE;
    if (offset + size > dst->size())  return VX_ERR_INVALID_VALUE;

    std::vector<uint8_t> pat(static_cast<const uint8_t*>(pattern),
                             static_cast<const uint8_t*>(pattern) + pattern_size);
    dst->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, dst, offset, size, pat = std::move(pat)]
               (uint64_t* s, uint64_t* e) {
        vx_result_t r = VX_SUCCESS;
        {
            *s = now_ns();
            std::lock_guard<std::mutex> g(enqueue_mu_);
            // Stage one bounded chunk of the repeated pattern and upload it
            // in a loop — avoids a host allocation the full size of a large
            // fill.
            const uint64_t kChunkCap = 64 * 1024;
            uint64_t chunk = kChunkCap - (kChunkCap % pat.size());  // whole patterns
            if (chunk > size) chunk = size;
            std::vector<uint8_t> staging(chunk);
            for (uint64_t i = 0; i < chunk; i += pat.size())
                std::memcpy(staging.data() + i, pat.data(), pat.size());
            for (uint64_t done = 0; done < size && r == VX_SUCCESS; done += chunk) {
                uint64_t n = std::min<uint64_t>(chunk, size - done);
                r = device_->dev_write(dst->dev_address() + offset + done,
                                       staging.data(), n);
            }
            *e = now_ns();
        }
        dst->release();
        return r;
    };
    auto rc = this->enqueue(std::move(cmd), nw, w, out);
    if (rc != VX_SUCCESS) dst->release();
    return rc;
}

vx_result_t Queue::enqueue_map(Buffer* buf, uint64_t offset, uint64_t size,
                               uint32_t flags, uint32_t nw,
                               const vx_event_h* w, vx_event_h* out,
                               void** out_host_ptr) {
    if (!buf || !out_host_ptr) return VX_ERR_INVALID_VALUE;
    // Reserve the host mirror now so the caller gets a valid pointer
    // synchronously; the READ population runs on the worker in queue order.
    auto r = buf->map_reserve(offset, size, flags, out_host_ptr);
    if (r != VX_SUCCESS) return r;

    buf->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, buf](uint64_t* s, uint64_t* e) {
        vx_result_t rc;
        {
            *s = now_ns();
            std::lock_guard<std::mutex> g(enqueue_mu_);
            rc = buf->map_commit();
            *e = now_ns();
        }
        buf->release();
        return rc;
    };
    r = this->enqueue(std::move(cmd), nw, w, out);
    if (r != VX_SUCCESS) {
        buf->release();
        buf->map_cancel();
        *out_host_ptr = nullptr;
    }
    return r;
}

vx_result_t Queue::enqueue_unmap(Buffer* buf, void* host_ptr, uint32_t nw,
                                 const vx_event_h* w, vx_event_h* out) {
    if (!buf) return VX_ERR_INVALID_HANDLE;
    buf->retain();
    Command cmd;
    cmd.queued_ns = now_ns();
    cmd.work = [this, buf, host_ptr](uint64_t* s, uint64_t* e) {
        vx_result_t r;
        {
            *s = now_ns();
            std::lock_guard<std::mutex> g(enqueue_mu_);
            r = buf->unmap(host_ptr);
            *e = now_ns();
        }
        buf->release();
        return r;
    };
    auto rc = this->enqueue(std::move(cmd), nw, w, out);
    if (rc != VX_SUCCESS) buf->release();
    return rc;
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
    VX_C_ENTRY_TRY
    if (!dev || !out) return VX_ERR_INVALID_VALUE;
    Queue* q = nullptr;
    auto r = Queue::create(to_device(dev), info, &q);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(q);
    return VX_SUCCESS;
    VX_C_ENTRY_CATCH
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
    VX_C_ENTRY_TRY
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->finish(timeout_ns);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_launch(vx_queue_h q,
                                         const vx_launch_info_t* info,
                                         uint32_t nw, const vx_event_h* w,
                                         vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_launch(info, nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_copy(vx_queue_h q,
                                       vx_buffer_h dst, uint64_t do_,
                                       vx_buffer_h src, uint64_t so,
                                       uint64_t sz, uint32_t nw,
                                       const vx_event_h* w, vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_copy(to_buffer(dst), do_, to_buffer(src), so,
                                     sz, nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_read(vx_queue_h q, void* host_dst,
                                       vx_buffer_h src, uint64_t so,
                                       uint64_t sz, uint32_t nw,
                                       const vx_event_h* w, vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_read(host_dst, to_buffer(src), so, sz, nw,
                                     w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_write(vx_queue_h q,
                                        vx_buffer_h dst, uint64_t off,
                                        const void* host_src, uint64_t sz,
                                        uint32_t nw, const vx_event_h* w,
                                        vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_write(to_buffer(dst), off, host_src, sz, nw,
                                      w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_barrier(vx_queue_h q, uint32_t nw,
                                          const vx_event_h* w,
                                          vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_barrier(nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_read_rect(vx_queue_h q, void* host_dst,
                                            vx_buffer_h src,
                                            const vx_rect_info_t* info,
                                            uint32_t nw, const vx_event_h* w,
                                            vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q || !src) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_read_rect(host_dst, to_buffer(src), info,
                                          nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_write_rect(vx_queue_h q, vx_buffer_h dst,
                                             const void* host_src,
                                             const vx_rect_info_t* info,
                                             uint32_t nw, const vx_event_h* w,
                                             vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q || !dst) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_write_rect(to_buffer(dst), host_src, info,
                                           nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_copy_rect(vx_queue_h q, vx_buffer_h dst,
                                            vx_buffer_h src,
                                            const vx_rect_info_t* info,
                                            uint32_t nw, const vx_event_h* w,
                                            vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q || !dst || !src) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_copy_rect(to_buffer(dst), to_buffer(src), info,
                                          nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_fill_buffer(vx_queue_h q, vx_buffer_h dst,
                                              uint64_t offset, uint64_t size,
                                              const void* pattern,
                                              size_t pattern_size,
                                              uint32_t nw, const vx_event_h* w,
                                              vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q || !dst) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_fill_buffer(to_buffer(dst), offset, size,
                                            pattern, pattern_size, nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_map(vx_queue_h q, vx_buffer_h buf,
                                      uint64_t offset, uint64_t size,
                                      uint32_t flags, uint32_t nw,
                                      const vx_event_h* w, vx_event_h* out,
                                      void** out_host_ptr) {
    VX_C_ENTRY_TRY
    if (!q || !buf) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_map(to_buffer(buf), offset, size, flags,
                                    nw, w, out, out_host_ptr);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_unmap(vx_queue_h q, vx_buffer_h buf,
                                        void* host_ptr, uint32_t nw,
                                        const vx_event_h* w, vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q || !buf) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_unmap(to_buffer(buf), host_ptr, nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_dcr_write(vx_queue_h q,
                                            uint32_t addr, uint32_t value,
                                            uint32_t nw, const vx_event_h* w,
                                            vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_dcr_write(addr, value, nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_dcr_read(vx_queue_h q,
                                           uint32_t addr, uint32_t* host_dst,
                                           uint32_t nw, const vx_event_h* w,
                                           vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_dcr_read(addr, host_dst, nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_signal(vx_queue_h q, vx_event_h ev,
                                         uint64_t value,
                                         uint32_t nw, const vx_event_h* w,
                                         vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q || !ev) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_signal(to_event(ev), value, nw, w, out);
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_enqueue_wait_value(vx_queue_h q, vx_event_h ev,
                                             uint64_t value,
                                             uint32_t nw, const vx_event_h* w,
                                             vx_event_h* out) {
    VX_C_ENTRY_TRY
    if (!q || !ev) return VX_ERR_INVALID_HANDLE;
    return to_queue(q)->enqueue_wait_value(to_event(ev), value, nw, w, out);
    VX_C_ENTRY_CATCH
}
