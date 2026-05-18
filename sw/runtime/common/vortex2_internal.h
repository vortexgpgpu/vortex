// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// vortex2_internal.h — internal C++ class declarations for vortex2.h.
//
// Not a public header. Backends include this to subclass vx::Platform.
// The C wrappers in vx_device.cpp / vx_queue.cpp / etc. translate the
// public vx_*_h handles into pointers to these classes.
// ============================================================================

#ifndef __VX_VORTEX2_INTERNAL_H__
#define __VX_VORTEX2_INTERNAL_H__

#include <vortex2.h>
#include <callbacks.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

namespace vx {

class Device;
class Buffer;
class Queue;
class Event;

// ============================================================================
// Refcount base.
// ============================================================================

template <class T>
class RefCounted {
public:
    void retain() { refs_.fetch_add(1, std::memory_order_relaxed); }

    bool release() {
        if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete static_cast<T*>(this);
            return true;
        }
        return false;
    }

    uint32_t refs() const { return refs_.load(std::memory_order_relaxed); }

protected:
    ~RefCounted() = default;

private:
    std::atomic<uint32_t> refs_{1};   // created with one reference
};

// ============================================================================
// Platform — backend abstraction.
//
// Each backend (simx, rtlsim, xrt) provides a concrete subclass and a
// single C-linkage factory function:
//
//   extern "C" vx::Platform* vx_create_platform();
//
// vx::Device::open() calls vx_create_platform() and owns the returned
// pointer.
//
// The Platform interface exposes the small set of synchronous primitives
// the dispatcher needs from each backend: capability queries, device
// memory management, raw DMA, and the CP MMIO surface. Higher-level
// async machinery (Queue/Event) lives in the dispatcher on top of it.
// ============================================================================

class Platform {
public:
    virtual ~Platform() = default;

    // ----- Capability queries -----
    virtual vx_result_t query_caps(uint32_t caps_id, uint64_t* out) = 0;
    virtual vx_result_t memory_info(uint64_t* free, uint64_t* used) = 0;

    // ----- Device memory allocation -----
    virtual vx_result_t mem_alloc  (uint64_t size, uint32_t flags,
                                    uint64_t* out_dev_addr) = 0;
    virtual vx_result_t mem_reserve(uint64_t dev_addr, uint64_t size,
                                    uint32_t flags) = 0;
    virtual vx_result_t mem_free   (uint64_t dev_addr) = 0;
    virtual vx_result_t mem_access (uint64_t dev_addr, uint64_t size,
                                    uint32_t flags) = 0;

    // ----- DMA -----
    virtual vx_result_t mem_upload  (uint64_t dst_dev_addr, const void* src,
                                     uint64_t size) = 0;
    virtual vx_result_t mem_download(void* dst, uint64_t src_dev_addr,
                                     uint64_t size) = 0;
    virtual vx_result_t mem_copy    (uint64_t dst_dev_addr,
                                     uint64_t src_dev_addr, uint64_t size) = 0;

    // ----- Command Processor MMIO surface (sole control path) -----
    // `off` is the CP-internal regfile offset (0x000..0x13F per the
    // VX_cp_axil_regfile address map). Backends translate to their own
    // physical address space (xrt/opae add 0x1000; simx/rtlsim proxy
    // to a software CommandProcessor).
    virtual vx_result_t cp_mmio_write(uint32_t off, uint32_t value) = 0;
    virtual vx_result_t cp_mmio_read (uint32_t off, uint32_t* out)  = 0;
};

// ============================================================================
// CallbacksAdapter — vx::Platform subclass that bridges the C ABI
// callbacks_t (filled by each backend's vx_dev_init) to the C++ Platform
// virtual interface used by vx::Device/Queue/Buffer/Event.
//
// Each Device owns one CallbacksAdapter holding the loaded backend's
// callbacks_t table and the backend's opaque device context pointer.
// All Platform virtual calls forward through the table; cb_.dev_close
// fires automatically when the adapter is destroyed.
// ============================================================================

class CallbacksAdapter final : public Platform {
public:
    CallbacksAdapter(const callbacks_t& cb, void* dev_ctx)
        : cb_(cb), dev_ctx_(dev_ctx) {}

    ~CallbacksAdapter() override {
        if (cb_.dev_close && dev_ctx_) cb_.dev_close(dev_ctx_);
    }

    static vx_result_t r(int rc) {
        return (rc == 0) ? VX_SUCCESS : VX_ERR_INVALID_VALUE;
    }

    vx_result_t query_caps(uint32_t caps_id, uint64_t* out) override {
        return r(cb_.query_caps(dev_ctx_, caps_id, out));
    }
    vx_result_t memory_info(uint64_t* free, uint64_t* used) override {
        return r(cb_.memory_info(dev_ctx_, free, used));
    }

    vx_result_t mem_alloc(uint64_t size, uint32_t flags,
                          uint64_t* out_dev_addr) override {
        return r(cb_.mem_alloc(dev_ctx_, size, flags, out_dev_addr));
    }
    vx_result_t mem_reserve(uint64_t dev_addr, uint64_t size,
                            uint32_t flags) override {
        return r(cb_.mem_reserve(dev_ctx_, dev_addr, size, flags));
    }
    vx_result_t mem_free(uint64_t dev_addr) override {
        return r(cb_.mem_free(dev_ctx_, dev_addr));
    }
    vx_result_t mem_access(uint64_t dev_addr, uint64_t size,
                           uint32_t flags) override {
        return r(cb_.mem_access(dev_ctx_, dev_addr, size, flags));
    }

    vx_result_t mem_upload(uint64_t dst_dev_addr, const void* src,
                           uint64_t size) override {
        return r(cb_.mem_upload(dev_ctx_, dst_dev_addr, src, size));
    }
    vx_result_t mem_download(void* dst, uint64_t src_dev_addr,
                             uint64_t size) override {
        return r(cb_.mem_download(dev_ctx_, dst, src_dev_addr, size));
    }
    vx_result_t mem_copy(uint64_t dst_dev_addr, uint64_t src_dev_addr,
                         uint64_t size) override {
        return r(cb_.mem_copy(dev_ctx_, dst_dev_addr, src_dev_addr, size));
    }

    vx_result_t cp_mmio_write(uint32_t off, uint32_t value) override {
        return r(cb_.cp_mmio_write(dev_ctx_, off, value));
    }
    vx_result_t cp_mmio_read(uint32_t off, uint32_t* out) override {
        return r(cb_.cp_mmio_read(dev_ctx_, off, out));
    }

private:
    callbacks_t cb_;
    void*       dev_ctx_;
};

// ============================================================================
// Device.
// ============================================================================

class Device : public RefCounted<Device> {
public:
    static vx_result_t open(uint32_t index, Device** out);

    Platform* platform()                     { return platform_.get(); }
    uint64_t  cycle_freq_hz()           const{ return cycle_freq_hz_; }

    // Legacy-wrapper helpers. The default queue is created lazily on the
    // first legacy call that needs one and destroyed at Device destruction.
    Queue*    legacy_default_queue();
    Event*    legacy_take_last_event();
    void      legacy_remember_last_event(Event* ev);

    // Tracks live queues / buffers so destruction at device close can
    // be ordered.
    void register_queue   (Queue*  q);
    void unregister_queue (Queue*  q);
    void register_buffer  (Buffer* b);
    void unregister_buffer(Buffer* b);

    // ----- Command Processor submission path -----
    // The CP is the sole control path: the device owns a CP ring +
    // completion slot in device memory, and the Queue layer calls
    // cp_submit_* for every launch and DCR op. cp_enabled() is always
    // true post-init and is exposed as a method only for readability
    // at the call sites.
    bool cp_enabled() const { return cp_enabled_; }

    // Post one CMD_DCR_WRITE to the ring, commit Q_TAIL, and wait for
    // Q_SEQNUM to reach the post's sequence number. Synchronous semantics.
    vx_result_t cp_submit_dcr_write(uint32_t addr, uint32_t value);

    // Post one CMD_LAUNCH to the ring, commit Q_TAIL, and wait for
    // Q_SEQNUM. Synchronous.
    vx_result_t cp_submit_launch();

    // Post one CMD_DCR_READ to the ring, wait for retire, and read the
    // response from the CP regfile's Q_LAST_DCR_RSP slot. `tag` is
    // forwarded as the DCR read's data bus payload (e.g. per-core
    // CACHE_FLUSH addressing).
    vx_result_t cp_submit_dcr_read(uint32_t addr, uint32_t tag,
                                   uint32_t* out_value);

private:
    friend class RefCounted<Device>;
    explicit Device(std::unique_ptr<Platform> plat);
    ~Device();

    // Allocate ring/head/cmpl buffers and program the CP regfile.
    // Called from Device::open() after the platform is ready.
    vx_result_t cp_init();

    // Push one pre-built CL into the ring + commit Q_TAIL + wait. Used by
    // cp_submit_dcr_write / cp_submit_launch — they just build the CL.
    vx_result_t cp_submit_cl_(const void* cl);

    std::unique_ptr<Platform>      platform_;
    uint64_t                       cycle_freq_hz_;

    std::mutex                     mu_;
    std::unordered_set<Queue*>     queues_;
    std::unordered_set<Buffer*>    buffers_;

    Queue*                         legacy_q_     = nullptr;
    Event*                         legacy_last_  = nullptr;

    // CP state — populated only when cp_enabled_ == true.
    bool                           cp_enabled_         = false;
    uint64_t                       cp_ring_dev_addr_   = 0;
    uint64_t                       cp_head_dev_addr_   = 0;
    uint64_t                       cp_cmpl_dev_addr_   = 0;
    uint64_t                       cp_tail_            = 0;
    uint64_t                       cp_expected_seqnum_ = 0;
    std::mutex                     cp_mu_;             // serialize ring writes
};

// ============================================================================
// Buffer.
// ============================================================================

class Buffer : public RefCounted<Buffer> {
public:
    static vx_result_t create (Device* dev, uint64_t size, uint32_t flags,
                               Buffer** out);
    static vx_result_t reserve(Device* dev, uint64_t address, uint64_t size,
                               uint32_t flags, Buffer** out);

    Device*  device()      { return device_; }
    uint64_t dev_address() const { return dev_addr_; }
    uint64_t size()        const { return size_; }
    uint32_t flags()       const { return flags_; }

    vx_result_t access(uint64_t off, uint64_t size, uint32_t flags);
    vx_result_t map   (uint64_t off, uint64_t size, uint32_t flags, void** out);
    vx_result_t unmap (void* host_ptr);

private:
    friend class RefCounted<Buffer>;
    Buffer(Device* dev, uint64_t dev_addr, uint64_t size, uint32_t flags);
    ~Buffer();

    Device*       device_;
    uint64_t      dev_addr_;
    uint64_t      size_;
    uint32_t      flags_;

    // Mapping state (only used when VX_MEM_PIN_MEMORY is honored; simx
    // does not expose a true host-visible buffer, so map() shadows
    // through a heap-allocated mirror — see Buffer::map for the policy).
    std::mutex    map_mu_;
    void*         host_mirror_  = nullptr;   // heap mirror, freed at unmap
    uint64_t      mapped_off_   = 0;
    uint64_t      mapped_size_  = 0;
    uint32_t      mapped_flags_ = 0;
    bool          mapped_       = false;
};

// ============================================================================
// Queue.
// ============================================================================

class Queue : public RefCounted<Queue> {
public:
    static vx_result_t create(Device* dev, const vx_queue_info_t* info,
                              Queue** out);

    Device*  device()                  { return device_; }
    uint32_t flags()              const{ return flags_; }
    bool     profiling_enabled()  const{ return (flags_ & VX_QUEUE_PROFILING_ENABLE) != 0; }

    vx_result_t flush();
    vx_result_t finish(uint64_t timeout_ns);

    // ----- Enqueue primitives -----
    vx_result_t enqueue_launch (const vx_launch_info_t* info,
                                uint32_t nw, const vx_event_h* w,
                                vx_event_h* out);
    vx_result_t enqueue_copy   (Buffer* dst, uint64_t do_, Buffer* src,
                                uint64_t so, uint64_t sz,
                                uint32_t nw, const vx_event_h* w,
                                vx_event_h* out);
    vx_result_t enqueue_read   (void* host, Buffer* src, uint64_t so,
                                uint64_t sz, uint32_t nw, const vx_event_h* w,
                                vx_event_h* out);
    vx_result_t enqueue_write  (Buffer* dst, uint64_t off, const void* host,
                                uint64_t sz, uint32_t nw, const vx_event_h* w,
                                vx_event_h* out);
    vx_result_t enqueue_barrier(uint32_t nw, const vx_event_h* w,
                                vx_event_h* out);
    vx_result_t enqueue_dcr_write(uint32_t addr, uint32_t value,
                                  uint32_t nw, const vx_event_h* w,
                                  vx_event_h* out);
    vx_result_t enqueue_dcr_read (uint32_t addr, uint32_t* host_dst,
                                  uint32_t nw, const vx_event_h* w,
                                  vx_event_h* out);

private:
    friend class RefCounted<Queue>;
    Queue(Device* dev, const vx_queue_info_t& info);
    ~Queue();

    // ------------------------------------------------------------------
    // Per-queue worker thread. Each enqueue builds a Command and pushes
    // it to commands_; the worker pops them one at a time, waits on the
    // command's dep events, then runs the work lambda. This decouples
    // enqueue latency from execution latency so an enqueue gated on an
    // unsignaled user event does not block the caller — the wait runs on
    // the worker thread instead.
    //
    // In-queue ordering is preserved (FIFO, single worker), matching the
    // OpenCL in-order queue semantics POCL relies on.
    // ------------------------------------------------------------------
    struct Command {
        std::vector<Event*>                                       waits;
        Event*                                                    completion = nullptr;
        uint64_t                                                  queued_ns  = 0;
        // work returns the platform result and fills start/end timestamps
        // when profiling is requested (caller writes 0s when it doesn't
        // know — barrier, dcr_read with sync read, etc.).
        std::function<vx_result_t(uint64_t* start_ns, uint64_t* end_ns)> work;
    };

    void worker_loop();

    // ------------------------------------------------------------------
    // Helper: capture a wait-list into a Command, retaining each event.
    // Builds + atomically pushes the command, notifies the worker. Always
    // produces a completion event (retained for the caller; an extra ref
    // for the worker is held internally).
    // ------------------------------------------------------------------
    vx_result_t enqueue(Command&& cmd, uint32_t nw, const vx_event_h* w,
                        vx_event_h* out);

    Device*                  device_;
    uint32_t                 priority_;
    uint32_t                 flags_;

    // Serializes per-command platform calls when multiple queues share
    // one backend (one Platform per device today).
    std::mutex               enqueue_mu_;

    // Command FIFO + worker thread state.
    std::mutex               cmd_mu_;
    std::condition_variable  cmd_cv_;
    std::deque<Command>      commands_;
    bool                     shutdown_ = false;
    std::thread              worker_;
};

// ============================================================================
// Event.
//
// Runtime-managed events are born QUEUED and complete()'d by the
// dispatcher when the underlying work finishes. User events are also
// QUEUED at birth and transition only on vx_user_event_signal.
// ============================================================================

class Event : public RefCounted<Event> {
public:
    // Internal factory: creates an event in QUEUED state. Runtime code calls
    // complete() on it once the underlying work finishes.
    static vx_result_t create(Device* dev, Event** out);

    // Public-API factory: creates a user event that only the host can signal
    // via signal_user().
    static vx_result_t create_user(Device* dev, Event** out);

    // Public API: signal a user event from the host. Rejects non-user events.
    vx_result_t signal_user(vx_result_t status);

    // Internal: mark this event complete with the given status. Works for
    // any event (user or runtime-managed).
    void complete(vx_result_t status);

    vx_result_t status(vx_event_status_e* out);
    vx_result_t wait  (uint64_t timeout_ns);

    void set_profile(uint64_t queued_ns, uint64_t submit_ns,
                     uint64_t start_ns, uint64_t end_ns);
    vx_result_t get_profile(vx_profile_info_t* out);

    bool is_user() const { return is_user_; }

private:
    friend class RefCounted<Event>;
    Event(Device* dev, bool is_user);
    ~Event() = default;

    Device*                       device_;
    bool                          is_user_;
    std::mutex                    mu_;
    std::condition_variable       cv_;
    vx_event_status_e             status_  = VX_EVENT_STATUS_QUEUED;
    vx_result_t                   error_   = VX_SUCCESS;
    bool                          has_profile_ = false;
    vx_profile_info_t             profile_ {};
};

// ============================================================================
// Handle conversion helpers.
// ============================================================================

inline Device* to_device(vx_device_h h) { return static_cast<Device*>(h); }
inline Buffer* to_buffer(vx_buffer_h h) { return static_cast<Buffer*>(h); }
inline Queue*  to_queue (vx_queue_h  h) { return reinterpret_cast<Queue*>(h);  }
inline Event*  to_event (vx_event_h  h) { return reinterpret_cast<Event*>(h);  }

inline vx_device_h to_handle(Device* d) { return static_cast<vx_device_h>(d); }
inline vx_buffer_h to_handle(Buffer* b) { return static_cast<vx_buffer_h>(b); }
inline vx_queue_h  to_handle(Queue*  q) { return reinterpret_cast<vx_queue_h>(q);  }
inline vx_event_h  to_handle(Event*  e) { return reinterpret_cast<vx_event_h>(e);  }

// ============================================================================
// Wall clock helper for runtime-synthesized profile timestamps.
// ============================================================================

inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

} // namespace vx

#endif // __VX_VORTEX2_INTERNAL_H__
