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
// The C wrappers in device.cpp / queue.cpp / etc. translate the
// public vx_*_h handles into pointers to these classes.
// ============================================================================

#ifndef __VX_VORTEX2_INTERNAL_H__
#define __VX_VORTEX2_INTERNAL_H__

#include <vortex2.h>
#include <callbacks.h>
#include <VX_types.h>   // VX_MEM_IO_COUT_* (console stream-ring layout)
#include <mem_alloc.h>  // MemoryAllocator — device-memory allocator (common core)
#include <vm.h>         // VMManager — virtual memory (empty when VM disabled)

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
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

    // ----- CP register channel (sole control path) -----
    // `off` is the CP-internal regfile offset (matching VX_cp_axil_regfile).
    // Backends translate to their own physical MMIO space. cp_reg_write to
    // Q_TAIL is the doorbell; cp_reg_read of Q_SEQNUM is the completion poll.
    virtual vx_result_t cp_reg_write(uint32_t off, uint32_t value) = 0;
    virtual vx_result_t cp_reg_read (uint32_t off, uint32_t* out)  = 0;

    // ----- CP-visible host memory -----
    // Allocates host-resident memory the CP's m_axi_host master can DMA.
    // Returns a CPU pointer (the runtime memcpy's through it) and the
    // device-side address the CP uses for the same bytes. host_mem_free
    // is keyed by that cp_addr.
    virtual vx_result_t host_mem_alloc(uint64_t size, void** out_host_ptr,
                                       uint64_t* out_cp_addr) = 0;
    virtual vx_result_t host_mem_free (uint64_t cp_addr) = 0;
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

    vx_result_t cp_reg_write(uint32_t off, uint32_t value) override {
        return r(cb_.cp_reg_write(dev_ctx_, off, value));
    }
    vx_result_t cp_reg_read(uint32_t off, uint32_t* out) override {
        return r(cb_.cp_reg_read(dev_ctx_, off, out));
    }

    vx_result_t host_mem_alloc(uint64_t size, void** out_host_ptr,
                               uint64_t* out_cp_addr) override {
        return r(cb_.host_mem_alloc(dev_ctx_, size, out_host_ptr,
                                    out_cp_addr));
    }
    vx_result_t host_mem_free(uint64_t cp_addr) override {
        return r(cb_.host_mem_free(dev_ctx_, cp_addr));
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
    // Q_SEQNUM. Synchronous. Followed by an implicit CMD_CACHE_FLUSH so
    // the host observes coherent results (see cp_submit_cache_flush).
    vx_result_t cp_submit_launch();

    // Post one CMD_CACHE_FLUSH to the ring (AMD ACQUIRE_MEM model): the CP
    // sweeps a per-core cache flush across all cores and retires the
    // command only when the last core's flush completes. A no-op on
    // write-through cache configs. Posted after every CMD_LAUNCH.
    vx_result_t cp_submit_cache_flush();

    // Post one CMD_DCR_READ to the ring, wait for retire, and read the
    // response from the CP regfile's Q_LAST_DCR_RSP slot. `tag` is
    // forwarded as the DCR read's data bus payload (e.g. per-core
    // CACHE_FLUSH addressing).
    vx_result_t cp_submit_dcr_read(uint32_t addr, uint32_t tag,
                                   uint32_t* out_value);

    // Post one CMD_EVENT_SIGNAL: VX_cp_event_unit writes `value` to the
    // 8-byte counter slot at `event_dev_addr`. Returns when retired.
    vx_result_t cp_submit_event_signal(uint64_t event_dev_addr,
                                       uint64_t value);

    // Post one CMD_EVENT_WAIT: VX_cp_event_unit spin-polls the 8-byte
    // counter slot at `event_dev_addr` until it reaches `value` (>=).
    // Returns when the CP retires the wait.
    vx_result_t cp_submit_event_wait(uint64_t event_dev_addr,
                                     uint64_t value);

    // ----- CP-driven host<->device DMA (CMD_MEM_*) -----
    // The CP's VX_cp_dma engine performs the transfer; the host only
    // appends the descriptor to the ring. cp_submit_mem_copy moves
    // device->device directly. cp_submit_mem_write stages `host_src`
    // into a VX_MEM_HOST buffer and posts CMD_MEM_WRITE (host->device);
    // cp_submit_mem_read posts CMD_MEM_READ (device->host) into a
    // VX_MEM_HOST buffer and copies it back to `host_dst`. Synchronous.
    vx_result_t cp_submit_mem_copy (uint64_t dst, uint64_t src, uint64_t size);
    vx_result_t cp_submit_mem_write(uint64_t dev_dst, const void* host_src,
                                    uint64_t size, bool physical = false);
    vx_result_t cp_submit_mem_read (void* host_dst, uint64_t dev_src,
                                    uint64_t size, bool physical = false);

    // ----- Device-memory transfer router -----
    // Every dispatcher path that moves data to/from device memory (module
    // image, kernel args, COUT rings, rect/fill/map, buffer copies) goes
    // through these. They always route through the Command Processor's DMA
    // engine (CMD_MEM_*) — identically on every backend. The runtime never
    // touches device memory directly: it only appends commands to the host
    // ring and the CP executes them.
    vx_result_t dev_write(uint64_t dev_addr, const void* src, uint64_t size);
    vx_result_t dev_read (void* dst, uint64_t dev_addr, uint64_t size);
    vx_result_t dev_copy (uint64_t dst, uint64_t src, uint64_t size);

    // Drain the lossless COUT stream rings (proposal §10): copy out each
    // hart's [rd,wr) bytes, print "#slot: <line>", publish the advanced
    // rd[]. Called every CP launch-wait poll iteration — concurrent with
    // the producing kernel, so the kernel's back-pressure never deadlocks.
    vx_result_t drain_cout();

    // ---- Phase 2: kernel-args scratch pool ----
    //
    // Acquire a device-memory slot to stage a kernel-args blob of `size`
    // bytes. Slots <= ARGS_SLOT_SIZE come from a recycled free-list (the
    // common case — kernel arg blocks are small); oversized requests get
    // a one-off allocation. `*out_pooled` reports which, so the matching
    // args_slot_release frees correctly. Because vx_enqueue_launch holds
    // a slot only for the duration of one synchronous CP launch, the
    // free-list naturally settles at the count of concurrent launches.
    vx_result_t args_slot_acquire(uint64_t size, uint64_t* out_addr,
                                  bool* out_pooled);
    void        args_slot_release(uint64_t addr, bool pooled);

    // ----- Device-memory allocation (common-core allocator) -----
    // The CP is the sole DMA engine, so a device buffer is pure host-side
    // address bookkeeping in a MemoryAllocator the Device owns. The
    // VX_MEM_HOST flag routes to platform host memory instead.
    vx_result_t mem_alloc  (uint64_t size, uint32_t flags, uint64_t* out_addr);
    vx_result_t mem_reserve(uint64_t addr, uint64_t size, uint32_t flags);
    vx_result_t mem_free   (uint64_t addr);
    vx_result_t memory_info(uint64_t* out_free, uint64_t* out_used);

    // Decode a VX_CAPS_* id: the CP caps window (read via cp_reg_read +
    // vortex::load_caps/decode_caps) plus VX_CFG_* platform constants.
    vx_result_t query_caps (uint32_t caps_id, uint64_t* out_value);

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

    // Build + submit a CMD_MEM_* descriptor (opcode, dst, src, size).
    // `physical` sets the CMD_MEM header flag so the CP DMA skips VM
    // translation — used for page-table writes / the PT region.
    vx_result_t cp_submit_mem_(uint8_t opcode, uint64_t arg0, uint64_t arg1,
                               uint64_t arg2, bool physical = false);

    // ----- CP-visible host memory (command ring + DMA staging) -----
    // host_alloc wraps platform()->host_mem_alloc and records the region
    // so cp_addr -> host_ptr lookups and host_free work.
    struct HostMem {
        void*    host_ptr = nullptr;   // CPU pointer (runtime memcpy's here)
        uint64_t cp_addr  = 0;         // device-side address the CP DMAs
        uint64_t size     = 0;
    };
    vx_result_t host_alloc(uint64_t size, HostMem* out);
    void        host_free (uint64_t cp_addr);

    std::unique_ptr<Platform>      platform_;
    uint64_t                       cycle_freq_hz_;

    // Device-memory allocator (the CP DMAs to these addresses). Pure
    // host-side bookkeeping — constructed in the Device ctor.
    vortex::MemoryAllocator        global_mem_;
    // Live CP-visible host regions, keyed by cp_addr.
    std::map<uint64_t, HostMem>    host_mems_;

    // Lazily-loaded device/ISA caps words (read once from the CP regfile).
    uint64_t                       dev_caps_    = 0;
    uint64_t                       isa_caps_    = 0;
    bool                           caps_loaded_ = false;

    std::mutex                     mu_;
    std::unordered_set<Queue*>     queues_;
    std::unordered_set<Buffer*>    buffers_;

    Queue*                         legacy_q_     = nullptr;
    Event*                         legacy_last_  = nullptr;

    // CP state — populated only when cp_enabled_ == true. The ring / head /
    // completion buffers are CP-visible host memory (HostMem): the runtime
    // writes commands straight through their host_ptr.
    bool                           cp_enabled_         = false;
    HostMem                        cp_ring_;
    HostMem                        cp_head_;
    HostMem                        cp_cmpl_;
    uint64_t                       cp_tail_            = 0;
    uint64_t                       cp_expected_seqnum_ = 0;
    uint64_t                       cp_num_cores_       = 0; // VX_CAPS_NUM_CORES, cached for CMD_CACHE_FLUSH
    std::mutex                     cp_mu_;             // serialize ring writes

    // Virtual memory — the Device owns the VMManager (the page-table
    // builder) iff the device reports an MMU (vm_enabled_, discovered from
    // CP DEV_CAPS at open). mem_alloc mints VAs; the CP DMA does the VA->PA
    // walk. CpMemIO is the VMManager's device-memory port — PA-direct CP
    // DMA. Always compiled; vm_mgr_/vm_io_ stay null on an MMU-less device.
    bool                                vm_enabled_ = false;
    class CpMemIO;
    std::unique_ptr<CpMemIO>            vm_io_;
    std::unique_ptr<vortex::VMManager>  vm_mgr_;
    std::mutex                          vm_mu_;   // serialize VMManager ops

    // Lossless COUT stream-ring consumer state (proposal §10). cout_rd_[h]
    // is the host read pointer for hart h's ring; cout_line_[h] holds a
    // partial (not-yet-newline-terminated) line. Both persist across the
    // per-poll drain_cout() calls.
    uint32_t                       cout_rd_[VX_MEM_IO_COUT_SLOTS] = {};
    std::string                    cout_line_[VX_MEM_IO_COUT_SLOTS];

    // Kernel-args scratch pool (Phase 2). Free-list of recycled fixed-size
    // device slots; ARGS_SLOT_SIZE comfortably covers typical kernel arg
    // blocks. Drained in ~Device.
    static constexpr uint64_t      ARGS_SLOT_SIZE = 4096;
    std::mutex                     args_pool_mu_;
    std::vector<uint64_t>          args_pool_free_;
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

    // Async-map support: map_reserve allocates the host mirror and records
    // the mapping (no data transfer), so vx_enqueue_map can hand the caller
    // a pointer synchronously. map_commit performs the device->host READ
    // population on the queue worker. map_cancel tears down a reservation
    // whose enqueue failed (frees the mirror without flushing).
    vx_result_t map_reserve(uint64_t off, uint64_t size, uint32_t flags,
                            void** out);
    vx_result_t map_commit();
    void        map_cancel();

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
// Module + Kernel.
//
// A Module is a loaded .vxbin: device-side image buffer + parsed symbol
// table. A Kernel is a named entry point within a module (with its PC and
// a refcount on the owning module). Kernel objects are cached on the
// module the first time they're requested by name and reused thereafter.
// ============================================================================

class Module;
class Kernel;

class Module : public RefCounted<Module> {
public:
    // Load a .vxbin file from disk: parse header, reserve+upload the image,
    // parse symbol-table footer if present (single-`main` fallback otherwise).
    static vx_result_t load_file (Device* dev, const char* path, Module** out);
    static vx_result_t load_bytes(Device* dev, const void* bytes, size_t size,
                                  Module** out);

    Device*  device()      { return device_; }
    Buffer*  image()       { return image_; }
    uint64_t base_address() const { return base_addr_; }

    // Look up a named kernel entry point. Returns a new ref (caller releases).
    // Subsequent lookups of the same name return the same cached Kernel.
    vx_result_t get_kernel(const char* name, Kernel** out);

private:
    friend class RefCounted<Module>;
    friend class Kernel;        // accesses kcache_mu_ + kernel_cache_ on destruct
    Module(Device* dev, Buffer* image, uint64_t base_addr);
    ~Module();

    // Parse `bytes` (the full .vxbin content). Fills the symbol table.
    // If the footer magic is absent, populates a single "main" entry at
    // base_addr_.
    vx_result_t parse_symbols(const void* bytes, size_t size);

    struct Symbol {
        std::string name;
        uint64_t    pc;
    };

    Device*                              device_;
    Buffer*                              image_;       // refcounted
    uint64_t                             base_addr_;
    std::vector<Symbol>                  symbols_;
    std::mutex                           kcache_mu_;
    std::map<std::string, Kernel*>       kernel_cache_;   // weak refs (no retain)
};

class Kernel : public RefCounted<Kernel> {
public:
    static vx_result_t create(Module* mod, uint64_t pc, Kernel** out);

    Module*  module()      { return module_; }
    uint64_t pc()    const { return pc_; }

    // Per-kernel max-block hints. v1 returns the device default (full warp
    // width); per-kernel introspection from compiler metadata is a Phase 1b
    // follow-up once vxbin symbol footer carries it.
    vx_result_t get_max_block_size(uint32_t* x, uint32_t* y, uint32_t* z);

private:
    friend class RefCounted<Kernel>;
    Kernel(Module* mod, uint64_t pc);
    ~Kernel();

    Module*  module_;
    uint64_t pc_;
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
    vx_result_t enqueue_read_rect  (void* host_dst, Buffer* src,
                                    const vx_rect_info_t* info,
                                    uint32_t nw, const vx_event_h* w,
                                    vx_event_h* out);
    vx_result_t enqueue_write_rect (Buffer* dst, const void* host_src,
                                    const vx_rect_info_t* info,
                                    uint32_t nw, const vx_event_h* w,
                                    vx_event_h* out);
    vx_result_t enqueue_copy_rect  (Buffer* dst, Buffer* src,
                                    const vx_rect_info_t* info,
                                    uint32_t nw, const vx_event_h* w,
                                    vx_event_h* out);
    vx_result_t enqueue_fill_buffer(Buffer* dst, uint64_t offset, uint64_t size,
                                    const void* pattern, size_t pattern_size,
                                    uint32_t nw, const vx_event_h* w,
                                    vx_event_h* out);
    vx_result_t enqueue_map        (Buffer* buf, uint64_t offset, uint64_t size,
                                    uint32_t flags, uint32_t nw,
                                    const vx_event_h* w, vx_event_h* out,
                                    void** out_host_ptr);
    vx_result_t enqueue_unmap      (Buffer* buf, void* host_ptr,
                                    uint32_t nw, const vx_event_h* w,
                                    vx_event_h* out);
    vx_result_t enqueue_dcr_write(uint32_t addr, uint32_t value,
                                  uint32_t nw, const vx_event_h* w,
                                  vx_event_h* out);
    vx_result_t enqueue_dcr_read (uint32_t addr, uint32_t* host_dst,
                                  uint32_t nw, const vx_event_h* w,
                                  vx_event_h* out);

    // Queue-ordered timeline ops. enqueue_signal advances `ev` to `value`
    // when the queue's prior work completes. enqueue_wait_value blocks the
    // queue worker until `ev` reaches `value` before proceeding to later
    // enqueued commands.
    vx_result_t enqueue_signal     (Event* ev, uint64_t value,
                                    uint32_t nw, const vx_event_h* w,
                                    vx_event_h* out);
    vx_result_t enqueue_wait_value (Event* ev, uint64_t value,
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
// Event — counter-based timeline primitive.
//
// Each event carries a monotonic uint64_t counter that starts at 0. signal()
// advances it; waiters block until counter >= target. The legacy binary
// semantics (QUEUED -> COMPLETE/ERROR) map onto:
//   counter == 0 && error == SUCCESS  -> QUEUED
//   counter >= 1 && error == SUCCESS  -> COMPLETE
//   error  != SUCCESS                 -> ERROR
// Runtime-managed completion events from vx_enqueue_* are signaled to value
// 1 by the queue worker via complete(), which also records error status.
// Pure timeline waiters use signal()/wait_value() and never touch error_.
// ============================================================================

class Event : public RefCounted<Event> {
public:
    // Factory.
    static vx_result_t create(Device* dev, Event** out);

    // Timeline API: advance the counter and unblock waiters whose target
    // value is now satisfied. value < current is a no-op (counter is
    // monotonic). No error semantics — pure counter advance.
    void signal(uint64_t value);

    // Internal completion path for runtime-managed events: equivalent to
    // signal(1) but also records the work's result status. If status !=
    // VX_SUCCESS, subsequent wait_value() returns that status instead of
    // VX_SUCCESS. Idempotent.
    void complete(vx_result_t status);

    // Timeline wait — block until counter >= value or timeout.
    // Returns VX_ERR_TIMEOUT on timeout, recorded error_ if set, else SUCCESS.
    vx_result_t wait_value(uint64_t value, uint64_t timeout_ns);

    // Snapshot the current counter value.
    uint64_t get_value() const;

    // Internal convenience: wait for the binary-completion case (a
    // runtime-managed event the worker complete()s to value 1).
    vx_result_t wait(uint64_t timeout_ns) {
        return wait_value(1, timeout_ns);
    }

    void set_profile(uint64_t queued_ns, uint64_t submit_ns,
                     uint64_t start_ns, uint64_t end_ns);
    vx_result_t get_profile(vx_profile_info_t* out);

    // ---- Phase 1c: CP-backed timeline counter slot ----
    //
    // Each event optionally owns an 8-byte device-resident counter slot
    // that the CP's VX_cp_event_unit can read/write directly via
    // CMD_EVENT_SIGNAL / CMD_EVENT_WAIT. cp_slot() returns the device
    // address (0 if not allocated yet). cp_alloc_slot() lazily allocates
    // it from the device on first use; idempotent. Slot lifetime is tied
    // to the event's lifetime (freed in ~Event).
    uint64_t cp_slot();   // returns 0 on failure / no platform
    Device*  device() { return device_; }

private:
    friend class RefCounted<Event>;
    explicit Event(Device* dev);
    ~Event();

    Device*                       device_;
    mutable std::mutex            mu_;
    std::condition_variable       cv_;
    // Counter is mutated under mu_; reads under mu_ for memory ordering with
    // waiters. Plain uint64_t rather than atomic because every read/write
    // path already takes mu_ for the condvar interaction.
    uint64_t                      counter_ = 0;
    vx_result_t                   error_   = VX_SUCCESS;
    bool                          has_profile_ = false;
    vx_profile_info_t             profile_ {};

    // CP-backed slot — lazily allocated by cp_slot() on first call.
    // 0 means "not yet allocated".
    uint64_t                      cp_slot_addr_ = 0;
};

// ============================================================================
// Handle conversion helpers.
// ============================================================================

inline Device* to_device(vx_device_h h) { return static_cast<Device*>(h); }
inline Buffer* to_buffer(vx_buffer_h h) { return static_cast<Buffer*>(h); }
inline Queue*  to_queue (vx_queue_h  h) { return reinterpret_cast<Queue*>(h);  }
inline Event*  to_event (vx_event_h  h) { return reinterpret_cast<Event*>(h);  }
inline Module* to_module(vx_module_h h) { return reinterpret_cast<Module*>(h); }
inline Kernel* to_kernel(vx_kernel_h h) { return reinterpret_cast<Kernel*>(h); }

inline vx_device_h to_handle(Device* d) { return static_cast<vx_device_h>(d); }
inline vx_buffer_h to_handle(Buffer* b) { return static_cast<vx_buffer_h>(b); }
inline vx_queue_h  to_handle(Queue*  q) { return reinterpret_cast<vx_queue_h>(q);  }
inline vx_event_h  to_handle(Event*  e) { return reinterpret_cast<vx_event_h>(e);  }
inline vx_module_h to_handle(Module* m) { return reinterpret_cast<vx_module_h>(m); }
inline vx_kernel_h to_handle(Kernel* k) { return reinterpret_cast<vx_kernel_h>(k); }

// ============================================================================
// Wall clock helper for runtime-synthesized profile timestamps.
// ============================================================================

inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

} // namespace vx

#endif // __VX_VORTEX2_INTERNAL_H__
