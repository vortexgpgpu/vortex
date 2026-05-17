# CP Runtime Implementation Proposal (`vortex2.h`)

Status: draft proposal
Branch: `feature_cp`
Parent: [command_processor_proposal.md](command_processor_proposal.md)
Related: [hip_support_proposal.md](hip_support_proposal.md),
[pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md),
[chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)

## 1. Scope

This proposal specifies the **software implementation** of the
runtime API defined in §8 of the parent CP proposal. It covers the
new `sw/runtime/include/vortex2.h` header, its C++ implementation
across the per-backend trees, the legacy `vortex.h` shim work, build
integration, and the per-phase task breakdown that engineering can
execute against directly.

It does **not** redesign the API. Every signature, every type, every
flag in this document is taken from §8 of the parent proposal verbatim.

### 1.1 In scope

- **Full backend redesign**: drop the existing `sw/runtime/stub/`
  dispatcher pattern (`dlopen` + `callbacks_t`); replace with
  compile-time backend selection. Each backend produces a single
  `libvortex.so` containing both `vortex.h` legacy entry points and
  `vortex2.h` new entry points.
- **`vortex.h` is a wrapper over `vortex2.h` from day one** — not a
  phase-8 follow-on. Every legacy `vx_*` call resolves into one or
  more `vortex2.h` calls inside the same library. No parallel
  implementations.
- C++ class hierarchy for `vx::Device`, `vx::Queue`, `vx::Buffer`,
  `vx::Event` behind the public C handles.
- `vx::Platform` abstract interface; one subclass per backend
  (`PlatformSimX`, `PlatformRtlsim`, `PlatformXrt`).
- Per-queue ring buffer management in pinned host memory.
- Event seqnum machinery (signal slot, wait comparator, profile
  writeback parsing).
- Buffer map/unmap cache-coherence implementation.
- SimX backend full implementation (v1 in-process target — drives
  every existing legacy test through the new wrapper).
- XRT backend full implementation (v1 hardware target).
- rtlsim backend full implementation.
- Build-system rework: `./configure --backend={simx|rtlsim|xrt}`,
  single `libvortex.so` per build, no `libvortex-<name>.so` indirection.
- Unit-test, integration-test, and hardware-test plans.

### 1.2 Out of scope

- OPAE backend (deprecated per parent proposal §7.2; existing
  `sw/runtime/opae/` is deleted in commit 1b).
- Per-block helper headers (`vortex_tex.h`, `vortex_raster.h`,
  `vortex_om.h`, `vortex_dxa.h`) — owned by their respective
  subsystem proposals.
- Upper-layer API translators (POCL, chipStar, Vulkan-on-Vortex,
  CUDA-on-Vortex, etc.) — separate projects that consume `vortex2.h`.
- The RTL side of the CP — see [cp_rtl_impl_proposal.md](cp_rtl_impl_proposal.md).
- Multi-context KMU (phase 7 follow-on).
- Interrupt-driven completion (phase 6, v1.1).

## 2. File layout

The redesign **replaces** the existing dispatcher-based tree with a
flat per-backend layout. Every backend produces a single
`libvortex.so` containing both the legacy `vortex.h` API (as a thin
wrapper) and the new `vortex2.h` API (as the primary implementation).

```
sw/runtime/
├── include/
│   ├── vortex.h                       # KEPT, API unchanged. Implementation is the wrapper below.
│   └── vortex2.h                      # NEW — canonical async API (§8.11 of parent)
├── common/
│   ├── callbacks.{h,inc}              # UNCHANGED — instrumentation hooks (used by Platform impls)
│   ├── common.{h,cpp}                 # KEPT — MemoryAllocator still needed
│   ├── scope.{h,cpp}                  # UNCHANGED
│   ├── utils.cpp                      # UNCHANGED
│   ├── vortex2_internal.h             # NEW — vx::Device/Queue/Buffer/Event class decls + vx::Platform
│   ├── vx_result.cpp                  # NEW — vx_result_string + result enum helpers
│   ├── vx_device.cpp                  # NEW — vx::Device class (refcount, Platform owner, queues table)
│   ├── vx_queue.cpp                   # NEW — vx::Queue + per-queue ring-buffer mgmt
│   ├── vx_buffer.cpp                  # NEW — vx::Buffer + refcount + map/unmap
│   ├── vx_event.cpp                   # NEW — vx::Event + wait_all + profile readback
│   ├── vx_command_encoder.cpp         # NEW — cache-line framing helper (§5.7)
│   └── vortex_legacy_wrapper.cpp      # NEW — every vx_dev_open / vx_start / vx_copy_* / etc.
│                                      #       implemented as wrapper over vortex2.h calls.
│                                      #       Same binary, no dispatcher needed.
├── simx/
│   └── platform_simx.cpp              # NEW — vx::Platform subclass over the in-process simx model
├── rtlsim/
│   └── platform_rtlsim.cpp            # NEW — vx::Platform subclass over rtlsim
├── xrt/
│   ├── platform_xrt.cpp               # NEW — vx::Platform subclass over XRT
│   └── driver.{h,cpp}                 # KEPT — libxrt dynamic loader (consumed by platform_xrt.cpp)
├── Makefile                           # REWORKED — see §10
└── common.mk                          # REWORKED — see §10
```

**Deleted from the existing tree** in commit 1b:

```
sw/runtime/stub/                       # the dispatcher pattern + its callbacks_t indirection
sw/runtime/opae/                       # deprecated backend (parent §7.2)
sw/runtime/<backend>/vortex.cpp        # old C-API implementations per backend (legacy callbacks_t)
sw/runtime/stub/perf.cpp               # absorbed into common/utils.cpp or vortex_legacy_wrapper.cpp
```

Conventions:

- One `platform_<backend>.cpp` per backend. It defines a concrete
  subclass of `vx::Platform` and exports the single C-linkage symbol
  `vx::Platform* vx_create_platform()` — picked up by
  `vx::Device::open` at compile time (§3.1).
- All shared C++ machinery lives in `common/`, parameterized over
  the `vx::Platform` interface (§4.3).
- `vortex_legacy_wrapper.cpp` is built into **every** `libvortex.so`
  regardless of backend, because the legacy `vortex.h` API must work
  identically on every backend.
- No backend depends on any other backend's source. `--backend=simx`
  doesn't pull in rtlsim or xrt code, and vice versa.

## 3. Per-backend strategy

| Backend | v1 status                                                           | Notes                                                                  |
|---------|---------------------------------------------------------------------|------------------------------------------------------------------------|
| simx    | **Full implementation** — Platform subclass over the in-process simx model | Primary backend for unit testing and legacy compatibility. No real CP hardware in v1 — simx implements the wire protocol in-process. |
| rtlsim  | **Full implementation** — Platform subclass over rtlsim             | Same wire protocol as simx; uses rtlsim's RTL-driven model.            |
| xrt     | **Full implementation** — Platform subclass over the CP-aware AFU   | Drives real CP hardware (RTL commit 1a + 2 must be in place to run end-to-end). |
| opae    | **Deleted**                                                         | Per parent §7.2.                                                       |
| stub    | **Deleted**                                                         | The old dispatcher pattern goes away (§3.1).                           |

The build system (§10) selects exactly one backend per build via
`./configure --backend={simx,rtlsim,xrt}`. The output is a single
`libvortex.so` containing both `vortex.h` and `vortex2.h` symbols
implemented over that backend.

### 3.1 Backend dispatch model

vortex2.h uses **compile-time single-backend selection**. This is a
**deliberate departure** from the legacy `sw/runtime/stub/`
dispatcher pattern (which used `dlopen` of `libvortex-<NAME>.so`
based on the `VORTEX_DRIVER` env var). The legacy dispatcher is
**deleted** in commit 1b.

How the new selection works:

1. `./configure --backend=simx` writes `VORTEX_BACKEND=simx` into
   `build/config.mk`.
2. The runtime Makefile builds exactly one `platform_<backend>.cpp`
   into `libvortex.so`. Other backends' source files are not
   compiled or linked.
3. Each backend exports a single C-linkage factory function:

   ```cpp
   /* In each backend's platform_<backend>.cpp */
   extern "C" vx::Platform* vx_create_platform();
   ```

   `vx::Device::open` calls `vx_create_platform()` once at device
   open time and wraps the returned `Platform*` in the new
   `vx::Device` instance. Because `vx_create_platform` is defined in
   exactly one TU per build, the linker resolves it unambiguously.
4. Backend-specific link dependencies stay scoped to the chosen
   backend (xrt's `libxrt` loader, simx's `libsimx.so`, etc.) — they
   don't accumulate across builds.

**Why drop the old `dlopen` dispatcher?**

- The dispatcher exists only because the legacy build produced
  multiple per-backend libraries that needed runtime selection. The
  new build produces *one* `libvortex.so` per backend, picked at
  configure time, so there is nothing to dispatch between.
- One less indirection layer to maintain and debug. Stack traces
  become legible (`vx_dev_open` → `vx_device_open` → `Platform::*`
  directly, no `g_callbacks.*` in between).
- POCL, chipStar, SimX harnesses, kernel tests link against
  `libvortex.so` exactly as today — no rebuild needed because the
  ELF library name is unchanged.
- `VORTEX_DRIVER` env var becomes a no-op (silently ignored for
  backward compatibility with old scripts).

### 3.2 Legacy `vortex.h` is a wrapper over `vortex2.h` from day one

There is **no transition period**. Every legacy `vortex.h` entry
point (`vx_dev_open`, `vx_mem_alloc`, `vx_copy_to_dev`, `vx_start`,
`vx_ready_wait`, `vx_dcr_*`, `vx_mpm_query`, the `vx_upload_*`
utilities, etc.) is implemented as a thin C wrapper over the
corresponding `vortex2.h` call, in `common/vortex_legacy_wrapper.cpp`.
That one file is built into every backend's `libvortex.so`.

Concretely:

```cpp
/* sw/runtime/common/vortex_legacy_wrapper.cpp */

extern "C" int vx_dev_open(vx_device_h* hdev) {
    return result_to_int(vx_device_open(0, hdev));
}

extern "C" int vx_dev_close(vx_device_h hdev) {
    return result_to_int(vx_device_release(hdev));
}

extern "C" int vx_mem_alloc(vx_device_h hdev, uint64_t size, int flags,
                            vx_buffer_h* buf) {
    return result_to_int(vx_buffer_create(hdev, size, (uint32_t)flags, buf));
}

extern "C" int vx_mem_free(vx_buffer_h buf) {
    return result_to_int(vx_buffer_release(buf));
}

extern "C" int vx_copy_to_dev(vx_buffer_h buf, const void* src,
                              uint64_t off, uint64_t size) {
    auto* dev = handle_to_buffer(buf)->device();
    vx_queue_h q = legacy_default_queue(dev);   /* lazy per-device singleton */
    vx_event_h ev = nullptr;
    vx_result_t r = vx_enqueue_write(q, buf, off, src, size, 0, nullptr, &ev);
    if (r != VX_SUCCESS) return result_to_int(r);
    r = vx_event_wait_all(1, &ev, VX_MAX_TIMEOUT_NS);
    vx_event_release(ev);
    return result_to_int(r);
}

extern "C" int vx_start(vx_device_h hdev, vx_buffer_h kernel, vx_buffer_h args) {
    auto* dev = handle_to_device(hdev);
    vx_queue_h q = legacy_default_queue(dev);
    vx_launch_info_t li = make_launch_info_from_legacy_dcrs(dev, kernel, args);
    vx_event_h ev = nullptr;
    vx_result_t r = vx_enqueue_launch(q, &li, 0, nullptr, &ev);
    legacy_remember_last_event(dev, ev);   /* for vx_ready_wait */
    return result_to_int(r);
}

extern "C" int vx_ready_wait(vx_device_h hdev, uint64_t timeout_ms) {
    auto* dev = handle_to_device(hdev);
    vx_event_h ev = legacy_take_last_event(dev);
    if (!ev) return 0;
    auto r = vx_event_wait_all(1, &ev, timeout_ms * 1'000'000ull);
    vx_event_release(ev);
    return result_to_int(r);
}

/* … remaining vx_mem_* / vx_dcr_* / vx_upload_* wrappers … */
```

Each backend's `Platform` subclass implements the per-call hooks
required by `vortex2.h`; the legacy wrapper file is backend-agnostic
because it only calls into `vortex2.h` — exactly the same code path
the new API uses.

Implications:

- **Zero behavioral regression** for legacy callers. Every existing
  test (vecadd on simx, the regression suite, POCL, chipStar) should
  pass byte-identically after the redesign because the public
  `vortex.h` surface is unchanged and the underlying execution is the
  same Platform implementation that backed it before.
- **One backend implementation per backend.** Backends no longer
  implement `callbacks_t` for legacy *and* `vortex2.h` symbols
  separately; they implement only `vx::Platform`. The legacy wrapper
  builds on top once.
- **Phase 8 of the original migration plan disappears.** What was
  "follow-on: re-implement vortex.h as a shim" is folded into commit
  1b itself.

`legacy_default_queue(dev)` is a small TLS-keyed singleton stored on
the `vx::Device` instance — created lazily on the first legacy call
that needs a queue, destroyed at `vx_dev_close` time. Legacy callers
never see the queue handle. Multi-threaded legacy code gets the same
implicit single-queue semantics it had before.

## 4. Core class design

### 4.1 Handle ↔ class relationship

The public `vx_*_h` handles in `vortex2.h` are opaque struct pointers
that resolve to internal C++ classes:

| Public handle | Internal class       | Header                             |
|---------------|----------------------|------------------------------------|
| `vx_device_h` | `vx::Device`         | `common/vortex2_internal.h`        |
| `vx_buffer_h` | `vx::Buffer`         | `common/vortex2_internal.h`        |
| `vx_queue_h`  | `vx::Queue`          | `common/vortex2_internal.h`        |
| `vx_event_h`  | `vx::Event`          | `common/vortex2_internal.h`        |

Inherited `vx_device_h` and `vx_buffer_h` keep their `void*` typedefs
in `vortex.h` for ABI compatibility (parent §8.2). At runtime they
point to the same `vx::Device` / `vx::Buffer` instances — the cast
happens at the C-API boundary.

### 4.2 Refcounting

All four classes derive from a single CRTP base:

```cpp
template <class T>
class RefCounted {
public:
    void retain()  { ++refs_; }
    bool release() {
        if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete static_cast<T*>(this);
            return true;
        }
        return false;
    }
    uint32_t refs() const { return refs_.load(std::memory_order_relaxed); }
private:
    std::atomic<uint32_t> refs_ { 1 };   // created with one reference
};
```

Public `vx_*_retain` / `vx_*_release` are one-line wrappers that
unwrap the handle and call into `RefCounted`.

### 4.3 Backend abstraction (`vx::Platform`)

To keep `common/` backend-agnostic, all platform-specific behavior
goes through a pure-virtual `vx::Platform` interface:

```cpp
namespace vx {

class Platform {
public:
    virtual ~Platform() = default;

    /* ----- AXI-Lite MMIO ----- */
    virtual vx_result_t mmio_write32(uint32_t off, uint32_t value) = 0;
    virtual vx_result_t mmio_read32 (uint32_t off, uint32_t* out)  = 0;

    /* ----- Pinned host memory ----- */
    virtual vx_result_t pinned_alloc(size_t size, void** out_ptr,
                                     uint64_t* out_io_addr) = 0;
    virtual vx_result_t pinned_free (void* ptr) = 0;

    /* ----- Device memory (allocator state lives in vx::Device) ----- */
    virtual vx_result_t dev_alloc   (size_t size, uint32_t flags,
                                     uint64_t* out_dev_addr) = 0;
    virtual vx_result_t dev_free    (uint64_t dev_addr) = 0;

    /* ----- Cache-coherence primitives for map/unmap ----- */
    virtual void cache_flush      (void* p, size_t size) = 0;
    virtual void cache_invalidate (void* p, size_t size) = 0;
};

} // namespace vx
```

XRT, SimX, rtlsim, and stub each provide a concrete subclass. The
stub Platform implements MMIO as writes to a plain memory buffer
the unit test harness can inspect.

### 4.4 `vx::Device`

```cpp
namespace vx {

class Device : public RefCounted<Device> {
public:
    static vx_result_t open(uint32_t index, vx_device_h* out);

    /* Public API entry points (called from vortex2.h C wrappers) */
    vx_result_t query(uint32_t caps_id, uint64_t* out);
    vx_result_t memory_info(uint64_t* free, uint64_t* used);

    /* Internal */
    Platform&            platform() { return *platform_; }
    MemoryAllocator&     allocator() { return allocator_; }
    uint32_t             alloc_queue_id();
    void                 release_queue_id(uint32_t qid);
    uint64_t             cycle_freq_hz() const { return cycle_freq_hz_; }

private:
    Device(std::unique_ptr<Platform>);
    ~Device();

    std::unique_ptr<Platform>  platform_;
    MemoryAllocator            allocator_;    // device address space mgr (existing)
    std::mutex                 queue_id_mu_;
    std::bitset<NUM_QUEUES>    queue_id_in_use_;
    uint64_t                   cycle_freq_hz_; // read once from CP_CYCLE_FREQ_HZ
    DeviceCaps                 caps_;          // cached at open
};

} // namespace vx
```

### 4.5 `vx::Buffer`

```cpp
namespace vx {

class Buffer : public RefCounted<Buffer> {
public:
    static vx_result_t create (Device* dev, uint64_t size, uint32_t flags,
                               vx_buffer_h* out);
    static vx_result_t reserve(Device* dev, uint64_t addr, uint64_t size,
                               uint32_t flags, vx_buffer_h* out);

    vx_result_t address(uint64_t* out)        const;
    vx_result_t access (uint64_t off, uint64_t size, uint32_t flags);
    vx_result_t map    (uint64_t off, uint64_t size, uint32_t flags, void** out);
    vx_result_t unmap  (void* host_ptr);

    /* Internal — used by Queue::enqueue_* to keep buffers alive
     * across in-flight commands (parent §8.5). */
    void in_flight_retain()  { retain(); }
    void in_flight_release() { release(); }

private:
    Device*  device_;
    uint64_t dev_addr_;
    uint64_t size_;
    uint32_t flags_;            // VX_MEM_READ/WRITE/READ_WRITE/PIN_MEMORY

    /* Mapping state (only used when VX_MEM_PIN_MEMORY) */
    std::mutex   map_mu_;
    void*        host_ptr_     = nullptr;  // pinned host VA
    uint64_t     host_io_addr_ = 0;        // FPGA-visible IO address
    uint32_t     map_count_    = 0;        // nested-map count

    /* When the buffer is *not* PIN_MEMORY, map() returns NOT_SUPPORTED. */
};

} // namespace vx
```

### 4.6 `vx::Queue`

```cpp
namespace vx {

class Queue : public RefCounted<Queue> {
public:
    static vx_result_t create(Device* dev, const vx_queue_info_t* info,
                              vx_queue_h* out);

    vx_result_t flush();
    vx_result_t finish(uint64_t timeout_ns);

    vx_result_t enqueue_launch (const vx_launch_info_t* info,
                                uint32_t nw, const vx_event_h* w,
                                vx_event_h* out);
    vx_result_t enqueue_copy   (Buffer* dst, uint64_t do_, Buffer* src,
                                uint64_t so, uint64_t sz,
                                uint32_t nw, const vx_event_h* w,
                                vx_event_h* out);
    vx_result_t enqueue_read   (void* host, Buffer* src, uint64_t so, uint64_t sz,
                                uint32_t nw, const vx_event_h* w, vx_event_h* out);
    vx_result_t enqueue_write  (Buffer* dst, uint64_t off, const void* host,
                                uint64_t sz, uint32_t nw, const vx_event_h* w,
                                vx_event_h* out);
    vx_result_t enqueue_barrier(uint32_t nw, const vx_event_h* w, vx_event_h* out);
    vx_result_t enqueue_dcr_write(uint32_t addr, uint32_t value,
                                  uint32_t nw, const vx_event_h* w, vx_event_h* out);
    vx_result_t enqueue_dcr_read (uint32_t addr, uint32_t* host_dst,
                                  uint32_t nw, const vx_event_h* w, vx_event_h* out);

private:
    Queue(Device*, uint32_t qid, const vx_queue_info_t&);
    ~Queue();

    /* Implementation helpers */
    vx_result_t emit_command   (CommandEncoder& enc);
    vx_result_t emit_wait_list (CommandEncoder& enc,
                                uint32_t nw, const vx_event_h* w);
    Event*      alloc_event    (bool profiled);
    void        write_doorbell (uint64_t tail);

    Device*               device_;
    uint32_t              qid_;            // 0..NUM_QUEUES-1
    uint32_t              priority_;
    bool                  profile_en_;

    /* Pinned ring buffer */
    void*                 ring_ptr_;       // host VA
    uint64_t              ring_io_addr_;   // FPGA-visible
    size_t                ring_bytes_;     // 2^VX_CP_RING_SIZE_LOG2
    std::atomic<uint64_t> tail_;           // byte offset, host-side producer
    /* head_ lives in pinned host memory written by CP; we just read it */
    uint64_t*             head_slot_ptr_;
    uint64_t              head_slot_io_addr_;

    /* Completion seqnum slot (CP writes; host reads) */
    uint64_t*             cmpl_slot_ptr_;
    uint64_t              cmpl_slot_io_addr_;
    std::atomic<uint64_t> next_seqnum_;    // host-side monotonic counter

    /* Pool of event slots (so we don't pin-alloc per event) */
    EventSlotPool         event_slots_;

    /* Pool of profile slots (32B each); enabled when profile_en_ */
    ProfileSlotPool       profile_slots_;

    std::mutex            enqueue_mu_;     // serializes host-side ring writes
};

} // namespace vx
```

### 4.7 `vx::Event`

```cpp
namespace vx {

class Event : public RefCounted<Event> {
public:
    static vx_result_t user_create(Device* dev, vx_event_h* out);
    static vx_result_t user_signal(Event* ev, vx_result_t status);

    vx_result_t status     (vx_event_status_e* out);
    vx_result_t wait       (uint64_t timeout_ns);
    vx_result_t get_profile(vx_profile_info_t* out);

    /* Internal — used by Queue::enqueue_* */
    void   bind(Queue* q, uint64_t seqnum, uint64_t* slot_ptr,
                uint64_t slot_io_addr, ProfileSlot* prof);
    bool   is_user() const         { return source_queue_ == nullptr; }
    uint64_t  expected_seqnum() const  { return expected_seqnum_; }
    uint64_t  signal_io_addr()   const { return slot_io_addr_; }

private:
    Queue*    source_queue_      = nullptr;   // NULL = user event
    uint64_t  expected_seqnum_   = 0;
    uint64_t* slot_ptr_          = nullptr;   // host VA of signal slot
    uint64_t  slot_io_addr_      = 0;         // FPGA-visible
    ProfileSlot* profile_slot_   = nullptr;   // NULL if not profiled
};

/* static wait helper used by both vx_event_wait_all and Queue::finish */
vx_result_t wait_all(Event** events, uint32_t n, uint64_t timeout_ns);

} // namespace vx
```

## 5. Per-queue ring buffer management

### 5.1 Allocation

At `vx_queue_create`:

1. `Device::alloc_queue_id()` returns a free queue id in `[0, NUM_QUEUES)`
   under `queue_id_mu_`.
2. `Platform::pinned_alloc` allocates `2^VX_CP_RING_SIZE_LOG2` bytes
   for the ring + 8 B for `head_slot` + 8 B for `cmpl_slot` (one
   allocation, sub-page-aligned slots).
3. Allocate a small pool of event slots (default 256 × 8 B) and, if
   `profile_en`, a pool of profile slots (default 64 × 32 B).
4. Write the per-queue AXI-Lite registers (parent §6.10):
   `Q_RING_BASE_*`, `Q_HEAD_ADDR_*`, `Q_CMPL_ADDR_*`,
   `Q_RING_SIZE_LOG2`, `Q_CONTROL` with `enable=1`, `priority`,
   `profile_en`.

### 5.2 Doorbell coalescing

Naive: write `Q_TAIL_*` after every `enqueue_*`. Wastes MMIO bandwidth
for back-to-back enqueues.

Strategy:

- Track `pending_tail_` (the value we want the CP to see).
- Skip the doorbell write if the CP's observed `head` is far behind
  `pending_tail_` AND the ring isn't close to full — the CP will
  catch up on its next fetch cycle without prompting.
- Always doorbell at `vx_queue_flush` and inside `vx_queue_finish`.
- Always doorbell when ring occupancy exceeds 50% — the CP must keep
  draining to avoid back-pressuring the producer.
- Always doorbell when a `CMD_LAUNCH` is enqueued (low-frequency,
  worth the wake-up).

Implementation: `Queue::write_doorbell(tail)` is the central point;
all enqueue paths route through it.

### 5.3 Tail / head bookkeeping

`tail_` is `std::atomic<uint64_t>` to allow lock-free reads from a
status thread (later), even though writes are serialized under
`enqueue_mu_`. `head_slot_ptr_` is `uint64_t*` into pinned memory
written by the CP; reads use `std::atomic_ref<uint64_t>` with
acquire semantics.

Wrap-around: ring is power-of-two sized. Byte offsets mask via
`offset & (ring_bytes_ - 1)`. Free space is
`ring_bytes_ - (tail - head)`; full when this hits zero.

### 5.4 Backpressure

If a `Queue::enqueue_*` finds insufficient free space:

1. Write the doorbell unconditionally to wake the CP.
2. Spin with exponential backoff on the head slot for up to
   `VX_CP_ENQUEUE_BACKPRESSURE_NS` (default 1 ms).
3. If still full, return `VX_ERR_OUT_OF_HOST_MEMORY`.

Callers can pre-flush with `vx_queue_finish` if they hit this.

### 5.5 Command encoding

A `CommandEncoder` accumulates a single command into a thread-local
64-byte staging buffer, then atomically copies it into the ring at
the reserved tail offset. This keeps the cache-line-framing rule
from the parent §6.3 enforced in one place:

```cpp
class CommandEncoder {
public:
    explicit CommandEncoder(uint32_t opcode, uint8_t flags);
    void put32(uint32_t);
    void put64(uint64_t);
    void put_bytes(const void*, size_t);
    size_t size() const;
    const uint8_t* data() const;
};
```

Per-command `emit_*` helpers build the encoder, then `Queue::emit_command`
reserves `size()` bytes in the ring (after rounding the tail to a CL
boundary if the new command wouldn't fit in the current line), memcpys
the encoded bytes in, and updates `tail_`.

### 5.6 Wait-list expansion

`Queue::emit_wait_list(enc, nw, w)` is called before every enqueue:

```cpp
for (uint32_t i = 0; i < nw; ++i) {
    Event* ev = handle_to_event(w[i]);
    if (ev->is_user() || ev->source_queue_ != this) {
        // emit CMD_EVENT_WAIT(ev->signal_io_addr(), ev->expected_seqnum(), GE)
        emit_event_wait_cmd(enc, ev);
    }
    // events from this same queue are subsumed by in-order semantics — skip
}
```

For long lists (>4 external events), a future optimization can
synthesize a merged event in software; v1 just emits one
`CMD_EVENT_WAIT` per external event.

### 5.7 Event signaling

Every `Queue::enqueue_*` that returns an `out_event` performs:

1. `alloc_event(profiled)` returns a fresh `Event` bound to the next
   seqnum on this queue and to a slot from the queue's event-slot
   pool (and a profile slot if `F_PROFILE`).
2. Encoder appends a `CMD_EVENT_SIGNAL(slot_io_addr, seqnum)` after
   the main command's payload.
3. Caller-visible `vx_event_h` points to the bound `Event`.

`Event::wait()` and `Event::status()` read `*slot_ptr_` with
acquire-load semantics and compare to `expected_seqnum_`.

## 6. Buffer map/unmap

### 6.1 Eligibility

`vx_buffer_map` returns `VX_ERR_NOT_SUPPORTED` unless `flags_ &
VX_MEM_PIN_MEMORY` is set at create time. Pinned buffers are
allocated via `Platform::pinned_alloc` and carry both `host_ptr_`
and `host_io_addr_`.

### 6.2 Map

```cpp
vx_result_t Buffer::map(uint64_t off, uint64_t size, uint32_t flags,
                        void** out) {
    if (!(flags_ & VX_MEM_PIN_MEMORY)) return VX_ERR_NOT_SUPPORTED;
    if (off + size > size_)            return VX_ERR_INVALID_VALUE;
    std::lock_guard g(map_mu_);
    ++map_count_;
    /* Invalidate CPU cache so we see whatever the GPU last wrote.
     * Required after VX_MEM_READ map; harmless for write-only. */
    if (flags & VX_MEM_READ) {
        device_->platform().cache_invalidate(
            static_cast<uint8_t*>(host_ptr_) + off, size);
    }
    *out = static_cast<uint8_t*>(host_ptr_) + off;
    return VX_SUCCESS;
}
```

### 6.3 Unmap

```cpp
vx_result_t Buffer::unmap(void* host_ptr) {
    std::lock_guard g(map_mu_);
    if (map_count_ == 0) return VX_ERR_INVALID_VALUE;
    --map_count_;
    /* Flush any pending CPU stores so the GPU sees them. We can't
     * track per-unmap whether the user wrote, so flush the whole
     * mapped range conservatively. Map-for-read is no-op here. */
    /* TODO(perf): track per-map flags to skip flush on read-only maps. */
    size_t offset = static_cast<uint8_t*>(host_ptr) -
                    static_cast<uint8_t*>(host_ptr_);
    device_->platform().cache_flush(host_ptr, size_ - offset);
    return VX_SUCCESS;
}
```

On x86_64, `cache_flush` is `clflushopt` + `mfence` over the range;
`cache_invalidate` is the same sequence (Intel guarantees `clflushopt`
invalidates as well). On other ISAs the Platform implementation
provides equivalents.

## 7. Profiling

### 7.1 Per-event profile slot

When `profile_en_` is set on the queue and an enqueue allocates an
event, `alloc_event(profiled=true)` also reserves a 32 B profile
slot from `profile_slots_` and binds it to the event. The encoder
sets `F_PROFILE` in the command header and appends `slot_io_addr` to
the command payload (parent §6.5, §6.11).

Slot layout: `{queued_ns, submit_ns, start_ns, end_ns}`, each
`uint64_t`. The CP writes the latter three in raw cycles; the host
side fills `queued_ns` before ringing the doorbell.

### 7.2 Cycle ↔ ns conversion

At `Device::open`:

```cpp
platform_->mmio_read32(CP_CYCLE_FREQ_HZ, &freq);
cycle_freq_hz_ = freq;
```

`Event::get_profile` reads the 32 B slot and converts each cycle
value: `ns = cycles * 1'000'000'000 / cycle_freq_hz_`.

### 7.3 Slot reclaim

Profile slots are returned to the queue's `ProfileSlotPool` when the
last reference to the parent `Event` is released. This means an
event the user retains forever pins its profile slot — documented
behavior; matches CUDA `cudaEvent_t` semantics.

## 8. Legacy `vortex.h` wrapper (commit 1b)

The full-redesign approach (§3.2) collapses the original migration
plan's phase 8 into commit 1b. Every legacy backend's `vortex.cpp` is
deleted; a single `common/vortex_legacy_wrapper.cpp` implements every
legacy `vx_*` function over `vortex2.h` primitives. Mapping is in §9
of the parent proposal; representative implementations:

```cpp
extern "C" int vx_dev_open(vx_device_h* hdev) {
    return result_to_int(vx_device_open(0, hdev));
}

extern "C" int vx_dev_close(vx_device_h hdev) {
    return result_to_int(vx_device_release(hdev));
}

extern "C" int vx_copy_to_dev(vx_buffer_h buf, const void* src,
                              uint64_t off, uint64_t size) {
    auto* dev = handle_to_buffer(buf)->device();
    vx_queue_h q = legacy_default_queue(dev);   // lazy-created, one per device
    vx_event_h ev = nullptr;
    vx_result_t r = vx_enqueue_write(q, buf, off, src, size, 0, nullptr, &ev);
    if (r != VX_SUCCESS) return result_to_int(r);
    r = vx_event_wait_all(1, &ev, VX_MAX_TIMEOUT_NS);
    vx_event_release(ev);
    return result_to_int(r);
}

extern "C" int vx_start(vx_device_h hdev, vx_buffer_h kernel,
                        vx_buffer_h args) {
    vx_queue_h q = legacy_default_queue(handle_to_device(hdev));
    vx_launch_info_t li = make_launch_info_from_legacy_dcrs(kernel, args);
    vx_event_h ev = nullptr;
    vx_result_t r = vx_enqueue_launch(q, &li, 0, nullptr, &ev);
    legacy_remember_last_event(hdev, ev);   // for vx_ready_wait
    return result_to_int(r);
}

extern "C" int vx_ready_wait(vx_device_h hdev, uint64_t timeout) {
    vx_event_h ev = legacy_take_last_event(hdev);
    if (!ev) return 0;   // nothing pending
    auto r = vx_event_wait_all(1, &ev, timeout * 1'000'000ull);
    vx_event_release(ev);
    return result_to_int(r);
}
```

`legacy_default_queue` lives in shim TLS keyed by `vx_device_h` and
is destroyed on `vx_dev_close`. Legacy callers see exactly the same
synchronous semantics they always have; new callers can mix
`vortex2.h` calls freely.

Because the wrapper lands in commit 1b alongside the new runtime,
the AFU's MMIO compatibility mode can be retired as soon as commit 1c
(CP RTL integration) brings the new control path online. See parent
proposal §9.3.

## 9. Test backend strategy

There is no separate "mock" or "stub" backend in this redesign — the
original proposal's §9 ("Stub backend") is dropped. Per §3.2, every
backend (simx, rtlsim, xrt) is a full Platform implementation and
serves as both the production target and the unit-test target.

Commit 1b's smoke verification target is **simx**: in-process,
deterministic, no FPGA required. The minimal smoke test
([tests/runtime/test_basic.cpp](../../tests/runtime/test_basic.cpp))
links against `libvortex.so` (simx backend) and exercises both legacy
`vortex.h` entry points and new `vortex2.h` entry points end-to-end.
A `PASSED` exit is the commit's verification gate.

## 10. Build system integration

### 10.1 Backend selection

```
make -C sw/runtime BACKEND=simx     (default)
make -C sw/runtime BACKEND=rtlsim
```

The top-level `sw/runtime/Makefile` defaults to `simx`. xrt support
returns in commit 1c (when the CP RTL lands and the AXI shim work is
ready). OPAE is permanently retired per parent §7.2.

### 10.2 Per-backend `Makefile`s

Each backend's `Makefile` (`sw/runtime/<name>/Makefile`) compiles:

- `platform_<name>.cpp` — the backend's `vx::Platform` subclass.
- `common/vx_result.cpp` + `vx_device.cpp` + `vx_buffer.cpp` +
  `vx_queue.cpp` + `vx_event.cpp` — vortex2.h runtime, backend-agnostic.
- `common/vortex_legacy_wrapper.cpp` + `legacy_utils.cpp` +
  `legacy_perf.cpp` + `utils.cpp` — vortex.h C wrappers + helpers.

into a single `libvortex.so` per build. No `libvortex-<name>.so`
indirection; no `dlopen` dispatcher.

### 10.3 Out-of-tree builds

Per the project convention ([feedback-out-of-tree-builds]), all
build artifacts land under `build/`. `configure` (in the build dir)
copies the per-backend Makefiles into `build/sw/runtime/<backend>/`
and the build does not touch the source tree. Any edit to a source
Makefile requires a re-run of `../configure` to take effect
([feedback-vortex-configure-copies-makefiles]).

## 11. Test plan

### 11.1 Smoke test (commit 1b verification gate)

[tests/runtime/test_basic.cpp](../../tests/runtime/test_basic.cpp)
links against `libvortex.so` (simx backend) and exercises:

- `vx_dev_open` + `vx_dev_close` (legacy → wrapper → `vx_device_open`/`release`)
- `vx_dev_caps` vs `vx_device_query` (compare legacy and new — must match)
- `vx_mem_alloc` (legacy) + `vx_buffer_release` (new) — cross-API
- `vx_buffer_create` (new) + `vx_buffer_address` + `vx_mem_free` (legacy) — cross-API
- `vx_queue_create` + `vx_queue_release`
- `vx_user_event_create` + `vx_event_status` + `vx_user_event_signal` + `vx_event_wait_all`
- Refcount semantics: `vx_buffer_retain` defers actual free until balanced release

Run with `make -C tests/runtime run` under a 120 s cap
([feedback-test-timeout-120s]). Verification gate: `PASSED` exit + 0
return code.

### 11.2 Expanded unit tests (post-commit-1b)

Future commits in this phase will add coverage for:

- Ring buffer wrap-around, backpressure, doorbell coalescing
  (relevant once CP RTL lands — commit 1c).
- Cross-queue event waits.
- Profile timestamp readback, including cycle→ns conversion.
- Map/unmap on PIN_MEMORY buffers (currently the wrapper falls back
  to staging copies — see §6.2).
- Concurrent enqueue from multiple host threads.

### 11.2 Integration tests (xrt backend on FPGA hardware)

Hosted on the self-hosted runner ([project-ci-machine]):

- Smoke: `tests/kernel/vecadd` ported to `vortex2.h` async DAG (the
  worked example from parent §8.9).
- Profile: same workload with `VX_QUEUE_PROFILING_ENABLE` verifies
  monotonically increasing QUEUED < SUBMIT < START < END.
- Multi-queue overlap: 2 queues, one DMA-only, one compute-only;
  measure wall time vs serialized baseline (expect ≥1.4× speedup on
  workloads with similar copy/compute durations).
- Cross-queue events: 3-queue DAG (H2D on Q0, kernel on Q1, D2H on
  Q2, all gated by events) — correctness only, no perf claim.

### 11.3 Hardware bring-up tests (xrt)

Phase 2 deliverable: smallest possible exercise that proves the CP
RTL + runtime are wired correctly. Just `vx_device_open` →
`vx_queue_create` → `vx_enqueue_write` (4 KB to device) →
`vx_event_wait_all` → `vx_enqueue_read` (4 KB from device) →
`vx_event_wait_all` → memcmp.

### 11.4 POCL / chipStar integration tests

Outside the scope of this proposal; tracked in the POCL and chipStar
proposals. The runtime project provides the `vortex2.h` library and
a minimum-conformance smoke test; POCL/chipStar own their own
conformance harnesses.

## 12. Phased implementation tasks

Aligns with parent proposal §13 migration plan, with the original
"phase 8 legacy shim" folded into commit 1b (full-redesign approach
per §3.2).

### Commit 1b — full runtime redesign (this commit) ✅

- [x] `include/vortex2.h` with the complete API surface (parent §8.11).
- [x] `common/vortex2_internal.h` — `vx::Device/Queue/Buffer/Event` +
      `vx::Platform`.
- [x] `common/vx_result.cpp` + `vx_device.cpp` + `vx_buffer.cpp` +
      `vx_queue.cpp` + `vx_event.cpp`.
- [x] `common/vortex_legacy_wrapper.cpp` — every legacy `vx_*` entry
      point implemented over `vortex2.h`.
- [x] `simx/platform_simx.cpp` + `rtlsim/platform_rtlsim.cpp` —
      `vx::Platform` subclasses over the existing in-process simulators.
- [x] Deleted: `stub/` (the old dispatcher), `opae/` (deprecated),
      `xrt/` (deferred to commit 1c), per-backend `vortex.cpp` files,
      `common/callbacks.{h,inc}` (dispatcher abstraction gone).
- [x] Rewritten build system: single `libvortex.so` per build, no
      `libvortex-<name>.so` indirection, `BACKEND=simx|rtlsim` selector.
- [x] `tests/runtime/test_basic.cpp` smoke test: PASSED on simx.

### Commit 1c — XRT backend + CP RTL integration (depends on RTL phase 2)

- [ ] `xrt/platform_xrt.cpp` — `vx::Platform` subclass over the
      CP-aware XRT AFU shell.
- [ ] AXI register-block decode for the new CP doorbells (parent §6.10).
- [ ] Replace the simx/rtlsim "fake-async" launch path with real
      ring-buffer submission to the CPE (when the CP RTL is online).
- [ ] Hardware smoke: vecadd via `vortex2.h` async path on FPGA.

### Commit 1d — N CPEs + events + barriers + profiling (depends on RTL phases 3-4)

- [ ] Per-queue ring-buffer allocation, doorbell, completion seqnum.
- [ ] Wait-list expansion in `Queue::emit_wait_list`.
- [ ] `enqueue_barrier`, `enqueue_dcr_write`, `enqueue_dcr_read`.
- [ ] `ProfileSlotPool`, `F_PROFILE` flag emission, `Event::get_profile`.
- [ ] `Buffer::map` / `Buffer::unmap` with cache flush/invalidate
      (replaces current heap-mirror fallback in §6).
- [ ] OpenCL 1.2 conformance smoke via POCL backed by `vortex2.h`.

### Commit 1e — perf pass (timing-driven)

Doorbell coalescing, head-write batching, ring-buffer pinning
optimizations. Driven by phase-4 perf measurements on hardware.

## 13. Open implementation questions

1. **Thread-local default queue lookup in the legacy shim.** Phase 8
   needs `legacy_default_queue(dev)` to be cheap. TLS keyed on
   `vx_device_h` is one option; an inline cache in the device handle
   is another. Decide before phase 8 starts.
2. **Profile-slot lifetime when the user never calls
   `vx_event_get_profile`.** Slot is currently held until event
   refcount drops; that's correct but a long-held event leaks a slot.
   Should the pool be sized to cover worst-case in-flight events
   only, with a slow fallback to malloc?
3. **Doorbell coalescing heuristic tuning.** v1 uses the simple "skip
   if CP is behind, force if >50% full." Measure on the smoke test
   in phase 5; adjust.
4. **`Buffer::map` for non-pinned buffers.** Returning
   `VX_ERR_NOT_SUPPORTED` is conservative but loses functionality
   that some upper layers (older OpenCL apps using `clEnqueueMapBuffer`
   on device-only buffers) expect. Should v1.1 add an internal
   "stage via DMA" fallback?
5. **Hot-path allocation.** `alloc_event(profiled)` and `CommandEncoder`
   construction are on the enqueue hot path. v1 uses freelist pools;
   if that proves insufficient under heavy load, switch to per-thread
   caches.

## 14. References

- [docs/proposals/command_processor_proposal.md](command_processor_proposal.md)
  — parent architecture proposal; this document implements §8 and §9 from there.
- [docs/proposals/cp_rtl_impl_proposal.md](cp_rtl_impl_proposal.md)
  — companion RTL implementation proposal.
- [sw/runtime/include/vortex.h](../../sw/runtime/include/vortex.h)
  — legacy public API; phase 8 re-implements it over vortex2.h.
- [docs/proposals/pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md)
  — POCL backend that will consume `vortex2.h`.
- [docs/proposals/chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)
  — chipStar HIP/OpenCL backend that will consume `vortex2.h`.
