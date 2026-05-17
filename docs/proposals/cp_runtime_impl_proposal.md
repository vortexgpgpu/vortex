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

- C++ class hierarchy for `vx_device`, `vx_queue`, `vx_buffer`,
  `vx_event`.
- Per-queue ring buffer management in pinned host memory.
- Event seqnum machinery (signal slot, wait comparator, profile
  writeback parsing).
- Buffer map/unmap cache-coherence implementation.
- XRT backend full implementation (v1 target).
- SimX / rtlsim / stub backends as v1 stubs returning
  `VX_ERR_NOT_SUPPORTED` for CP-only operations.
- Legacy `vortex.h` shim re-implementation (phase 8).
- Build-system integration (Makefile, configure, conditional
  compilation).
- Unit-test, integration-test, and hardware-test plans.

### 1.2 Out of scope

- OPAE backend (deprecated per parent proposal §7.2).
- Per-block helper headers (`vortex_tex.h`, `vortex_raster.h`,
  `vortex_om.h`, `vortex_dxa.h`) — owned by their respective
  subsystem proposals.
- Upper-layer API translators (POCL, chipStar, Vulkan-on-Vortex,
  CUDA-on-Vortex, etc.) — separate projects that consume `vortex2.h`.
- The RTL side of the CP — see [cp_rtl_impl_proposal.md](cp_rtl_impl_proposal.md).
- Multi-context KMU (phase 7 follow-on).
- Interrupt-driven completion (phase 6, v1.1).

## 2. File layout

```
sw/runtime/
├── include/
│   ├── vortex.h                       # UNCHANGED in v1 (legacy public API)
│   └── vortex2.h                      # NEW — async public API (§8 of parent)
├── common/
│   ├── callbacks.{h,inc}              # UNCHANGED — instrumentation hooks
│   ├── common.{h,cpp}                 # MODIFIED — MemoryAllocator extended for retain
│   ├── scope.{h,cpp}                  # UNCHANGED
│   ├── vortex2_internal.h             # NEW — internal C++ class declarations
│   ├── vx_device.cpp                  # NEW — vx_device class implementation
│   ├── vx_queue.cpp                   # NEW — vx_queue class + ring-buffer mgmt
│   ├── vx_buffer.cpp                  # NEW — vx_buffer class + map/unmap
│   ├── vx_event.cpp                   # NEW — vx_event class + wait machinery
│   ├── vx_command_encoder.cpp         # NEW — fills ring-buffer cache lines (§5.7)
│   └── vortex2_legacy_shim.cpp        # NEW (phase 8) — legacy vortex.h over vortex2.h
├── xrt/
│   ├── vortex.cpp                     # UNCHANGED until phase 8 (then deleted)
│   ├── vortex2_xrt.cpp                # NEW — XRT-specific vx_device::open, AXI surface
│   ├── vortex2_xrt_axi.{h,cpp}        # NEW — wraps xrt::ip / xrt::bo for AXI access
│   └── driver.{h,cpp}                 # UNCHANGED — dynamic loader for libxrt
├── simx/
│   ├── vortex.cpp                     # UNCHANGED — legacy backend
│   └── vortex2_simx.cpp               # NEW (stub in v1) — returns VX_ERR_NOT_SUPPORTED
├── rtlsim/
│   ├── vortex.cpp                     # UNCHANGED
│   └── vortex2_rtlsim.cpp             # NEW (stub in v1)
├── stub/
│   ├── vortex.cpp                     # UNCHANGED
│   └── vortex2_stub.cpp               # NEW — in-memory mock backend for unit tests
├── opae/                              # NOT BUILT in v1 (parent §7.2)
├── Makefile                           # MODIFIED — see §10
└── common.mk                          # MODIFIED — see §10
```

Conventions:

- Every `vortex2_*.cpp` is a v1 deliverable, even if it's a stub.
  This keeps the symbol surface uniform across backends.
- Legacy `vortex.cpp` per backend is **not** modified in phases 1-7;
  it is replaced wholesale by `vortex2_legacy_shim.cpp` in phase 8.
- All shared C++ machinery lives in `common/`, parameterized over a
  backend "platform" interface (§4.3).

## 3. Per-backend strategy

| Backend | Phase 1-4 status                                                    | Notes                                                                  |
|---------|---------------------------------------------------------------------|------------------------------------------------------------------------|
| xrt     | **Full vortex2.h implementation** through the CP                    | Only target that drives real CP hardware in v1.                        |
| simx    | Stub: queue/enqueue/event return `VX_ERR_NOT_SUPPORTED`             | Legacy `vortex.h` path keeps working. CP support deferred to phase X.  |
| rtlsim  | Stub: same as simx                                                  | Lets rtlsim users keep running legacy tests.                           |
| stub    | **Full in-memory mock** of `vortex2.h` (no HW, no CP, no simulator) | For unit testing the runtime independent of any backend.               |
| opae    | Not built                                                           | Architecture proposal §7.2.                                            |

The build system (§10) selects exactly one backend per build via
`./configure --backend={xrt,simx,rtlsim,stub}`. The stub backend is
also built as a static library used by the unit test harness.

### 3.1 Backend dispatch model

vortex2.h uses **compile-time single-backend selection** — there is no
runtime dispatch table, no `dlopen` of a backend plugin, no abstract
factory registry. The choice is:

1. `./configure --backend=xrt` writes the selected backend name into
   `build/config.mk`.
2. The Makefile links exactly one `vortex2_<backend>.cpp` into
   `libvortex.so` per build, matching what legacy `vortex.h` already
   does (one `vortex.cpp` per backend, picked at configure time).
3. Every backend exports a single C-linkage factory function:

   ```cpp
   /* In each backend's vortex2_<backend>.cpp */
   extern "C" std::unique_ptr<vx::Platform> vx_make_platform(uint32_t index);
   ```

   `vx::Device::open(index, &dev)` calls `vx_make_platform(index)` once
   and stores the returned `unique_ptr` in the new `vx::Device`
   instance. Because `vx_make_platform` is defined in exactly one TU
   per build, the linker resolves it unambiguously.
4. `vx_device_count` is similarly backend-private:
   `extern "C" vx_result_t vx_count_devices(uint32_t* out);` lives in
   the same TU as `vx_make_platform`.

**Why not runtime dispatch?**

- Legacy `vortex.h` already works this way; matching the convention
  avoids surprising existing users.
- Zero new dispatch machinery to write or test.
- Backend-specific link dependencies (libxrt, libsimx, etc.) stay
  scoped to the chosen backend — a runtime dispatch table would force
  every backend's dependencies onto every build.
- Upper-layer translators (POCL, chipStar, future Vulkan ICD) choose
  the active backend by picking which `libvortex.so` they link
  against. They don't see backend selection through the API.

The shared dynamic-loader helpers (e.g. `runtime/xrt/driver.{h,cpp}`
that `dlopen`s `libxrt.so` to resolve XRT symbols at runtime) are
reused across legacy `vortex.cpp` and new `vortex2_xrt.cpp` in the
same backend. They don't get duplicated.

### 3.2 Coexistence with legacy `vortex.cpp` during phases 1-7

During phases 1 through 7 (before the phase 8 shim collapses them
into one), both the legacy `vortex.cpp` and the new
`vortex2_<backend>.cpp` are linked into the same `libvortex.so` per
backend. They expose disjoint C-API symbol sets (`vx_dev_open` etc.
vs `vx_device_open` etc.), so there is no link-time collision.

Runtime coexistence rules:

- **Shared sub-helpers**: per-backend driver helpers
  (`runtime/xrt/driver.{h,cpp}`, OPAE's `runtime/opae/driver.{h,cpp}`
  when it returns) are shared between legacy and new code paths.
  `libxrt` is loaded once per process; the handle is held in a
  process-global, accessed by both `vortex.cpp` and
  `vortex2_xrt.cpp`.
- **No shared device state across APIs**: each API opens its own
  connection to the FPGA. The XRT AFU exposes two parallel control
  surfaces (legacy MMIO command FSM for `vortex.h`, CP doorbells for
  `vortex2.h`); the AFU's compatibility mode (parent §17) makes them
  mutually exclusive within a single process — legacy mode is engaged
  only when no `vortex2.h` queue is enabled.
- **Don't mix APIs against the same device in one process.** Use
  `vortex.h` *or* `vortex2.h`, not both. Mixing is not enforced at
  link time; the compat-mode check at the AFU prevents data corruption
  but the failure mode (`VX_ERR_DEVICE_BUSY` from `vx_device_open`
  when legacy AP_CTRL is active, and vice-versa) is a runtime surprise
  rather than a compile-time error.
- **Phase 8** collapses the duality: `vortex.cpp` is deleted; the
  legacy `vortex.h` entry points are re-implemented in
  `common/vortex2_legacy_shim.cpp` as wrappers around
  `vortex2.h`'s default queue (§8). After phase 8, the AFU's
  compatibility mode can be retired and both APIs share state by
  construction.

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

## 8. Legacy `vortex.h` shim (phase 8)

In phase 8 of the migration plan, every legacy backend's
`vortex.cpp` is deleted and replaced by a single
`common/vortex2_legacy_shim.cpp` that implements every `vx_*`
function from `vortex.h` over `vortex2.h` primitives. Mapping is in
§9 of the parent proposal; representative implementations:

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

Once phase 8 lands, the AFU's MMIO compatibility mode can be
retired (parent §9.3).

## 9. Stub backend

A `vortex2_stub.cpp` provides a minimal in-process mock for unit
tests. It implements `vx::Platform` over plain heap allocations and a
small in-process command "consumer" thread that mimics the CP:
fetches commands from the mock ring, completes them (memcpy for
copy/read/write, no-op for launch/DCR), and writes back completion
seqnums and profile timestamps.

This lets every test in `tests/runtime/` run without any FPGA, RTL
simulation, or SimX dependency. It also serves as a reference for
"what the CP is supposed to do" — the stub's consumer thread mirrors
the CPE FSM at a high level.

## 10. Build system integration

### 10.1 `configure` flags

```
--enable-cp                   default: yes  (build CP-aware code paths)
--backend={xrt,simx,rtlsim,stub}  default: xrt
--cp-num-queues=N             default: 4
--cp-ring-size-bytes=N        default: 65536
--cp-profile-default          default: off
```

These set the corresponding `VX_CP_*` macros (parent §10) and pick
which backend's `vortex2_*.cpp` is linked into `libvortex.so`.

### 10.2 `Makefile` changes

Add to `sw/runtime/common.mk`:

```makefile
VORTEX2_COMMON_SRCS := \
    common/vx_device.cpp \
    common/vx_queue.cpp \
    common/vx_buffer.cpp \
    common/vx_event.cpp \
    common/vx_command_encoder.cpp

ifeq ($(BACKEND),xrt)
  BACKEND_SRCS += xrt/vortex2_xrt.cpp xrt/vortex2_xrt_axi.cpp
endif
ifeq ($(BACKEND),simx)
  BACKEND_SRCS += simx/vortex2_simx.cpp
endif
ifeq ($(BACKEND),rtlsim)
  BACKEND_SRCS += rtlsim/vortex2_rtlsim.cpp
endif
ifeq ($(BACKEND),stub)
  BACKEND_SRCS += stub/vortex2_stub.cpp
endif

# Phase 8 only:
LEGACY_SHIM_SRCS := common/vortex2_legacy_shim.cpp
```

### 10.3 Conditional compilation

`#ifdef VX_CP_ENABLE` only guards code that allocates ring buffers or
talks to the CP MMIO surface. The header `vortex2.h` itself is
always installed (so out-of-tree builds can include it), but its
implementations may be stubs.

### 10.4 Out-of-tree builds

Per the project convention ([feedback-out-of-tree-builds]), all
build artifacts land under `build/`. `configure` (in the build dir)
copies the per-backend Makefiles into `build/sw/runtime/<backend>/`
and the build does not touch the source tree.

## 11. Test plan

### 11.1 Unit tests (`tests/runtime/`, new directory)

Run against the stub backend. Cover:

- Refcounting: `retain`/`release` on every handle class.
- Ring buffer wrap-around, backpressure, doorbell coalescing.
- Event signal/wait, including cross-queue wait, user events, host signaling.
- Profile timestamp readback, including cycle→ns conversion.
- Map/unmap on PIN_MEMORY buffers; `VX_ERR_NOT_SUPPORTED` on others.
- Concurrent enqueue from multiple host threads on the same queue.
- Concurrent enqueue from multiple queues on the same device.
- Legacy shim (phase 8): every `vx_*` function in `vortex.h`
  re-implemented over `vortex2.h` produces identical results to the
  pre-shim implementation.

Framework: existing `tests/Makefile` with a new `runtime/` subdir
built against `-lvortex_stub`. CI runs per [feedback-test-timeout-120s]
under a 120 s cap.

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

Aligns with parent proposal §13 migration plan.

### Phase 1 — `vortex2.h` skeleton (1 PR, ~1 week)

- [ ] Write `include/vortex2.h` exactly as §8.11 of parent.
- [ ] Write `common/vortex2_internal.h` with empty class declarations.
- [ ] Write `common/vx_device.cpp` with `vx_device_open` returning
      `VX_ERR_NOT_SUPPORTED` plus the refcount methods.
- [ ] Same skeleton for `vx_buffer.cpp`, `vx_queue.cpp`, `vx_event.cpp`.
- [ ] Write `vx_result_string`.
- [ ] Stub backends: `vortex2_xrt.cpp`, `vortex2_simx.cpp`,
      `vortex2_rtlsim.cpp`, `vortex2_stub.cpp`, all returning
      `VX_ERR_NOT_SUPPORTED` for everything.
- [ ] Build-system integration: configure flag, Makefile updates,
      `libvortex.so` exports the new symbols.
- [ ] Compile-only test: `gcc -include vortex2.h -shared empty.c` succeeds.

### Phase 2 — single-CPE runtime over CP (3-4 PRs, ~3 weeks)

Depends on RTL phase 2.

- [ ] Implement `Platform` interface for xrt (`vortex2_xrt_axi.cpp`).
- [ ] Implement `vx::Device::open` for xrt (queries device caps,
      reads `CP_CYCLE_FREQ_HZ`).
- [ ] Implement `vx::Buffer::create` using existing `MemoryAllocator`.
- [ ] Implement `vx::Queue::create` for single-CPE config (`NUM_QUEUES=1`):
      ring/head/cmpl allocation, MMIO writes to `Q_*` registers,
      `enqueue_mu_`, `tail_`.
- [ ] Implement `CommandEncoder` + `Queue::emit_command`.
- [ ] Implement `Queue::enqueue_write`, `enqueue_read`,
      `enqueue_launch` (no events yet — `out_event` ignored).
- [ ] Implement `Queue::flush` (write doorbell) and `Queue::finish`
      (poll completion slot for the last submitted seqnum).
- [ ] Integration test: vecadd on hardware.

### Phase 3 — multi-CPE + events (2-3 PRs, ~3 weeks)

Depends on RTL phase 3.

- [ ] `Device::alloc_queue_id` + per-queue id selection in
      `Queue::create`.
- [ ] `EventSlotPool` + `Event::bind` + `alloc_event`.
- [ ] Wire `out_event` parameter through every `enqueue_*`.
- [ ] `Event::status`, `Event::wait`, `vx::wait_all`,
      `vx_user_event_create` / `vx_user_event_signal`.
- [ ] Stress test: 4 queues each enqueueing 1k commands, all events
      wait_all'd at the end, no leaks under valgrind.

### Phase 4 — barriers, profiling, raw DCR, map/unmap (2-3 PRs, ~2 weeks)

Depends on RTL phase 4.

- [ ] Wait-list expansion in `Queue::emit_wait_list`.
- [ ] `Queue::enqueue_barrier`, `enqueue_dcr_write`, `enqueue_dcr_read`.
- [ ] `ProfileSlotPool`, `F_PROFILE` flag emission, profile slot
      writeback parsing, `Event::get_profile`.
- [ ] `Buffer::map` / `Buffer::unmap` with cache flush/invalidate.
- [ ] OpenCL 1.2 conformance smoke test passes through a POCL build
      backed by `vortex2.h`.

### Phase 5 — perf pass (1-2 PRs, timing-driven)

Doorbell coalescing, head-write batching, ring-buffer pinning
optimizations. Driven by phase-4 perf measurements.

### Phase 8 — legacy shim (1 PR, ~1 week)

- [ ] Implement `common/vortex2_legacy_shim.cpp` covering every
      `vortex.h` entry point per parent §9.1.
- [ ] Delete per-backend `vortex.cpp` files (xrt/simx/rtlsim/stub).
- [ ] Verify SimX/rtlsim/legacy tests pass unchanged.
- [ ] Update build system to link legacy shim by default.

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
