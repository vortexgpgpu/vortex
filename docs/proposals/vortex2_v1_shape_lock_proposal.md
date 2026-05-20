**Date:** 2026-05-18
**Status:** Draft — not yet started
**Author:** Blaise Tine
**Related:**
[pocl_on_vortex2_proposal.md](pocl_on_vortex2_proposal.md),
[command_processor_proposal.md](command_processor_proposal.md),
[cp_runtime_impl_proposal.md](cp_runtime_impl_proposal.md),
[chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md),
[hip_support_proposal.md](hip_support_proposal.md).

### Update history

- **2026-05-18** — Initial draft. Selected three changes from the
  vortex2 SOTA-alignment shortlist that **lock API shape** (vs.
  additive features). Pools, indirect dispatch, host functions,
  rect/fill, SVM, command buffers, etc. are all strictly-additive
  entry points and are deferred to v2 / on-demand.

---

# vortex2.h v1 SOTA — Shape-Lock Proposal

## 1. Summary

vortex2.h ships today as a clean async runtime built on the Command
Processor, but three pieces of its public shape diverge from the
convergent SOTA found across Vulkan, CUDA, HIP, and Metal. These three
are the only items where shipping v1 without them forces a painful
client migration later — every other modern-accelerator feature
(memory pools, indirect dispatch, host functions, command buffers,
SVM, …) is a strictly-additive new entry point that can land any time
without disturbing existing clients.

The three shape locks:

1. **Timeline events.** Replace today's binary `vx_event_h` with a
   counter-based primitive. Universal industry convergence: Vulkan
   `VkTimelineSemaphore` (deprecated binary fences), Metal
   `MTLSharedEvent.value`, CUDA `cuStreamWaitValue64`. Touches every
   client that uses events; retrofitting later requires a dual-type
   API forever (see Vulkan's binary-fence/timeline-semaphore mess).

2. **Module + kernel handles.** Introduce `vx_module_h` (a loaded
   .vxbin) and `vx_kernel_h` (a named entry point within a module).
   `vx_launch_info_t.kernel` becomes `vx_kernel_h` instead of
   `vx_buffer_h`. Universal: CUDA `cuModule`/`cuFunction`, HIP same,
   Metal `MTLLibrary`/`MTLFunction`, Vulkan `VkShaderModule` + entry
   name. Without this, every client invents the same switch-table
   hack POCL ships today.

3. **Raw-pointer kernel args (UVA).** `vx_launch_info_t.args` becomes
   a flat host pointer + size, not a `vx_buffer_h`. Buffers passed
   as args are `uint64_t` addresses in that blob. The runtime
   allocates and manages the device-side args buffer; clients never
   see it. Aligns with CUDA UVA, HIP, Vulkan
   `VK_KHR_buffer_device_address`, Metal `gpuAddress` — a single
   address space from the kernel's POV.

All three rewrite the shape of just two definitions: `vx_event_h`
semantics, and `vx_launch_info_t`. The rest of vortex2.h is unchanged.

**Not in this proposal** (additive, defer):

- Memory pools (`vx_pool_h`) — new entry points, no shape impact.
- Indirect dispatch (`vx_enqueue_launch_indirect`) — new entry point.
- Host functions (`vx_enqueue_host_func`) — new entry point.
- Rect / fill / async-map DMA — new entry points or info flags.
- SVM / managed memory — new flag bit on `vx_buffer_create`.
- Command buffers (`cl_khr_command_buffer`-class) — separate machinery.
- Per-arg API (`vx_kernel_set_arg`) — useful, but `vx_launch_info_t.args`
  as a flat blob covers v1; can layer per-arg on top later.

See [pocl_on_vortex2_proposal.md](pocl_on_vortex2_proposal.md) §9 for
the SVM / cmdbuf deferred designs.

---

## 2. Current state — what's wrong with v0 vortex2.h

Inventory of shape mismatches against SOTA (current API at
[sw/runtime/include/vortex2.h](../../sw/runtime/include/vortex2.h)):

| Today | SOTA convergence | Migration cost if deferred |
|-------|------------------|----------------------------|
| `vx_event_h` is binary (`status_e` enum: QUEUED/SUBMITTED/RUNNING/COMPLETE/ERROR) | Counter-based timeline event (uint64_t value, signal/wait at values) | Every user of `vx_enqueue_*`'s `out_event` and `vx_event_wait_all` |
| `vx_launch_info_t.kernel` is `vx_buffer_h` | `vx_kernel_h` (resolved at module-load time) | Every `vx_enqueue_launch` call site |
| `vx_launch_info_t.args` is `vx_buffer_h` (caller allocates + DMAs every launch) | Host pointer blob (runtime manages device-side args buffer) | Every `vx_enqueue_launch` call site + the legacy `vortex.h` wrapper |
| `vx_buffer_load_kernel_file` exposes kernel-image upload as a buffer op | Module-load returns a `vx_module_h`; entries resolved by name | One entry point in vortex2.h |

Costs **today** (already paid by POCL):

- Switch table in `__vx_get_kernel_callback` ([pocl-vortex.c kernel_main.c:40](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/kernel_main.c#L40))
  to fake `clCreateKernel(prog, "name")`.
- 60 lines of host-side arg marshalling ([pocl-vortex.c:584-648](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c#L584-L648)).
- Per-launch alloc + DMA of the kargs buffer; per-launch free.
- POCL's `pocl_vortex_driver_thread` re-implementing the wait
  loop on top of `vx_ready_wait` (would benefit from timeline
  events directly).

---

## 3. Feature 1 — Timeline events

### 3.1 Design

Replace today's binary event semantics with a Vulkan-style timeline.
Each `vx_event_h` carries a monotonically-increasing `uint64_t`
counter; signal advances it, wait blocks until counter ≥ value. The
binary case (today's behavior) is exactly "signal value=1 once,
wait value=1."

API:

```c
// Create — counter starts at 0.
vx_result_t vx_event_create        (vx_device_h dev, vx_event_h* out);

// Host-side signal (replaces vx_user_event_signal).
vx_result_t vx_event_signal        (vx_event_h ev, uint64_t value);

// Host-side queries.
vx_result_t vx_event_get_value     (vx_event_h ev, uint64_t* out);
vx_result_t vx_event_wait_value    (vx_event_h ev, uint64_t value,
                                    uint64_t timeout_ns);
vx_result_t vx_event_wait_values   (uint32_t n,
                                    const vx_event_h* evs,
                                    const uint64_t*   values,
                                    uint64_t timeout_ns);

// Queue-ordered signal / wait on an arbitrary event + value.
vx_result_t vx_enqueue_signal      (vx_queue_h q, vx_event_h ev,
                                    uint64_t value,
                                    uint32_t nw, const vx_event_h* w,
                                    vx_event_h* out_event);
vx_result_t vx_enqueue_wait_value  (vx_queue_h q, vx_event_h ev,
                                    uint64_t value,
                                    uint32_t nw, const vx_event_h* w,
                                    vx_event_h* out_event);

// Retain / release / profiling — unchanged shape.
vx_result_t vx_event_retain        (vx_event_h ev);
vx_result_t vx_event_release       (vx_event_h ev);
vx_result_t vx_event_get_profiling (vx_event_h ev, vx_profile_info_t* out);
```

**What changes for existing `vx_enqueue_*` calls:** nothing in
signature. The `out_event` returned by every enqueue is still a
single-event handle; it's signaled to **value = 1** when the work
completes. Existing code that does `vx_event_wait_all(1, &ev, …)`
keeps working — internally this is now `vx_event_wait_values(1, &ev,
&one, …)` where `one = 1`.

**What disappears:**

- `vx_event_status` and the `vx_event_status_e` enum — replaced by
  `vx_event_get_value` (any non-zero value means "completed at least
  once"). The QUEUED/SUBMITTED/RUNNING distinctions weren't surfaced
  by the CP path anyway.
- `vx_user_event_create` / `vx_user_event_signal` — replaced by the
  unified `vx_event_create` + `vx_event_signal` (any event can be
  host-signaled; "user event" was a vestigial OpenCL distinction).
- `vx_event_wait_all(n, evs, timeout)` — replaced by
  `vx_event_wait_values`. Migration is a single-line change for the
  legacy `vortex.h` wrapper.

### 3.2 Implementation

Each `vx_event_h` is a host-side reference-counted struct:

```c
struct vx_event {
    std::atomic<uint64_t> counter;
    std::atomic<uint32_t> refcount;
    std::mutex            mu;          // protects waiters
    std::condition_variable cv;
    vx_device*            owner;
    // Optional CP-backed counter slot (Phase 1c, see §3.3) — null in v1a.
    uint64_t              dev_addr;
    bool                  error;
};
```

- `vx_event_signal(ev, value)` atomically updates `counter = max(counter, value)`,
  then `cv.notify_all()`. O(1) plus waiter wakeup.
- `vx_event_wait_value` blocks on the condvar until `counter >= value`
  or timeout. Single-event fast path.
- `vx_event_wait_values` does the same but holds a small array of
  refs and loops; sufficient performance for the few-events case
  every OpenCL client hits.

The per-queue worker thread (existing infrastructure in
[sw/runtime/common/vx_queue.cpp](../../sw/runtime/common/vx_queue.cpp))
calls `signal(ev, 1)` after each command's work-lambda returns. No
change to the queue worker's lifecycle. Wait events at the queue
worker decode-time become a synchronous `wait_value` before
proceeding — same as today's wait-list handling.

### 3.3 CP-backed event slots (Phase 1c — optional, perf only)

In v1 (Phases 1a + 1b) events live in host RAM and the CP only signals
its own retire seqnum. This means queue-to-queue waits go through the
host (worker A signals event X, worker B's wait wakes up via condvar,
worker B proceeds). For most workloads this is fine — POCL today
already round-trips through the host on every command.

In a follow-up phase (1c), event counters can additionally live in a
device-pinned `uint64_t` slot the CP can read/write directly:

- `CMD_EVENT_SIG` (opcode 0x08, already in [cmd_processor.h:122](../../sim/common/cmd_processor.h#L122))
  writes a value to a device-side counter slot.
- `CMD_EVENT_WAIT` (opcode 0x09) spins the CP engine until the slot ≥ value.

Then GPU-only chains (kernel A on queue 1 produces event N, kernel B
on queue 2 waits for N, all without host round-trip) become possible.
Pure perf optimization, no API change — the host-RAM path remains the
fallback when the CP doesn't have the slot allocated.

The functional CP model already has placeholder enum slots
([cmd_processor.h:122-123](../../sim/common/cmd_processor.h#L122-L123))
but the `tick_engine` `Bid` state treats them as no-resource NOPs
([cmd_processor.cpp:207-211](../../sim/common/cmd_processor.cpp#L207-L211)).
1c wires them up.

---

## 4. Feature 2 — Module + kernel handles

### 4.1 Design

A `vx_module_h` is a loaded `.vxbin` (device memory + parsed symbol
table). A `vx_kernel_h` is a named entry point within a module, with
its PC and (optionally) arg-layout metadata cached.

```c
typedef struct vx_module* vx_module_h;
typedef struct vx_kernel* vx_kernel_h;

vx_result_t vx_module_load_file   (vx_device_h dev, const char* path,
                                   vx_module_h* out);
vx_result_t vx_module_load_bytes  (vx_device_h dev, const void* bytes,
                                   size_t size, vx_module_h* out);
vx_result_t vx_module_retain      (vx_module_h mod);
vx_result_t vx_module_release     (vx_module_h mod);

vx_result_t vx_module_get_kernel  (vx_module_h mod, const char* name,
                                   vx_kernel_h* out);
vx_result_t vx_kernel_retain      (vx_kernel_h k);
vx_result_t vx_kernel_release     (vx_kernel_h k);

// Useful kernel-introspection queries (optional v1, can grow later):
vx_result_t vx_kernel_get_max_block_size (vx_kernel_h k,
                                          uint32_t* x, uint32_t* y, uint32_t* z);
```

**What changes in `vx_launch_info_t`:**

```c
typedef struct {
    size_t       struct_size;
    const void*  next;
    vx_kernel_h  kernel;           // was vx_buffer_h
    const void*  args_host;        // was vx_buffer_h args — see Feature 3
    size_t       args_size;        // was implicit in vx_buffer_h size
    uint32_t     ndim;
    uint32_t     grid_dim [3];
    uint32_t     block_dim[3];
    uint32_t     lmem_size;
} vx_launch_info_t;
```

**What disappears:**

- `vx_buffer_load_kernel_file` ([vortex2.h:159-160](../../sw/runtime/include/vortex2.h#L159-L160)).
  Replaced by `vx_module_load_file`.
- The "kernel binary as a buffer" pun.
- POCL's `__vx_get_kernel_callback` switch table + `kernel_id` field
  in its args struct ([pocl-vortex.c kernel_args.h](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/kernel_args.h)).

### 4.2 Implementation

`vx_module` internals:

```c
struct vx_module {
    std::atomic<uint32_t> refcount;
    vx_device*            owner;
    vx_buffer_h           dev_image;       // the .vxbin in device memory
    uint64_t              base_addr;       // dev_image's device address
    // Symbol table parsed from the vxbin header.
    struct Symbol {
        std::string name;
        uint64_t    pc;        // absolute device address of entry
    };
    std::vector<Symbol>                  symbols;
    std::map<std::string, vx_kernel_h>   kernel_cache;
};

struct vx_kernel {
    std::atomic<uint32_t> refcount;
    vx_module*            owner;
    uint64_t              pc;
    // Future: arg-layout metadata, max-block hints, etc.
};
```

`vx_module_load_file`:
1. Read file bytes.
2. Allocate device buffer via existing `vx_buffer_create` / `vx_buffer_reserve`
   (preserves the .vxbin's min_vma/max_vma reserve semantics today
   handled in [legacy_utils.cpp:25-72](../../sw/runtime/common/legacy_utils.cpp#L25-L72)).
3. DMA bytes via `vx_enqueue_write` on the device's internal helper queue;
   block until done.
4. Parse the vxbin's symbol table. (Today's `.vxbin` has only an entry
   point at `min_vma + 0` — multi-symbol support is part of this
   feature's work; see §4.3.)
5. Return a refcounted `vx_module_h`.

`vx_module_get_kernel`:
- Linear scan of `symbols` (typically ≤32 kernels per module).
- Cache the resulting `vx_kernel_h` so repeated lookups for the same
  name don't allocate.

`vx_enqueue_launch` (post-rewrite):
1. Extract `vx_kernel_h k` from `info->kernel`.
2. Use `k->pc` to program `VX_DCR_KMU_STARTUP_ADDR0/1` via the CP
   ring (replaces the current code at
   [vx_queue.cpp:291-303](../../sw/runtime/common/vx_queue.cpp#L291-L303)
   that reads PC from `kernel->dev_address()`).
3. The args portion is Feature 3 below.

### 4.3 .vxbin symbol table

Today's `.vxbin` is just `[min_vma : 8][max_vma : 8][raw bytes]`. To
support `vx_module_get_kernel(mod, "name")`, we need a symbol table.

Minimal additive change to the vxbin format (the `vxbin.py` tool at
[sw/kernel/scripts/vxbin.py](../../sw/kernel/scripts/vxbin.py)):

- Append a footer: `[magic:8]['VXSYMTAB':8][n_symbols:4][entries...]`,
  where each entry is `[name_offset:4][pc:8][name_len:2][reserved:2]`
  followed by a contiguous string blob.
- Footer position is `file_size - sizeof(footer)`.
- Loader checks for the magic; if absent, falls back to single-entry
  mode where the sole kernel name is `"main"` and its PC is `min_vma`
  (preserves current `.vxbin` files).

The Clang toolchain side ([vx_start.S](../../sw/kernel/src/vx_start.S))
already exports `main` as the entry. Multi-kernel support requires
the toolchain to emit multiple `__attribute__((annotate("vortex.kernel")))`
symbols, which it can already do (POCL uses this annotation).

### 4.4 Backwards compatibility

The legacy `vortex.h` wrapper functions (`vx_upload_kernel_file`,
`vx_upload_kernel_bytes`, `vx_start`, `vx_start_g`) keep their public
signatures but call `vx_module_load_file` internally and use the
module's default ("main") kernel:

```c
extern "C" int vx_upload_kernel_file(vx_device_h h, const char* path,
                                     vx_buffer_h* out_legacy) {
    // Load as module, return the legacy buffer handle pointing at the
    // module's underlying image (refcounted).
    vx_module_h mod;
    if (vx_module_load_file(h, path, &mod) != VX_SUCCESS) return -1;
    *out_legacy = vx_module_legacy_buffer_view(mod);   // hidden helper
    return 0;
}
```

`vx_start` / `vx_start_g` resolve the legacy "kernel buffer" back to
its owning module via a runtime-side reverse-lookup table and call
`vx_module_get_kernel(mod, "main")`. This keeps every test in
`tests/regression/` working without changes.

---

## 5. Feature 3 — Raw-pointer kernel args (UVA)

### 5.1 Design

Today, the caller allocates a device buffer for the args block, DMAs
it, passes the `vx_buffer_h`, and frees it after the launch retires.
POCL pays this cost on every kernel invocation
([pocl-vortex.c:558-651](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c#L558-L651)).

Replace with:

```c
typedef struct {
    size_t       struct_size;
    const void*  next;
    vx_kernel_h  kernel;
    const void*  args_host;        // host blob — runtime DMAs to device
    size_t       args_size;
    uint32_t     ndim;
    uint32_t     grid_dim [3];
    uint32_t     block_dim[3];
    uint32_t     lmem_size;
} vx_launch_info_t;
```

The caller hands the runtime a host-side blob. Buffers passed as args
appear as their `uint64_t` device addresses (already obtainable via
`vx_buffer_address`) inline in the blob — no `vx_buffer_h`
indirection at the launch interface. The kernel reads its args via
MSCRATCH exactly as today; nothing changes device-side.

### 5.2 Implementation

The runtime owns a per-device scratch pool for args buffers (~64 KiB
ring; sized for typical kernel-arg blocks of 64–512 bytes). On
`vx_enqueue_launch`:

1. Round-robin allocate `args_size` bytes from the scratch pool.
2. DMA `args_host[0..args_size]` to the slot via the existing CP
   `CMD_MEM_WRITE` opcode.
3. Program `VX_DCR_KMU_STARTUP_ARG0/1` with the slot address (replaces
   [vx_queue.cpp:292,302-303](../../sw/runtime/common/vx_queue.cpp#L292)).
4. After the launch retires, return the slot to the pool's free list
   in the per-queue worker's completion callback.

The scratch pool is bounded; if exhausted (long-running async launches
with large arg blocks), the runtime grows it (doubles size, up to a
cap). This is a v0 implementation that the v2 memory-pool feature
([pocl_on_vortex2_proposal.md](pocl_on_vortex2_proposal.md) §9 future
work) generalizes.

### 5.3 Buffer pointers in args

Callers building the args blob marshal `vx_buffer_h` to its address:

```c
vx_buffer_h buf;
vx_buffer_create(dev, 4096, VX_MEM_READ_WRITE, &buf);
uint64_t buf_addr;
vx_buffer_address(buf, &buf_addr);

struct { uint64_t input; uint64_t output; uint32_t n; } args = {
    buf_addr, /* ... */, 1024
};

vx_launch_info_t li = {
    .struct_size = sizeof(li),
    .kernel      = my_kernel,
    .args_host   = &args,
    .args_size   = sizeof(args),
    .ndim        = 1,
    .grid_dim    = {64, 1, 1},
    .block_dim   = {16, 1, 1},
};
vx_enqueue_launch(q, &li, 0, NULL, &ev);
```

This matches what every modern accelerator client (CUDA, HIP, Metal,
Vulkan with BDA) does today — and what POCL already does internally
before sticking the result into a `vx_buffer_h`.

### 5.4 What disappears

- The convention that `vx_launch_info_t.args` is a `vx_buffer_h` the
  caller allocates and frees.
- Per-launch caller-side allocate / DMA / free pattern.
- The need for clients to call `vx_mem_alloc(kargs_buffer_size,
  VX_MEM_READ, …)` + `vx_copy_to_dev(…)` + `vx_mem_free(…)` around
  every launch.

---

## 6. Migration impact

### 6.1 Public API surface

```
deleted     vx_user_event_create, vx_user_event_signal,
            vx_event_status, vx_event_wait_all,
            vx_event_status_e enum,
            vx_buffer_load_kernel_file
added       vx_event_create, vx_event_signal, vx_event_get_value,
            vx_event_wait_value, vx_event_wait_values,
            vx_enqueue_signal, vx_enqueue_wait_value,
            vx_module_load_file, vx_module_load_bytes,
            vx_module_retain, vx_module_release,
            vx_module_get_kernel, vx_kernel_retain, vx_kernel_release,
            vx_kernel_get_max_block_size
modified    vx_launch_info_t (3 fields changed)
unchanged   vx_device_* (all), vx_buffer_* (except load_kernel_file),
            vx_queue_* (all), vx_enqueue_{read,write,copy,barrier,
            dcr_read,dcr_write}, vx_event_{retain,release,
            get_profiling}
```

Net change: ~6 entry points removed, ~16 added, 1 struct rewritten.
The "modified" struct (`vx_launch_info_t`) carries `struct_size` and
`next` already, so callers passing the new struct to runtimes built
against the old definition would fail with `VX_ERR_INVALID_INFO` —
clean breakage, not silent corruption.

### 6.2 Legacy `vortex.h` wrapper

The `vortex.h` shim is the *only* place inside this repo that has to
absorb the breakage. Touchpoints:

- [legacy_runtime.cpp](../../sw/runtime/common/legacy_runtime.cpp):
  `vx_start_g`, `vx_dcr_*`, `vx_copy_*`, all the `enqueue_and_wait`
  call sites. The launch-info build switches from `vx_buffer_h kernel,
  args` to `vx_kernel_h kernel, void* args_host, size_t args_size`.
- [legacy_utils.cpp](../../sw/runtime/common/legacy_utils.cpp):
  `vx_upload_kernel_file`, `vx_upload_bytes` rewritten as thin
  wrappers around `vx_module_load_*`.
- Internal `legacy_default_queue` / `legacy_remember_last_event` —
  events are timeline now; `wait_value(ev, 1)` replaces the
  previous wait.

This is bounded — ~50 lines of edits in two files.

### 6.3 POCL backend

POCL's vortex backend ([pocl_on_vortex2_proposal.md](pocl_on_vortex2_proposal.md))
moves directly onto the new shape:

- `vx_dev_caps → vx_device_query` (unchanged semantically)
- `vx_upload_kernel_file → vx_module_load_file`
- POCL's `__vx_get_kernel_callback` switch table **deleted**; each
  `cl_kernel` holds a `vx_kernel_h`.
- The 60-line `kernel_args_t` marshalling **deleted**; POCL builds
  the host args blob inline and hands it to `vx_enqueue_launch`.
- `pocl_vortex_driver_thread` deletes the sync-execute path; remains
  only as the event-completion notifier (blocking on
  `vx_event_wait_value`).

POCL migration is **independent of and after** the runtime work in
this proposal — Phase 4 in §7.

### 6.4 Other clients

- chipStar / HIP: not landed yet; will write directly to the v1 API.
- Direct `tests/runtime/test_*.cpp`: ~5 files using the v0 API; one-touch
  update each.
- `tests/regression/`: uses only the legacy `vortex.h` wrapper —
  unchanged once the wrapper is migrated.

---

## 7. Phases

### Phase 1a — Timeline event primitive ✅ no public-API removal yet

Add the new event API entry points (§3.1). Internally rewrite
`vx_event` from binary to counter (atomic + condvar). Keep
`vx_user_event_create`, `vx_user_event_signal`, `vx_event_status`,
`vx_event_wait_all` as **deprecated shims** that forward to the new
API. Runtime ships dual-API.

**Definition of done:**
- New event tests in `tests/runtime/test_timeline_events.cpp` pass on
  simx (functional baseline) and xrt (RTL coverage via the full AFU
  surface). opae-sim added where parallel coverage matters.
- All existing `tests/regression/` apps pass through the deprecated
  shims (no behavior change).
- Wall-time delta on `tests/regression run-simx`: within ±5%.

### Phase 1b — Module + kernel handles ✅ vxbin format extended

Add `vx_module_*` and `vx_kernel_*` (§4). Extend `.vxbin` to carry a
symbol table footer (§4.3); update [vxbin.py](../../sw/kernel/scripts/vxbin.py).
`vx_buffer_load_kernel_file` becomes a deprecated shim that loads as
a module and returns a synthetic buffer handle.

`vx_launch_info_t.kernel` accepts either `vx_kernel_h` (new) or
`vx_buffer_h` (legacy) — disambiguated by a hidden type tag in the
handle's first word, exactly as Vulkan disambiguates `VkObject`
handles. Cleaner than struct versioning here because the field stays
opaque to callers.

**Definition of done:**
- New tests `tests/runtime/test_module_kernel.cpp` (load module, get
  two kernels by name, launch each) pass on simx and xrt.
- POCL build target updated to call new API (one-line change to
  `pocl-vortex.c`'s `vx_upload_kernel_file` site).
- All existing tests pass through legacy shim.

### Phase 1c — CP-backed event slots — perf only, optional

Wire `CMD_EVENT_SIG` / `CMD_EVENT_WAIT` opcodes (§3.3) into the CP and
its functional model. Backend-side change in
[sim/common/cmd_processor.cpp](../../sim/common/cmd_processor.cpp)
plus the RTL CP. Public API unchanged.

**Definition of done:**
- Multi-queue ping-pong test (queue A signals event N, queue B waits
  on N) no longer routes through host condvar; CP-side counter slot
  carries the value. Verified on xrt (full CP RTL path).
- Microbenchmark: cross-queue dependency latency improves by ≥30% on
  simx (functional baseline) and reduces host MMIO traffic measurably
  on xrt (counter-slot polling moves device-side).

### Phase 2 — Launch info reshape

Change `vx_launch_info_t.args` from `vx_buffer_h` to `const void*
args_host, size_t args_size` (Feature 3). Bump
`struct_size`-recognized version. Old callers passing the old struct
get `VX_ERR_INVALID_INFO` — clean breakage.

Implement the per-device scratch pool for args (§5.2).

**Definition of done:**
- Runtime test passes a flat-blob args struct including a buffer
  pointer; kernel reads via MSCRATCH and dereferences correctly.
  Verified on simx and xrt.
- Legacy `vortex.h` `vx_start_g` rewritten to build the host blob
  inline rather than allocating a device buffer.
- `tests/regression run-simx` passes; representative regression
  apps (`arith`, `vecadd`, `sgemm`) pass via `run-xrt`.
- Wall-time on regression suite: ≥10% improvement attributable to
  removed per-launch alloc/free pairs.

### Phase 3 — Remove deprecated shims

Once a transition window has elapsed (Phase 1a/b shims have been live
for one tag), delete:
- `vx_user_event_create`, `vx_user_event_signal`, `vx_event_status`,
  `vx_event_wait_all` (Phase 1a shims)
- `vx_buffer_load_kernel_file` (Phase 1b shim)
- `vx_event_status_e` enum
- buffer-handle disambiguation in `vx_launch_info_t.kernel`

**Definition of done:**
- `grep -r "vx_user_event\|vx_event_wait_all\|vx_buffer_load_kernel" .`
  returns only the proposal/changelog.

### Phase 4 — POCL migration

Implements [pocl_on_vortex2_proposal.md](pocl_on_vortex2_proposal.md)
Phases 1-3 on top of this proposal's API. Independent track once
Phase 2 lands.

---

## 8. Risks & open questions

**R1. Timeline-event waiter wakeup storm.** A single `vx_event_signal`
to a value that satisfies many waiters does an `O(n)` condvar
broadcast. For Vortex's typical workload (a few events, ≤32 waiters)
this is irrelevant. Mitigation: if it ever becomes a problem, swap
the condvar for a per-target-value waiter list.

**R2. .vxbin symbol-table compatibility.** Existing `.vxbin` files in
test trees won't have the footer. §4.3's fallback ("treat as
single-kernel `main`") covers them, but the Clang/LLVM toolchain
update to emit multi-symbol footers is a coordinated change with the
[llvm_vortex_v3_proposal.md](llvm_vortex_v3_proposal.md) work.
Schedule Phase 1b after that.

**R3. Buffer-pointer marshalling in args.** Clients constructing the
host args blob need to know the device-pointer width (32-bit on
rv32, 64-bit on rv64). The current API offers no helper. Either
require callers to query `VX_CAPS_ISA_FLAGS` and conditionalize, or
add a `vx_kernel_get_pointer_size(kernel, &bytes)` helper. Leaning
toward the helper — cheap and self-documenting.

**R4. Scratch pool sizing under high concurrency.** Phase 2's
per-device scratch pool defaults to 64 KiB. A workload with 1000
in-flight launches × 256-byte args = 256 KiB, exceeding the pool.
Growth strategy (§5.2: double on exhaustion) handles this, but the
allocator path becomes a latency outlier on the first exhaustion. Pre-
sizing via cap query (`VX_CAPS_MAX_INFLIGHT_LAUNCHES`) is an option
if real workloads hit it.

**R5. Phase 3 timing.** Removing deprecated shims is the painful part
for external clients (HIP, chipStar) that may have written against the
shim API. The "one tag" transition window is a placeholder — actual
duration depends on which external clients exist by then.

---

## 9. Out-of-scope

Deferred to follow-up proposals or on-demand:

- **Memory pools** (`vx_pool_h`). Strictly additive; add when a
  client hits the allocator hot path.
- **Per-arg setter API** (`vx_kernel_set_arg`). Convenience over the
  flat-blob args; adds CUDA-style ergonomics without disturbing v1.
- **Indirect dispatch**, **host functions**, **rect/fill/async map**.
  All additive new entry points, designed in
  [pocl_on_vortex2_proposal.md](pocl_on_vortex2_proposal.md) §5/§9
  but not blocking v1.
- **Managed / SVM memory.** Additive flag bit + cap query; depends on
  per-backend HW capability.
- **Command buffers** (`cl_khr_command_buffer`-class graph replay).
  Separate machinery; designed in
  [pocl_on_vortex2_proposal.md](pocl_on_vortex2_proposal.md) §9.2.
- **Multi-device contexts**, **multi-HW-queue mapping**, **external
  memory interop**. Separate work each.

The principle behind every deferral: if it's a *new entry point* that
doesn't change `vx_event_h`'s semantic shape, `vx_launch_info_t`'s
fields, or the kernel-handle identity, it's strictly additive and can
land any time without forcing client migration. This proposal closes
v1 SOTA by changing exactly the things that can't be retrofitted
cleanly.
