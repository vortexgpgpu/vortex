**Date:** 2026-05-18
**Status:** Draft — not yet started
**Author:** Blaise Tine
**Related:**
[pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md),
[chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md),
[cp_runtime_impl_proposal.md](cp_runtime_impl_proposal.md),
[command_processor_proposal.md](command_processor_proposal.md).

### Update history

- **2026-05-18** — Initial draft. Companion to
  [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md): that proposal
  targets the **legacy synchronous `vortex.h`** wrapper (just plumbing
  KMU through `vx_start_g`). This proposal targets the **async
  `vortex2.h`** runtime directly and treats the legacy wrapper as a
  dead-end for POCL. The two are not mutually exclusive — phases 0–2
  of the v3 proposal land first and ship POCL on legacy `vortex.h` /
  KMU; this proposal then redesigns on top.

---

# POCL on vortex2.h — Redesign Proposal

## 1. Summary

The current `pocl-vortex.c` ([lib/CL/devices/vortex/pocl-vortex.c](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c))
drives Vortex through the **synchronous legacy `vortex.h` API**:
`vx_dev_open` → `vx_mem_alloc` → `vx_copy_to_dev` → `vx_start_g` →
`vx_ready_wait`. Every OpenCL command serializes through a single
POCL-side driver thread that blocks on the device for each step.

`vortex2.h` ([sw/runtime/include/vortex2.h](../../sw/runtime/include/vortex2.h)) —
the canonical async runtime built on the Command Processor — exposes the
Vulkan/CUDA/Metal-style queue/buffer/event model that POCL itself uses
internally. Targeting it directly lets us:

1. **Delete POCL's driver thread** — vortex2 already runs a per-queue
   worker; one redundant layer is removed.
2. **Pipeline H2D/D2H around launches** — async DMAs unblock concurrent
   `clEnqueueWriteBuffer` + `clEnqueueNDRangeKernel`, which today serialize.
3. **Bridge `cl_event` ↔ `vx_event_h` directly** — POCL's event
   completion fires from the vortex2 event itself, not from a polling
   wait. Eliminates the `vx_ready_wait` hot path.
4. **Cache kernel uploads** — drop the "free + re-upload on every
   launch" pattern at [pocl-vortex.c:660-680](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c#L660-L680);
   the CP DMA cost of re-uploading megabytes of kernel image per launch is
   the single largest avoidable cost in the current backend.

But — **vortex2.h has gaps** that block a clean POCL mapping. §3 of
this document enumerates them, prioritized for **OpenCL 1.2
conformance** (post-1.2 surface — SVM, `cl_khr_command_buffer`,
multi-HW-queue — is captured separately in §9). §5 proposes the
additions to `vortex2.h` needed for 1.2. Some are pure missing-API;
one (kernel modules with N entry points) is a refactor; one (rect /
fill DMA) is new feature surface; one (event callbacks) is the
difference between "POCL spawns a polling thread" and "POCL piggybacks
the CP's notification path."

The end state: ~250 lines of POCL device-ops over a `vortex2.h` that
has grown by ~20 entry points. No legacy `vx_*` calls left in the
POCL backend; legacy `vortex.h` becomes a non-POCL compat shim only.

---

## 2. Current state — what POCL touches in vortex.h today

Inventory taken from [pocl-vortex.c](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c):

| POCL device-op | vortex.h calls | Synchronous? |
|----------------|----------------|--------------|
| `init` | `vx_dev_open`, `vx_dev_caps × 5` | yes |
| `uninit` | `vx_mem_free`, `vx_dev_close` | yes |
| `alloc_mem_obj` | `vx_mem_alloc`, `vx_mem_address`, `vx_copy_to_dev` | yes |
| `free` | `vx_mem_free` | yes |
| `read` | `vx_copy_from_dev` | **yes — blocks the driver thread** |
| `write` | `vx_copy_to_dev` | **yes** |
| `copy` | `vx_copy_dev_to_dev` | **yes** |
| `run` (per launch) | `vx_check_occupancy`, `vx_mem_alloc` (kargs), `vx_mem_address`, `vx_copy_to_dev` (kargs), `vx_dump_perf`, `vx_mem_free` (prev kernel), `vx_upload_kernel_file`, `vx_start_g`, `vx_ready_wait`, `vx_mem_free` (kargs) | **all yes** |

Two structural problems jump out:

**P1. Every command blocks the driver thread.** POCL's
`pocl_vortex_driver_thread` ([pocl-vortex.c:854-880](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c#L854-L880))
drains a ready-list serially via `pocl_exec_command`, and every leaf
`vx_*` call is synchronous. A `clEnqueueWriteBuffer` followed by an
unrelated `clEnqueueNDRangeKernel` on the **same** queue waits for the
write's DMA to fully retire on the host before the launch can even
start argument marshalling. The CP can serve both back-to-back from
its ring with no host idle time — the layering above it just doesn't
let it.

**P2. Kernel binary re-uploaded per launch.** [pocl-vortex.c:660-680](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c#L660-L680):
```c
if (dd->vx_kernel_buffer != NULL) {
  vx_dump_perf(dd->vx_device, stdout);
  vx_mem_free(dd->vx_kernel_buffer);  // free previous
}
vx_err = vx_upload_kernel_file(dd->vx_device, sz_program_vxbin,
                                &dd->vx_kernel_buffer);  // re-DMA
```

The kernel image is freed and re-uploaded on **every** `pocl_run`,
even when the same `cl_kernel` is launched in a loop. For a 1 MB
`.vxbin` and a per-step kernel, that's 1 MB of host→device DMA per
iteration. The cache key is "same program object" — POCL has the
information to avoid the re-upload; the current code throws it away.

There is also a correctness smell: the second `cl_kernel` launch on
the same `cl_command_queue` frees `dd->vx_kernel_buffer` from launch N
**before** waiting for the in-flight `vx_start_g` of launch N to
retire. Today this is safe only because `vx_ready_wait` is called at
the end of every `pocl_run` and the driver thread is single — so
there is no overlap. The moment we want overlap, this becomes a UAF.

---

## 3. vortex2.h shortcomings for POCL

These are the gaps that block a 1-to-1 redesign onto `vortex2.h` as
shipped today. Ordered by **OpenCL 1.2 priority** — items §3.1–§3.5
are required for 1.2 conformance, §3.6–§3.7 are perf wins, and the
post-1.2 items (SVM, command-buffer extension, multi-HW-queue) are
deferred to §9. Numbered so §5 (API additions) and §6 (phases) can
refer back.

### Quick map: which OpenCL 1.2 surface each gap blocks

| Gap | Blocks 1.2 API |
|-----|----------------|
| §3.1 event callbacks | `clSetEventCallback`, every async `clEnqueue*` returning an event |
| §3.2 module / multi-kernel | `clCreateKernel(program, "name")`, `clCreateKernelsInProgram` |
| §3.3 kernel arg API | `clSetKernelArg` (semantically; POCL works around today) |
| §3.4 async map/unmap | `clEnqueueMapBuffer`, `clEnqueueUnmapMemObject` |
| §3.5 rect + fill DMA | `clEnqueueRead/WriteBufferRect`, `clEnqueueCopyBufferRect`, `clEnqueueFillBuffer` |
| §3.6 kernel image cache | (perf only — `cl_program` re-binding loop) |
| §3.7 perf / DCR helpers | (perf / introspection only — `clGetEventProfilingInfo`-shaped) |

### 3.1 No event callbacks (`vx_event_set_callback`) — 1.2 required

OpenCL 1.2 mandates `clSetEventCallback` for command-status
notifications, and every async `clEnqueue*` returns a `cl_event` whose
state transitions POCL must observe. With sync `vortex.h`, POCL fires
the notification itself after `vx_ready_wait` returns — trivial.
With async `vortex2.h`, POCL needs to be told when a `vx_event_h`
transitions to `COMPLETE` so it can run
`pocl_update_event_completed(cl_event)`.

Current vortex2 surface: `vx_event_status` (poll-only) and
`vx_event_wait_all` (blocking). To wire OpenCL semantics, POCL would
have to maintain a polling thread per device that wakes every N ms,
calls `vx_event_status` on every outstanding event, and dispatches
callbacks for state transitions. That's an extra context-switch
per launch retirement and adds visible latency to short kernels.

**The CP already publishes seqnum to a host-pinned completion slot**
on retire ([cp_runtime_impl_proposal.md](cp_runtime_impl_proposal.md)
§"completion ring"). The dispatcher's per-queue worker polls that
slot today. Surfacing a callback into user code is one extra function
pointer call from a worker that already exists — no new thread.

### 3.2 No kernel module / multi-entry-point abstraction — 1.2 required

OpenCL 1.2: one `cl_program` → many `cl_kernel`, retrieved by name
via `clCreateKernel(program, "name")`. The Vortex backend fakes this
today via a switch table in `__vx_get_kernel_callback`
([pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md) §3.2,
[kernel_main.c:40](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/kernel_main.c#L40))
keyed by an integer `kernel_id` packed into the args buffer. The
runtime's `vx_buffer_load_kernel_file` loads the whole image as a
single opaque buffer and never sees individual entry points.

Cost: every kernel launch costs an extra indirect call inside
`kernel_main`; the args buffer carries a `kernel_id` field that
the runtime should know already; cross-kernel data sharing (a single
program's `__constant` globals across multiple kernels) has no
runtime-blessed expression.

What's missing: `vx_module_h` (loaded .vxbin, opaque) and `vx_kernel_h`
(symbol within a module, with PC + arg layout metadata).
`vx_module_load_file`, `vx_module_get_kernel(module, name)`,
`vx_module_release`. The launch then takes a `vx_kernel_h` instead
of a `vx_buffer_h`, the runtime knows the PC, and the user
`kernel_id` switch goes away.

### 3.3 Kernel arg block has no API — POCL builds it raw — 1.2 required (semantic)

The kernel-arg buffer ([pocl-vortex.c:584-648](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c#L584-L648))
is built by POCL: scalar/pointer/local args, alignment rules,
local-memory packing — 60 lines of pointer arithmetic that any
runtime should provide. CUDA: `cuLaunchKernel(... void** kernelParams)`.
HIP: similar. The runtime knows the kernel's arg layout (from
metadata or compiler output) and assembles the on-wire descriptor.

Strictly speaking POCL works around this today (it knows OpenCL's arg
rules), so this is not a *conformance* blocker — but it lives in the
1.2 group because it's the natural API surface of §3.2's `vx_kernel_h`,
and without it the §3.2 work is half-done.

With `vx_kernel_h` (§3.2) carrying arg metadata, vortex2.h can offer
`vx_kernel_set_arg(kernel, idx, size, ptr)` /
`vx_kernel_set_local_arg(kernel, idx, size)`, and the launch enqueue
takes a configured `vx_kernel_h` rather than a flat args buffer.

This is also the natural place to surface `VX_CAPS_NUM_THREADS`
auto-occupancy: `vx_kernel_get_max_block_size(kernel, &x, &y, &z)`.
Today POCL calls the legacy `vx_check_occupancy` helper for this.

### 3.4 No `vx_buffer_map_async` (synchronous map only) — 1.2 required

`vx_buffer_map`/`vx_buffer_unmap` exist
([vortex2.h:167-169](../../sw/runtime/include/vortex2.h#L167-L169))
but are synchronous: no event in, no event out. OpenCL 1.2's
`clEnqueueMapBuffer` / `clEnqueueUnmapMemObject` are async — they
must complete in command-queue order, after any in-flight writes to
the buffer, and they take a wait list + return an event.

What's missing: `vx_enqueue_map(queue, buf, off, size, flags,
n_wait, wait_events, out_event, out_host_ptr)` and
`vx_enqueue_unmap(queue, buf, host_ptr, n_wait, wait_events,
out_event)`. Same shape as the other enqueue ops.

### 3.5 No rect / fill DMA primitives — 1.2 required

OpenCL 1.2: `clEnqueueRead/WriteBufferRect`, `clEnqueueCopyBufferRect`
(carried over from 1.1), and `clEnqueueFillBuffer` (new in 1.2). POCL
drivers like `cuda` and `level0` lower each of these to one device
call. vortex2.h has only linear `vx_enqueue_read/write/copy`. POCL
would have to slice each rect into N linear enqueues, each carrying
its own event — a 64×64 rect with row-stride mismatch becomes 64
separate CP submissions.

What's missing: `vx_enqueue_read_rect`, `vx_enqueue_write_rect`,
`vx_enqueue_copy_rect` (origin/region/row_pitch/slice_pitch), plus
`vx_enqueue_fill_buffer` (pattern bytes + size).

The CP can already accept a 3D-stride DMA descriptor in CMD_MEM_COPY
once the DMA engine grows the field — this is mostly a runtime API
+ a wider on-wire descriptor, not new HW.

### 3.6 No async kernel-image cache — 1.2 perf

OpenCL 1.2 does not require it, but the cost of *not* having it is
the single largest avoidable overhead in [pocl-vortex.c:660-680](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c#L660-L680) —
1 MB of kernel image re-DMA'd per launch. POCL knows the cache key
(program object → device-local content hash); vortex2 doesn't expose
a way to communicate it.

Even with §3.2 fixed, every `vx_module_load_file` would re-DMA without
a cache. What's missing: a content-addressed module store inside the
runtime, keyed by SHA of the .vxbin bytes; `vx_module_load_file`
becomes a lookup-then-upload. Refcounted on the runtime side so
multiple POCL `cl_program` objects sharing the same binary share
device memory.

### 3.7 No DCR access for per-kernel features (profiling, gfx state) — 1.2 perf

The legacy [pocl-vortex.c:662](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c#L662)
calls `vx_dump_perf` to read MPM counters — needed for
`clGetEventProfilingInfo` quality and POCL's perf dump. vortex2.h has
`vx_enqueue_dcr_read/write` ([vortex2.h:225-235](../../sw/runtime/include/vortex2.h#L225-L235))
but `vx_dump_perf`-shaped helpers (formatted counter dump per core /
per cluster / per cache) aren't wrapped. Same for graphics-pipeline
DCRs once Vortex's gfx extensions ride the CP.

Lowest priority — `vx_enqueue_dcr_read` is sufficient, this is purely
a convenience wrapper.

---

## 4. Target architecture

```
┌──────────────────────────────────────────────────┐
│              POCL (~250 lines)                   │
│  pocl-vortex.c device-ops:                       │
│  init   → vx_device_open                         │
│  alloc  → vx_buffer_create                       │
│  write  → vx_enqueue_write   → cl_event bridge   │
│  read   → vx_enqueue_read    → cl_event bridge   │
│  copy   → vx_enqueue_copy    → cl_event bridge   │
│  run    → vx_kernel_set_arg* + vx_enqueue_launch │
│  notify → vx_event_set_callback                  │
│  (NO polling thread, NO driver thread,           │
│   NO host-side arg marshalling)                  │
└──────────────────────────────────────────────────┘
                       │ vortex2.h
                       ▼
┌──────────────────────────────────────────────────┐
│  libvortex.so dispatcher (sw/runtime/common)     │
│  vx::Device / Queue / Buffer / Event             │
│  + module cache (§5.6)                           │
│  + event callback registry (§5.1)                │
└──────────────────────────────────────────────────┘
                       │ callbacks_t
                       ▼
┌──────────────────────────────────────────────────┐
│  Backend (simx / rtlsim / xrt / opae / gem5)     │
│  cp_mmio_* + mem_* + queue_caps                  │
└──────────────────────────────────────────────────┘
```

### 4.1 cl_event ↔ vx_event_h bridge

```c
struct vortex_event_data {
  vx_event_h vx_ev;   // owns one ref
};

void pocl_vortex_event_completed_cb(vx_event_h ev, vx_event_status_e s,
                                    void* user) {
  cl_event clev = (cl_event)user;
  if (s == VX_EVENT_STATUS_COMPLETE)
    pocl_update_event_completed(clev);
  else if (s == VX_EVENT_STATUS_ERROR)
    pocl_update_event_failed(CL_FAILED, NULL, 0, clev, NULL);
}

// In every pocl_vortex_{read,write,copy,run,...}:
vx_event_h vx_ev = NULL;
vx_enqueue_<op>(queue, ..., &vx_ev);
vx_event_set_callback(vx_ev, VX_EVENT_STATUS_COMPLETE,
                      pocl_vortex_event_completed_cb, cl_ev);
```

POCL's `pocl_vortex_driver_thread` is **deleted**. Its only job was
to serialize blocking `vx_*` calls; that's no longer needed. The CP
worker on the runtime side already does the work.

### 4.2 Kernel-arg marshalling

```c
// pocl_vortex_create_kernel
vx_kernel_h kobj;
vx_module_get_kernel(program_data->vx_module, kernel->name, &kobj);
kernel->data[device_i] = kobj;

// pocl_vortex_run (replaces lines 480-712 — 230 lines → ~30)
vx_kernel_h k = kernel->data[device_i];
for (int i = 0; i < meta->num_args; ++i) {
  struct pocl_argument* a = &cmd->command.run.arguments[i];
  if (ARG_IS_LOCAL(meta->arg_info[i])) {
    vx_kernel_set_local_arg(k, i, a->size);
  } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
    if (a->is_raw_ptr) {
      cl_ulong addr; memcpy(&addr, a->value, sizeof(addr)); addr += a->offset;
      vx_kernel_set_arg(k, i, sizeof(addr), &addr);
    } else if (a->value) {
      vortex_buffer_data_t* bd = ((cl_mem)(*(cl_mem*)a->value))
                                 ->device_ptrs[mid].extra_ptr;
      uint64_t addr = bd->buf_address + a->offset;
      vx_kernel_set_arg(k, i, sizeof(addr), &addr);
    } else {
      uint64_t zero = 0; vx_kernel_set_arg(k, i, sizeof(zero), &zero);
    }
  } else {
    vx_kernel_set_arg(k, i, a->size, a->value);
  }
}

vx_launch_info_t li = {
  .struct_size = sizeof(li),
  .kernel      = k,                                   // vx_kernel_h now
  .ndim        = pc->work_dim,
  .grid_dim    = { pc->num_groups[0], pc->num_groups[1], pc->num_groups[2] },
  .block_dim   = { pc->local_size[0], pc->local_size[1], pc->local_size[2] },
  .lmem_size   = local_mem_size,
};

vx_event_h ev = NULL;
vx_enqueue_launch(dd->vx_queue, &li, n_wait, wait_evs, &ev);
vx_event_set_callback(ev, VX_EVENT_STATUS_COMPLETE,
                      pocl_vortex_event_completed_cb, cmd_event);
```

230 lines down to ~30. The args struct, kernel-id switch, kargs
device-buffer allocation, free-and-reupload kernel dance, and explicit
`vx_ready_wait` all go away.

### 4.3 Module cache

```c
// pocl_vortex_post_build_program:
vx_module_h mod;
vx_module_load_file(dd->vx_device, sz_program_vxbin, &mod);
program->data[device_i] = mod;

// pocl_vortex_free_program:
vx_module_release(program->data[device_i]);
```

Refcount > 1 if two POCL programs hash to the same .vxbin. Re-launch
of the same kernel pays one DMA upload, ever. Compare with
[pocl-vortex.c:660-680](https://github.com/vortexgpgpu/pocl/blob/vortex_2.x/lib/CL/devices/vortex/pocl-vortex.c#L660-L680).

---

## 5. vortex2.h additions

Numbered to match §3 gap numbers. All §5.1–§5.7 are additive
(no breaking changes to current `vortex2.h` callers).

### 5.1 `vx_event_set_callback` — closes §3.1

```c
typedef void (*vx_event_callback_fn)(vx_event_h ev,
                                     vx_event_status_e status,
                                     void* user_data);

vx_result_t vx_event_set_callback (vx_event_h ev,
                                   vx_event_status_e on_status,
                                   vx_event_callback_fn fn,
                                   void* user_data);
```

Hooks into the per-queue worker's existing completion path; just an
extra `std::function` call after `event->complete()`.

### 5.2 Module + kernel handles — closes §3.2

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
```

`vx_launch_info_t.kernel` changes from `vx_buffer_h` to `vx_kernel_h`.
Internally `vx_kernel_h` knows its PC and arg layout, so
`vx_enqueue_launch` no longer programs `STARTUP_ADDR0/1` via the
caller's flat byte buffer.

### 5.3 Kernel arg API — closes §3.3

```c
vx_result_t vx_kernel_set_arg          (vx_kernel_h k, uint32_t idx,
                                        size_t size, const void* data);
vx_result_t vx_kernel_set_local_arg    (vx_kernel_h k, uint32_t idx,
                                        size_t size);

vx_result_t vx_kernel_get_max_block_size (vx_kernel_h k,
                                          uint32_t* x, uint32_t* y, uint32_t* z);
```

`vx_kernel_h` owns its arg block. `vx_enqueue_launch` reads from it.
`vx_kernel_set_arg` may be called from any thread between launches
(uses an internal mutex). Concurrent launches of the same `vx_kernel_h`
with different arg values: clone the args buffer at enqueue time.

### 5.4 Async map/unmap — closes §3.4

```c
vx_result_t vx_enqueue_map   (vx_queue_h q, vx_buffer_h buf,
                              uint64_t offset, uint64_t size, uint32_t flags,
                              uint32_t nw, const vx_event_h* w,
                              vx_event_h* out_event,
                              void** out_host_ptr);

vx_result_t vx_enqueue_unmap (vx_queue_h q, vx_buffer_h buf, void* host_ptr,
                              uint32_t nw, const vx_event_h* w,
                              vx_event_h* out_event);
```

Backends without a coherent host BAR (xrt, opae today) implement this
as "alloc host bounce buffer, queue a `vx_enqueue_read` after wait
events resolve, signal completion." simx/rtlsim/gem5 can short-circuit
to a host pointer into the simulator RAM.

### 5.5 Rect + fill DMA — closes §3.5

```c
typedef struct {
    size_t   struct_size;
    const void* next;
    size_t      buffer_origin[3];
    size_t      host_origin  [3];     // for {read,write}_rect
    size_t      region       [3];
    size_t      buffer_row_pitch;
    size_t      buffer_slice_pitch;
    size_t      host_row_pitch;
    size_t      host_slice_pitch;
} vx_rect_info_t;

vx_result_t vx_enqueue_read_rect  (vx_queue_h q,
                                   void* host_dst,
                                   vx_buffer_h src,
                                   const vx_rect_info_t* info,
                                   uint32_t nw, const vx_event_h* w,
                                   vx_event_h* out);
vx_result_t vx_enqueue_write_rect (...);
vx_result_t vx_enqueue_copy_rect  (...);

vx_result_t vx_enqueue_fill_buffer(vx_queue_h q, vx_buffer_h dst,
                                   uint64_t offset, uint64_t size,
                                   const void* pattern, size_t pattern_size,
                                   uint32_t nw, const vx_event_h* w,
                                   vx_event_h* out);
```

Initial implementation: host-side software fallback that decomposes
into linear `vx_enqueue_{read,write,copy}` + `memset` for fill, when
the CP DMA engine doesn't natively support strided ops. Behind the
API, so POCL keeps one entry point and the runtime evolves.

### 5.6 Content-addressed module cache — closes §3.6

No new public API — `vx_module_load_file` internally SHA-256s the
file, dedups against the device-resident module store, refcounts.
This is purely a dispatcher implementation detail.

`vx_module_load_bytes` takes an explicit hash arg for callers that
already computed it (POCL has the program object's hash):

```c
vx_result_t vx_module_load_bytes_hashed (vx_device_h dev,
                                         const void* bytes, size_t size,
                                         const uint8_t hash[32],
                                         vx_module_h* out);
```

### 5.7 Perf helpers — closes §3.7

```c
vx_result_t vx_device_dump_perf (vx_device_h dev, FILE* out);
```

Wraps the legacy `vx_dump_perf`. Async variant not needed — this is
called from `clFinish` paths.

---

## 6. Phases

Scoped to OpenCL 1.2 conformance. SVM and `cl_khr_command_buffer`
are deferred to §9.

### Phase 0 — vortex2.h skeleton additions (runtime-only) ✅ no POCL changes

Land §5.1, §5.2, §5.3, §5.6 as additive APIs in `vortex2.h`. Legacy
`vortex.h` wrapper continues to work unchanged. Unit-test via
`tests/runtime/test_module_kernel.cpp` and a v2 smoke that creates
a module, gets two kernels, launches each with set-arg.

**Definition of done:** runtime tests pass on all four backends; POCL
unchanged.

### Phase 1 — POCL device-ops migration on bare vortex2

Rewrite `pocl-vortex.c` ops on the new vortex2 surface. Bridge
`cl_event ↔ vx_event_h` via §5.1 callbacks. Delete
`pocl_vortex_driver_thread`. Delete the per-launch kernel re-upload
(now handled by §5.6 module cache). Replace 230-line kargs marshalling
with §5.3 API.

**Definition of done:** the POCL regression suite (`make check` in
the POCL build dir) passes on Vortex with simx and rtlsim drivers,
including all 1.2 conformance tests that don't depend on §5.4/§5.5
yet. End-to-end time on `tests/runtime/test_vecadd` ≤ 80% of the
v3-proposal baseline (re-upload elimination should dominate).

### Phase 2 — rect / fill / async-map (§5.4, §5.5)

Add the rect/fill enqueues with the software fallback. Add async
map/unmap. POCL starts using them where the current code aborts
or punts.

**Definition of done:** POCL's 1.2 conformance suite passes the
`clEnqueueRead/WriteBufferRect`, `clEnqueueCopyBufferRect`,
`clEnqueueFillBuffer`, `clEnqueueMapBuffer`, and
`clEnqueueUnmapMemObject` tests on simx and rtlsim. (Image-format
tests still abort — that's a hardware extension issue, not a runtime
one.)

### Phase 3 — Perf + cleanup

Land §5.7 (`vx_device_dump_perf`). Remove all legacy `vortex.h`
calls from `pocl-vortex.c`. Remove `kernel_args.h` and
`kernel_main.c`'s `kernel_id` switch (no longer needed;
`vx_module_get_kernel` resolves to PC directly). Remove unused
`vx_check_occupancy` glue in POCL — use `vx_kernel_get_max_block_size`.

**Definition of done:** `grep -r "vx_dev_open\|vx_start\|vx_ready_wait\|vx_copy_to_dev\|vx_dump_perf\|vx_check_occupancy" lib/CL/devices/vortex/` returns nothing. Full OpenCL 1.2
conformance still green.

---

## 7. Risks & open questions

**R1. `vx_event_set_callback` reentrancy.** The callback fires from
the runtime's per-queue worker thread. POCL's
`pocl_update_event_completed` takes locks. If POCL's lock order
across notify paths assumes a single notify thread, we may deadlock
when two queues complete simultaneously. Mitigation: POCL's
`basic`/`pthread` driver already handles this (they have per-device
worker threads with similar patterns). Survey
`pocl_update_event_completed` callers before Phase 1.

**R2. Module cache eviction.** §5.6 says refcounted; what if a
long-running app loads thousands of distinct programs? Cap memory
with an LRU, evict when device-side store hits a threshold. Open:
what threshold (% of device global mem)?

**R3. Phase 0 timing.** POCL won't see any benefit until Phase 1.
The runtime work in Phase 0 is pure additive surface; if review
pushback delays it, POCL keeps shipping on
[pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md)'s legacy
plan, and this proposal starts when Phase 0 lands.

---

## 8. Out-of-scope

- **Image / sampler support.** Still aborts in POCL. Requires Vortex
  TEX hardware exposure through CP, which is its own proposal.
- **Multi-device contexts.** POCL `cl_context` over multiple Vortex
  cards. Possible once we have multiple FPGA targets; gem5 multi-device
  exists but no API plumbing.
- **`vx_queue_h` mapping onto multiple HW queues.** Separate
  dispatcher-layer optimization, gated on the multi-queue work
  noted earlier.

---

## 9. Post-OpenCL-1.2 — deferred

Captured here so the API surface designed in §5 is forward-compatible.
None of these are in scope for the phases in §6.

### 9.1 SVM / unified memory (OpenCL 2.0)

OpenCL 2.0+ SVM and `cl_ext_buffer_device_address`, which chipStar/HIP
needs. vortex2.h has no host-visible mapping of device memory and
no unified address space.

Future addition:
```c
#define VX_MEM_UNIFIED              (1u << 16)
#define VX_CAPS_SVM_SUPPORTED       0x100   // 0/1
#define VX_CAPS_SVM_GRANULARITY     0x101   // byte alignment for atomics
vx_result_t vx_alloc_unified (vx_device_h dev, uint64_t size,
                              uint32_t flags, vx_buffer_h* out);
```

Backend-side: simx/rtlsim/gem5 implement as host malloc + register the
range with the simulator's address translator. xrt/opae require AFU
support — return `VX_ERR_NOT_SUPPORTED` when the cap is 0. The cap
query keeps the API decoupled from hardware reality per backend.

Note for §5.4 (`vx_enqueue_map`): when SVM lands, map can short-circuit
to a refcount-bump on an already-mapped page; today's bounce-buffer
implementation is forward-compatible.

### 9.2 `cl_khr_command_buffer` (extension, not 1.2 core)

OpenCL extension that records a sequence of commands once and replays
them — CUDA Graphs / HIP Graphs analog. Big win for graph-shaped
workloads (ML inference, repeated stencils).

Future addition:
```c
typedef struct vx_cmdbuf* vx_cmdbuf_h;
vx_result_t vx_cmdbuf_begin           (vx_device_h dev, vx_cmdbuf_h* out);
vx_result_t vx_cmdbuf_end             (vx_cmdbuf_h cb);
vx_result_t vx_cmdbuf_record_launch   (vx_cmdbuf_h cb,
                                       const vx_launch_info_t* info);
vx_result_t vx_cmdbuf_record_read     (vx_cmdbuf_h cb, ...);
vx_result_t vx_cmdbuf_record_write    (vx_cmdbuf_h cb, ...);
// ... record_copy, record_barrier, retain, release ...
vx_result_t vx_enqueue_cmdbuf         (vx_queue_h q, vx_cmdbuf_h cb,
                                       uint32_t nw, const vx_event_h* w,
                                       vx_event_h* out_event);
```

Implementation strategy: a `vx_cmdbuf` is a list of CP CL descriptors
in host memory. `vx_cmdbuf_end` pre-allocates a device ring slice
sized for the descriptors and uploads them once. `vx_enqueue_cmdbuf`
is a CP DMA-copy from the cached slice into the live ring + tail
bump. Replays cost one DMA-copy and one MMIO write, independent of
the number of recorded commands.

Open question for whenever this lands: invalidation when a referenced
buffer is freed. OpenCL semantics: cmdbuf holds a ref. Match Vulkan/
CUDA — cmdbuf keeps the buffer alive.

### 9.3 Multi-HW-queue `vx_queue_h` mapping

vortex2's `vx_queue_h` is software; multiple POCL `cl_command_queue`
serialize through CP queue 0. Not a 1.2 conformance issue, but the
biggest concurrency lever left on the table for workloads that
already saturate one queue.
