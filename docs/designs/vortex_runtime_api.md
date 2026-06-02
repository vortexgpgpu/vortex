# Vortex Runtime API (`vortex2.h`) — Design

**Scope:** the shape and design philosophy of the `vortex2.h` async
runtime API — the locked v1 surface, its handle/function set, and the
three "shape-lock" features (timeline events, module/kernel handles,
raw-pointer UVA args). Covers
[`sw/runtime/include/vortex2.h`](../../sw/runtime/include/vortex2.h) and
the common implementation in
[`sw/runtime/common/`](../../sw/runtime/common/).

The CP submit path that backs this API is in
[`command_processor_control_plane.md`](command_processor_control_plane.md) §9; this
document is about the **API shape** itself.

---

## 1. Design philosophy

`vortex2.h` is a minimal, Vulkan/CUDA-style core: a small set of opaque
handles and functions, with complexity pushed to upper-layer translators
(PoCL, chipStar, vortexpipe) and per-block helpers rather than into the
core runtime. The governing rule is **additive vs. shape-breaking**: the
v1 surface is locked, and future capability is added without changing the
existing handle/function shapes.

Handles ([`vortex2.h:47-53`](../../sw/runtime/include/vortex2.h#L47)):
`vx_device_h`, `vx_buffer_h`, `vx_queue_h`, `vx_event_h`, `vx_module_h`,
`vx_kernel_h`.

---

## 2. The three shape-lock features

### 2.1 Timeline events

`vx_event` is a host-side refcounted struct with a `std::atomic` monotonic
counter + condvar
([`sw/runtime/common/event.cpp:55-64`](../../sw/runtime/common/event.cpp#L55)).
API: `vx_event_create/signal/get_value/wait_value/wait_values`, plus
`vx_enqueue_signal`/`vx_enqueue_wait_value` and
`vx_event_{retain,release,get_profiling}`
([`vortex2.h:446-484`](../../sw/runtime/include/vortex2.h#L446)). A
device-side counter-slot mirror is present so the CP's `CMD_EVENT_WAIT`
sees host signals (the optional perf path). This replaced the legacy
`vx_user_event_*` / `vx_event_status` / `vx_event_wait_all` entry points
(all removed).

### 2.2 Module + kernel handles

`vx_module_load_file/bytes` parse a `.vxbin` (with its `VXSYMTAB` footer),
`vx_module_get_kernel(name)` returns a `vx_kernel_h`, with
`vx_module/kernel_{retain,release}` and `vx_kernel_get_max_block_size`
([`vortex2.h:293-308`](../../sw/runtime/include/vortex2.h#L293)). Legacy
single-entry binaries fall back to a single `"main"` entry. The on-disk
format and dispatch are in
[`kernel_entry_and_dispatch.md`](kernel_entry_and_dispatch.md).

### 2.3 Raw-pointer (UVA) args

`vx_launch_info_t` carries a `vx_kernel_h kernel` plus
`const void* args_host` / `size_t args_size`
([`vortex2.h:179-202`](../../sw/runtime/include/vortex2.h#L179)) — a host
blob that the runtime stages into a per-device scratch slot
(`args_slot_get/release`,
[`queue.cpp:289-359`](../../sw/runtime/common/queue.cpp#L289)) and programs
into the KMU ARG DCRs. The caller never allocates an args buffer. A
`kernel == NULL` (and `args_host == NULL`, `ndim == 0`) "legacy escape
hatch" lets a caller pre-program the KMU PC/ARG DCRs directly.

---

## 3. Proposed but not yet implemented

The API ships the locked post-removal shape directly (no deprecation-shim
transition window was used). Deferred, out-of-scope-by-design:

1. **Memory/event pools, indirect dispatch, host functions, SVM, command
   buffers, per-arg setters** — explicitly deferred (the "additive"
   surface for a future v2).
2. **Risk-register items** worth preserving as forward guidance: waiter
   wakeup-storm mitigation; buffer-pointer width marshalling (a proposed
   `vx_kernel_get_pointer_size` helper — only `get_max_block_size`
   shipped); scratch-pool growth/sizing; and external-client shim timing.

**Superseded directions** (recorded to avoid revival): the phased
dual-API / deprecated-shim rollout (Phase 1a/1b shims) — the API shipped
the final shape directly; and the kernel-handle disambiguation *tag* on
`vx_launch_info_t.kernel` — replaced by the simpler `NULL` escape hatch.

The PoCL `vortex2.h` API additions (event callbacks `vx_event_set_callback`
and `vx_kernel_set_arg`/`set_local_arg`) requested by the retained
`pocl_on_vortex2_proposal.md` are **not** in the header yet — see that
proposal.

---

## 4. Source proposal

This design consolidates and supersedes `vortex2_v1_shape_lock_proposal.md`
(now removed from `docs/proposals/`).
