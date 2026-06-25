# Multi-core kernel-launch failure — CTA-dispatch device-`busy` gap

**Status:** Fixed — prism_v3 commit `92d3d28f`
**Date:** 2026-06-25
**Component:** `hw/rtl/core/VX_cta_dispatch.sv` (core CTA / kernel-launch path)
**Scope:** non-graphics; affects any `SOCKET_SIZE>1` or multi-socket build
**Upstream relevance:** the identical buggy code exists in `vortex_ci`
(`VX_cta_dispatch.sv` + `VX_socket.sv`), so this is an upstream candidate.

---

## 1. Summary

On multi-core configurations the device could report itself idle **one cycle into
kernel launch**, before the kernel ever ran. The rtlsim host's completion-wait
loop latched that transient idle as "done", stopped clocking the model, and read
back an unwritten result (e.g. a blank framebuffer for graphics, stale output for
compute). The failure was intermittent across rebuilds because it hinged on a
1-cycle timing window perturbed by Verilator's randomized reset state.

The trigger is a **gap in the aggregated device `busy` signal during the
KMU → core CTA-dispatch handoff**, present only when the per-core busy is
registered before aggregation (`SOCKET_SIZE>1`).

---

## 2. Symptom

- `gfx_draw3d` at `NUM_CORES≥2`: triangle/scene entirely missing (framebuffer =
  clear color), `instrs≈43` (only the `vx_start` prologue ran), test FAILS.
- `NUM_CORES=1`: always PASS.
- General multi-core compute (e.g. `vecadd`) at `cores=2`: PASS — so it looked
  graphics-specific, and the pixel-error count varied run-to-run (7 / 128 / 2048),
  which looked like a flaky raster race.

This misdirected debugging toward the graphics RASTER fragment-distribution path
("FWD-6 raster drain race"). That was a **red herring**: the raster unit never ran
because the kernel never launched.

---

## 3. How it was localized

A cores=2 pipeline trace (`DBG_TRACE_PIPELINE`) of `gfx_draw3d` showed:

- Both cores commit only the prologue PCs `0x80000000..0x80000018` (over ~400
  cycles — cold-icache misses), then the device goes idle.
- The kernel entry is read correctly: `csrr s11, 0xce1` (kernel entry) = `0x80000024`
  and `csrr a0, mscratch` (arg) = `0x13000`. The `jalr s11` is committed and the
  scheduler **dispatches PC `0x80000024`** (kernel_main), but it never commits.

Instrumenting the rtlsim host loop pinned it precisely:

```
[FWD6DBG] busy rose at t=194
[FWD6DBG] run loop exit at t=198 busy=0 mon_break=0   <-- idle after 4 cycles
```

`busy` deasserted 4 cycles after rising, with `mon_break=0` (not an HTIF/tohost
completion) — i.e. the device genuinely reported idle during launch.

---

## 4. Root cause

### 4.1 The signals

Device `busy` is an OR-reduction up the hierarchy (`Vortex.sv`):

```
busy = kmu_busy | dcr | (| per_cluster_busy)
```

- **`kmu_busy`** (`VX_kmu.sv`): `assign busy = running;` — the KMU is "running"
  only while it is *issuing* CTAs onto the broadcast bus. `running` clears the
  cycle the **last** CTA fires (`kmu_bus_if_fire` on the final descriptor).

- **per-core `busy`** (`VX_cta_dispatch.sv`):
  ```
  assign busy = (state == DISPATCH);
  ```
  The dispatcher's FSM is `IDLE → DISPATCH`. The accept of a CTA
  (`kmu_bus_if_fire` in `IDLE`) sets `state <= DISPATCH` — a **registered**
  transition. So on the accept cycle itself `state` is still `IDLE` and the core
  reports **not busy**.

- **socket aggregation** (`VX_socket.sv`):
  ```
  `BUFFER_EX(busy_r, dcr | (| per_core_busy), 1'b1, 1, (SOCKET_SIZE > 1));
  ```
  When `SOCKET_SIZE>1` this inserts a **1-cycle register** between per-core busy
  and the socket's reported busy. (`SOCKET_SIZE=1` ⇒ combinational, no delay.)

### 4.2 The handoff race (final CTA, `SOCKET_SIZE>1`)

Let cycle **N** be when the last CTA fires (`kmu_bus_if_fire`):

| cycle | `kmu_busy` | core busy (`state==DISPATCH`) | socket busy (buffered) | device `busy` |
|-------|-----------|-------------------------------|------------------------|---------------|
| N     | 1 (running)| 0 (state still IDLE on accept)| buf(per_core@N-1)=0    | **1** (kmu)   |
| N+1   | 0 (running cleared) | 1 (state→DISPATCH)   | buf(per_core@N=0)=**0**| **0** ← gap   |
| N+2   | 0          | 1                             | buf(per_core@N+1=1)=1  | 1             |

At **N+1** the KMU has finished dispatching (`kmu_busy=0`) but the core's busy —
which only rose at N+1 and is further delayed one cycle by the socket buffer —
has not yet reached the aggregation. Device `busy` is **0 for one cycle**.

### 4.3 Why it becomes an observable failure

The rtlsim host (`sim/rtlsim/processor.cpp`) waits for completion with an
edge-sensitive loop:

```cpp
while (!device_->busy) tick();   // wait for launch
while ( device_->busy) tick();   // wait for completion  <-- exits on the 1-cycle gap
```

The second loop exits the instant it samples `busy==0`. The N+1 gap satisfies
that, so the host stops clocking the model while the warps are only at the kernel
entry, then reads back an unwritten result.

### 4.4 Why `cores=1` always worked

With `SOCKET_SIZE=1` the socket busy is combinational (no `BUFFER_EX` register),
so per-core busy reaches the aggregation with no lag and there is no gap. Hence
the bug was multi-core-only and appeared "X-init flaky" — it depended on the exact
cycle alignment of the KMU drop vs. the buffered core-busy rise, which shifts with
Verilator's randomized reset.

---

## 5. Fix

`hw/rtl/core/VX_cta_dispatch.sv`:

```systemverilog
-    assign busy = (state == DISPATCH);
+    // Busy from the cycle a CTA is ACCEPTED (kmu_bus_if_fire), not just while in
+    // DISPATCH. The accept→DISPATCH transition is registered, so gating on state
+    // alone leaves the accept cycle un-busy. With SOCKET_SIZE>1 the socket-level
+    // busy aggregation is registered (1-cycle lag), so on the final CTA dispatch
+    // kmu_busy drops before the buffered per_core_busy rises — a 1-cycle device
+    // busy gap the host's edge-sensitive idle-wait latches as premature completion
+    // (cores>1 kernel-launch failure). Covering the accept cycle closes the gap.
+    assign busy = (state == DISPATCH) || kmu_bus_if_fire;
```

The dispatcher is committed to work the moment it accepts a CTA, so its `busy`
must cover the accept cycle. With the fix, per-core busy is asserted at cycle N;
the socket buffer presents it at N+1, exactly when `kmu_busy` drops — the device
`busy` stays continuously high across the handoff:

| cycle | `kmu_busy` | core busy (fixed) | socket busy (buffered) | device `busy` |
|-------|-----------|-------------------|------------------------|---------------|
| N     | 1         | **1** (accept)    | buf(@N-1)=0            | 1             |
| N+1   | 0         | 1                 | buf(per_core@N=**1**)  | **1**         |

This cannot suppress a legitimate idle: `kmu_bus_if_fire` is only high during
dispatch, so once dispatch ends and warps run, `busy` is driven solely by the
scheduler's active-warp/pending state and drops normally at true completion.

---

## 6. Validation (rtlsim, prism_v3)

After fix — all PASS, no regressions:

| Test | Config | Result |
|------|--------|--------|
| gfx_draw3d (triangle, 128px) | cores=1 / SS=1 | PASS |
| gfx_draw3d (triangle, 128px) | cores=2 / SS=2 | PASS |
| gfx_draw3d (triangle, 128px) | cores=4 / SS=4 | PASS |
| gfx_draw3d (triangle, 32px)  | cores=2 / SS=1 (2 sockets) | PASS |
| vecadd  | cores=1, cores=2 | PASS |
| sgemm   | cores=2 | PASS |
| async_barrier | cores=2 | PASS |

Before fix, gfx cores=2 ran `instrs=43` (prologue only); after, `instrs=6508`
(32px) / `72376` (128px) — the kernel runs to completion.

---

## 7. Implications & scope

- **Non-graphics, core path.** The bug is in the shared CTA dispatcher / device
  busy aggregation; graphics merely exposed it first (its persistent-worker kernel
  + long cold-icache prologue widened the window). Any multi-core kernel at
  `SOCKET_SIZE>1` / multi-socket is exposed.
- **Likely related to prior multi-core CTA-dispatch parity gaps** — worth
  re-checking SimX↔RTL parity at cores>1 after this fix.
- **Other busy/done consumers.** The `busy` signal genuinely glitches in RTL, so
  consumers beyond rtlsim (XRT AFU completion, opae) could in principle be
  affected; the fix corrects the signal at its source rather than papering over it
  in the host.
- **Upstream:** the same `assign busy = (state == DISPATCH);` and the
  `BUFFER_EX(... SOCKET_SIZE>1)` socket aggregation are present in `vortex_ci`
  (clean-master line). Recommend reproducing a multi-core launch there, then
  upstreaming this one-line change.
