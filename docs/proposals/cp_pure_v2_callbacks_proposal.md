# CP-Pure v2 Callbacks + Software CP for simx/rtlsim

**Status:** Drafted May 17 2026 (after `196c4e56` CP engine retire-on-done).
**Scope:** Strip `callbacks_t` to pure vortex2.h primitives by replacing
backend-specific launch + DCR callbacks with a single CP MMIO interface,
and add a shared software `CommandProcessor` class so simx and rtlsim can
satisfy that interface without a hardware CP.

Companion docs:
- [`command_processor_proposal.md`](command_processor_proposal.md) — the
  CP architecture this builds on.
- [`cp_xrt_integration_plan.md`](cp_xrt_integration_plan.md) — XRT
  integration that this generalizes.
- [`cp_opae_integration_plan.md`](cp_opae_integration_plan.md) — OPAE
  counterpart.

---

## 1. Motivation

Today `callbacks_t` ([sw/runtime/common/callbacks.h](../../sw/runtime/common/callbacks.h))
mixes platform primitives (memory, device lifecycle, queries) with two
legacy-shaped control-plane fields:

```c
int (*launch_start)(void* dev_ctx);                         // AP_CTRL "go" kick
int (*launch_wait) (void* dev_ctx, uint64_t timeout_ms);    // AP_DONE poll
int (*dcr_write)   (void* dev_ctx, uint32_t addr, uint32_t value);
int (*dcr_read)    (void* dev_ctx, uint32_t addr, uint32_t tag,
                    uint32_t* out_value);
```

These pre-date the Command Processor design and embed the v1 model
("host pokes registers, pokes AP_START, polls AP_DONE") into the
backend ABI. In a pure CP world the host instead:

1. Writes `CMD_DCR_WRITE` / `CMD_LAUNCH` descriptors to a ring in
   device memory (uses `mem_upload`).
2. Bumps `Q_TAIL` in the CP regfile to commit the ring entries.
3. Polls `Q_SEQNUM` in the CP regfile for completion.

So in the long term `launch_*` and `dcr_*` simply have no caller — the
dispatcher's v2 API path uses only `mem_upload` + CP regfile MMIO.
Keeping these fields forces every backend to maintain a synchronous
"start kernel / wait for done" path that the v2 API doesn't use, and
forces the simx/rtlsim runtimes to maintain a `start()/ready_wait()`
implementation parallel to (and inconsistent with) what xrt/opae now do.

**Goal:** make `callbacks_t` 100% pure vortex2.h:

```c
typedef struct {
  // Device lifecycle
  int (*dev_open)(void** out_dev_ctx);
  int (*dev_close)(void* dev_ctx);

  // Queries
  int (*query_caps)(void* dev_ctx, uint32_t caps_id, uint64_t* out_value);
  int (*memory_info)(void* dev_ctx, uint64_t* out_free, uint64_t* out_used);

  // Device memory
  int (*mem_alloc)(void* dev_ctx, uint64_t size, uint32_t flags,
                   uint64_t* out_dev_addr);
  int (*mem_reserve)(void* dev_ctx, uint64_t dev_addr, uint64_t size,
                     uint32_t flags);
  int (*mem_free)(void* dev_ctx, uint64_t dev_addr);
  int (*mem_access)(void* dev_ctx, uint64_t dev_addr, uint64_t size,
                    uint32_t flags);

  // DMA
  int (*mem_upload)(void* dev_ctx, uint64_t dst, const void* src,
                    uint64_t size);
  int (*mem_download)(void* dev_ctx, void* dst, uint64_t src, uint64_t size);
  int (*mem_copy)(void* dev_ctx, uint64_t dst, uint64_t src, uint64_t size);

  // Command Processor control plane (the ONLY control path)
  int (*cp_mmio_write)(void* dev_ctx, uint32_t offset, uint32_t value);
  int (*cp_mmio_read) (void* dev_ctx, uint32_t offset, uint32_t* value);
} callbacks_t;
```

That's it. Every kernel launch, every DCR write, every status query —
they all flow through `mem_upload` (writing CMD_* descriptors) plus
`cp_mmio_*` (writing Q_TAIL / reading Q_SEQNUM).

---

## 2. Problem: simx and rtlsim have no CP

`xrt` and `opae` ship a hardware CP (`VX_cp_core` is in their AFU). They
already implement `cp_mmio_write/read` trivially — `fpgaWriteMMIO64` to
byte offset `0x1000+` ([XRT integration commit `15440a55`](../../hw/rtl/afu/xrt/VX_afu_wrap.sv), [OPAE commit `8b4fdc8b`](../../hw/rtl/afu/opae/vortex_afu.sv)).

`simx` and `rtlsim` don't have a CP. They run Vortex directly (functional
or RTL) without the surrounding AFU+CP fabric. Today they implement
`launch_start` by calling `processor_.start()` and `dcr_write` by
calling `processor_.dcr_write()` — both routes that bypass the CP
entirely.

If we strip the legacy callbacks, simx and rtlsim need a way to satisfy
`cp_mmio_*` and to do whatever the hardware CP does internally
(fetch ring, dispatch DCRs to Vortex, signal launch).

---

## 3. Proposal: shared `CommandProcessor` C++ simulator

Add a new C++ class `vortex::CommandProcessor` in `sim/common/` that
models the hardware CP functionally. Both simx and rtlsim instantiate
one, wire it to their existing `Processor` (Vortex), and tick it once
per simulator cycle.

### 3.1 Header sketch (`sim/common/CommandProcessor.h`)

```cpp
namespace vortex {

class CommandProcessor {
public:
  // The backend gives us a way to:
  //   - read CP commands from device DRAM (ring buffer fetches)
  //   - write seqnum back to device DRAM (completion writebacks)
  //   - issue DCR writes to Vortex (for CMD_DCR_WRITE)
  //   - kick Vortex / observe its busy state (for CMD_LAUNCH)
  struct Hooks {
    std::function<void(uint64_t addr, void* dst, size_t bytes)> dram_read;
    std::function<void(uint64_t addr, const void* src, size_t bytes)> dram_write;
    std::function<void(uint32_t addr, uint32_t value)> vortex_dcr_write;
    std::function<void()> vortex_start;        // pulse vx_start
    std::function<bool()> vortex_busy;         // read vx_busy
  };

  explicit CommandProcessor(const Hooks& hooks);

  // Host-facing MMIO surface (same address map as VX_cp_axil_regfile §17).
  void     mmio_write(uint32_t off, uint32_t value);
  uint32_t mmio_read (uint32_t off) const;

  // Advance the CP one functional "cycle". Called by the simulator's
  // per-cycle (rtlsim) or per-instruction-batch (simx) loop. The number
  // of FSM steps per tick is small (single-digit) so this is cheap.
  void tick();

  // Optional: in NO-CP mode the backend can still write DCRs / start
  // Vortex directly (helpful during early bring-up). When the dispatcher
  // is built CP-pure, those direct paths are unused.
  bool enabled() const;

private:
  // Per-queue state (head, tail, base, control, seqnum)
  // Engine FSM (mirrors VX_cp_engine.sv)
  // DCR proxy FSM, Launch FSM, DMA FSM (mirrored functionally)
  // ...
};

} // namespace vortex
```

### 3.2 Why a single-threaded tick model (not a worker thread)

The user proposal mentioned running the CP in a separate thread for
realism. I'd argue against:

| Concern | Tick model | Separate thread |
|---|---|---|
| **Determinism** | Each sim cycle advances CP deterministically; reproducible | Race against `Processor::run()` → non-deterministic ordering of memory + DCR accesses; reproducibility lost |
| **simx use case** | simx is a *functional* simulator — its whole reason to exist is fast, deterministic test runs. A threaded CP forces simx to add mutexes on `RAM`, `DCR`, and `Processor` interfaces, killing the fast-path | Forces simx to thread-protect every primitive |
| **rtlsim/Verilator** | Verilator's `eval()` is single-threaded by default. CP's `tick()` slots in alongside `eval()` cleanly | Concurrent thread would race against `eval()` — Verilator state isn't thread-safe |
| **Debugging** | Linear execution = `gdb` step works | Race conditions need TSAN, intermittent failures |
| **Performance** | Negligible (CP FSM is a handful of comparisons per tick) | Mutex acquire dominates; CP-host MMIO is high-frequency |
| **Realism** | Matches the hardware reality — the real CP is a synchronous FSM clocked off the same clock as Vortex, not an independent agent | Doesn't model real hardware better; it just adds artificial concurrency |

**Recommendation:** single-threaded `tick()` called once per simulator
cycle. Match what the hardware actually does.

### 3.3 Integration into simx

Current `sim/simx/Processor.cpp` runs Vortex one cycle (or one instruction
batch) at a time. simx's `vx_device::ready_wait()` polls `processor_.is_done()`.

New flow:
- `simx/vortex.cpp` instantiates `CommandProcessor` alongside `Processor`.
- The two CP hooks `vortex_dcr_write` and `vortex_start` route to
  `processor_.dcr_write` and `processor_.start`. The `vortex_busy`
  hook reads `processor_.busy()` (already exposed for `is_done`).
- The CP hooks `dram_read` / `dram_write` route to the existing `RAM`
  object.
- The backend's `cp_mmio_write` / `cp_mmio_read` callbacks forward
  directly to `cp_.mmio_write/read`.
- The main sim loop: while `cp_.enabled() || processor_.busy()`,
  call `cp_.tick()` and `processor_.tick()`.

### 3.4 Integration into rtlsim

rtlsim is Verilator-driven, but the top module is `Vortex` (not the
AFU). There's no MMIO bus at the top — just memory + DCR + start/busy
wires connected to test-bench logic.

Same pattern as simx:
- `rtlsim/vortex.cpp` instantiates `CommandProcessor`.
- `vortex_dcr_write` hook drives the Verilator `dcr_req_*` signals.
- `vortex_start` pulses `start`. `vortex_busy` reads `busy`.
- `dram_read/write` use the rtlsim DRAM model (`sim/common/mem.cpp`).
- Per Verilator cycle: tick the CP, then `top->eval()`.

### 3.5 NO-CP transitional mode (default: off)

Per user request: default `VORTEX_USE_CP=0` for simpler bring-up.

In NO-CP mode the `CommandProcessor` is still instantiated (to satisfy
the `cp_mmio_*` callbacks) but the *runtime* doesn't use the CP path.
Instead, the simx/rtlsim `vx_device` exposes a small "direct" surface
that the dispatcher uses when `cp_enabled_ == false`.

**But this is exactly the legacy `launch_start` / `dcr_write` shape we
want to strip!** Two ways to reconcile:

**(A)** Keep the legacy callbacks alive transitionally. `callbacks_t`
has both sets; dispatcher picks based on `cp_enabled_`. Cleanup deferred
until simx/rtlsim CP path is shaken out. (Pragmatic, partial cleanup.)

**(B)** Strip the legacy callbacks now. `cp_mmio_write` is the *only*
control path. When `VORTEX_USE_CP=0`, the simx/rtlsim CP class runs in
"transparent mode": each `CMD_DCR_WRITE` posted to the ring is
immediately consumed and forwarded via the `vortex_dcr_write` hook
(no FSM cycles, just a function call). Each `CMD_LAUNCH` immediately
fires `vortex_start` and blocks until `!vortex_busy`. This makes
`VORTEX_USE_CP` purely a "use fancy CP timing vs. fast-path
direct-forward" toggle, both via the same callback surface.

**Recommendation: (B).** Fewer code paths, cleaner ABI, and the
"transparent mode" is trivial to implement (it's literally what
the dispatcher already does today, just moved one layer down). The
debug story is the same — in NO-CP mode the dispatcher's behavior
is identical to today; only the impl moved.

---

## 4. Concrete change list

### 4.1 New files

| File | Purpose | ~LOC |
|---|---|---|
| `sim/common/CommandProcessor.h` | Class header + hooks struct | 60 |
| `sim/common/CommandProcessor.cpp` | FSM impl (engine, fetch, DCR proxy, launch, completion) + transparent mode | 350 |
| `hw/unittest/cp_sim/` | Standalone unit test exercising the C++ CP against a mock processor | 200 |
| `docs/proposals/cp_pure_v2_callbacks_proposal.md` | This doc | (done) |

### 4.2 Modified files

| File | Change |
|---|---|
| `sw/runtime/common/callbacks.h` | Drop `launch_start`, `launch_wait`, `dcr_write`, `dcr_read`. Add `cp_mmio_write`, `cp_mmio_read`. Stop including `<vortex.h>`; nothing in the header references it. |
| `sw/runtime/common/callbacks.inc` | Drop the lambdas that wire `launch_*` and `dcr_*`. Add `cp_mmio_*` lambdas that call `vx_device::cp_mmio_write/read`. |
| `sw/runtime/stub/vortex.cpp` | Replace `callbacks->launch_start/wait` calls with the CP ring submission helper (`cp_post_launch`-equivalent moved from xrt/opae runtime into the dispatcher itself). Replace `callbacks->dcr_write/read` calls with `cp_post_dcr_write` / `cp_post_dcr_read`. The dispatcher becomes the single source of truth for CP command building. |
| `sw/runtime/simx/vortex.cpp` | Remove `start()` / `ready_wait()` / `dcr_write()` / `dcr_read()` from `vx_device`. Add `cp_mmio_write/read(uint32_t, uint32_t)` that forward to the new `CommandProcessor`. Instantiate `CommandProcessor` in the ctor with hooks wired to `processor_` + `ram_`. Drive `cp_.tick()` from the main sim loop. |
| `sw/runtime/rtlsim/vortex.cpp` | Same shape as simx. |
| `sw/runtime/xrt/vortex.cpp` | Remove `start()` / `ready_wait()` / `dcr_write()` / `dcr_read()` from `vx_device` (move the CP ring submission into the dispatcher per row above). Add `cp_mmio_write/read` that wraps `write_register/read_register` to MMIO offset `0x1000 + off`. The `cp_post_launch` / `cp_post_dcr_write` helpers go away from here — they live in the dispatcher now. |
| `sw/runtime/opae/vortex.cpp` | Mirror of xrt. |
| `sw/runtime/stub/Makefile` | Add `CommandProcessor.cpp` reference? No — it lives in `sim/common/`. Backends that include the simulator (simx, rtlsim) link it; dispatcher doesn't. |
| `sw/runtime/simx/Makefile`, `sw/runtime/rtlsim/Makefile` | Add `$(SIM_COMMON_DIR)/CommandProcessor.cpp` to `SRCS`. |

### 4.3 Migration sequence

These can't all land at once without breaking the world mid-flight. Phased
ordering:

**Phase A — Stand up `CommandProcessor` class + unit test.**
Add the new files, write the FSM, unit-test it standalone with a mock
DRAM and mock hooks. No other files change. Commit.

**Phase B — Add `cp_mmio_*` callbacks alongside legacy ones.**
`callbacks_t` grows; nothing shrinks. simx/rtlsim wire their new
`CommandProcessor` to the new callbacks. xrt/opae's `cp_mmio_*` is a
trivial wrapper over their existing MMIO write/read. Legacy callbacks
stay populated. Verify nothing regresses. Commit.

**Phase C — Move CP ring helpers from backends into the dispatcher.**
`cp_post_launch` / `cp_post_dcr_write` (currently in xrt + opae
runtimes, repeated) move into `stub/vortex.cpp`. They use
`callbacks->cp_mmio_write` + `callbacks->mem_upload`. xrt/opae
runtimes shrink. Verify 8-corner regression. Commit.

**Phase D — Wire dispatcher's `vx_start` / `vx_ready_wait` to the
CP path.** Dispatcher always uses CP commands; the existing
`callbacks->launch_start/wait` calls go away from the dispatcher.
At this point simx/rtlsim's `CommandProcessor` runs in transparent
mode (no FSM cycles, immediate forward to Vortex). Verify everything.
Commit.

**Phase E — Strip legacy fields from `callbacks_t`.**
Remove `launch_start`, `launch_wait`, `dcr_write`, `dcr_read` from
the struct definition. Remove the corresponding lambdas in
`callbacks.inc`. Remove the now-dead methods from each backend's
`vx_device`. Verify. Commit.

Phase A and B can happen independently of the rest of the CP roadmap.
Phases C–E require step 1 (dcr_write through CP ring) to be working on
xrt/opae, OR the dispatcher's CP path to be exercised end-to-end on
simx/rtlsim first (whichever lands first establishes the contract).

---

## 5. Verification plan

### 5.1 Standalone CP unit test (Phase A)

`hw/unittest/cp_sim/` — drives the `CommandProcessor` directly:
- CMD_NOP retires
- CMD_DCR_WRITE invokes `vortex_dcr_write` hook with correct addr/value
- CMD_LAUNCH pulses `vortex_start` exactly once, waits for `!vortex_busy`
- CMD_MEM_WRITE / CMD_MEM_READ exercise DMA path via `dram_read/write`
- Sequence of N back-to-back commands retires in order, seqnum increments correctly
- Q_SEQNUM matches retire count

### 5.2 Per-phase regression

Each phase keeps the **8-corner regression** as exit criterion:
legacy + CP × sgemm + vecadd × XRT + OPAE. Plus simx and rtlsim
must pass legacy OpenCL throughout, and v2 regression tests after
Phase B (when their CP path is wired).

### 5.3 Exit criterion (after Phase E)

- All 4 backends (simx, rtlsim, xrt, opae) run sgemm + vecadd
  through the **same** v2 dispatcher code path
- `callbacks_t` has no `launch_*` / `dcr_*` fields
- No grep for `dcr_write` / `launch_start` outside of CP-internal code
- `VORTEX_USE_CP=0` (transparent mode) and `VORTEX_USE_CP=1` (full FSM
  mode) both produce correct results on simx/rtlsim; mode toggles only
  affect timing/observability, not correctness

---

## 6. Open questions

1. **`CommandProcessor` accuracy vs. speed.** The hardware CP is a
   cycle-accurate Verilog FSM. The C++ model is functional. How close
   do they need to match? My read: close enough that the regression
   tests produce identical results, not cycle-by-cycle identical.
   Performance counters from simx CP mode will be approximate.
2. **NO-CP transparent mode semantics for DMA commands.** `CMD_MEM_WRITE`
   etc. issued in transparent mode would copy via the host (not via
   simulated AXI). Probably fine — they're for host↔device DMA, which
   in simx/rtlsim is already a direct memory copy.
3. **Address-of-CP-MMIO contract.** Currently xrt/opae put the CP
   regfile at host byte offset `0x1000` (bit-12 split). simx/rtlsim
   have no host bus — they receive an `offset` from `0` directly.
   `cp_mmio_write(off=0x100, val=...)` should mean the same thing on
   all backends (CP-internal offset). xrt/opae wrappers add `0x1000`
   on their side.
4. **Per-cycle tick cost in simx.** simx already runs slow on big
   tests; adding a `tick()` to the inner loop could regress speed.
   Mitigation: the CP FSM is a handful of branches per tick; should
   be < 1% overhead. Measure during Phase B.
5. **`VORTEX_USE_CP` default off vs. on long-term.** User asked for
   off by default during bring-up. End-state: on by default everywhere,
   then the env var goes away entirely (CP is the only path).

---

## 7. Sequencing notes

This proposal **doesn't** depend on step 1 (CP DCR writes through the
ring on xrt/opae) working first — Phase A and B can land independently
and even help diagnose step 1's hang by giving us a functional reference
implementation to compare against.

After Phase B lands, the v2 regression test failures (segfault on simx,
misaligned access on rtlsim/xrt/opae) become tractable: we have one
control-plane code path to debug instead of four divergent ones.

Total estimated effort: **~5 substantial commits** (one per phase),
2–4 hours each.

---

# Addendum (May 2026) — Minimal 6-function transport `callbacks_t`

**Status:** supersedes the §1 "pure v2" target. Implemented alongside the
Track-1 CP host-memory ring work.

## A.1 Principle

Once the Command Processor is the **sole** command + DMA engine, the
backend driver does no memory management, no DMA, and no capability
decoding. Those are either the CP's job (it has a DMA engine and a
regfile) or generic host-side code (an address allocator). `callbacks_t`
collapses to a pure **transport HAL**: a device handle, a register
channel to the CP, and CP-visible host memory — nothing else.

## A.2 The interface

```c
typedef struct {
  int (*dev_open) (void** out_dev_ctx);
  int (*dev_close)(void*  dev_ctx);

  // CP register channel — doorbell + status + caps window.
  int (*cp_reg_read) (void* dev_ctx, uint32_t off, uint32_t* out_value);
  int (*cp_reg_write)(void* dev_ctx, uint32_t off, uint32_t  value);

  // CP-visible host memory — command ring + DMA staging.
  // Returns BOTH a host pointer (the runtime memcpy's through it) and the
  // device-side address the CP's m_axi_host master uses to reach it.
  int (*host_mem_alloc)(void* dev_ctx, uint64_t size,
                        void** out_host_ptr, uint64_t* out_cp_addr);
  int (*host_mem_free) (void* dev_ctx, uint64_t cp_addr);
} callbacks_t;
```

13 callbacks → **6**. Three pairs: lifecycle, register channel, host memory.

## A.3 What moves, and where

| Removed callback | Fate |
|---|---|
| `mem_upload`, `mem_download` | gone — runtime `memcpy`s through `host_mem_alloc`'s `host_ptr`; the CP moves bytes via `CMD_MEM_*` |
| `mem_copy` | gone — was already dead (`dev_copy`→`cp_submit_mem_copy`) |
| `mem_alloc`, `mem_reserve`, `mem_free` | → **common core**: `vx::Device` owns one `MemoryAllocator` for device memory (the free-list was copy-pasted into all 5 backends) |
| `mem_access` | gone — a no-op on every backend |
| `memory_info` | → **common core** — answered from the allocator |
| `query_caps` | → **common core** — `cp_reg_read` of the CP caps window + `vortex::load_caps`/`decode_caps` (already shared helpers) + `VX_CFG_*` constants for `CACHE_LINE_SIZE`/`CLOCK_RATE`/`PEAK_MEM_BW` |
| `has_native_dma` | gone — every backend routes device transfers through the CP, identically |
| `cp_mmio_read/write` | kept, renamed `cp_reg_read/write` |

The coherence contract: host memory from `host_mem_alloc` must be
coherent with the CP's `m_axi_host` view (true for XRT host-only BOs,
OPAE CCI-P, and the unified-memory sims). No explicit sync callback.

## A.4 CP hardware queues

The CP RTL (`hw/rtl/cp/VX_cp_axil_regfile.sv`) instantiates
`VX_CP_NUM_QUEUES` independent hardware queues, each with its own 64-byte
register window at `0x100 + qid*0x40` (`RING_BASE`/`HEAD`/`CMPL`/`TAIL`/
`SEQNUM`/`CONTROL`/`ERROR`). The count is **runtime-queryable** from
`CP_DEV_CAPS` (offset `0x008`, bits `[7:0]`).

Multi-queue needs **zero extra callbacks** — a queue is just a different
register offset. The common core reads `NUM_QUEUES` and maps software
`vx::Queue`s onto CP HW queues through the same `cp_reg_*` channel. This
is also the proper fix for the serial-ring COUT-drain limitation: COUT
draining runs on a separate HW queue, concurrently with a launch.

## A.5 Sim backends (simx / rtlsim / gem5)

The sims have no host/device split — one unified memory. `host_mem_alloc`
returns a real `malloc`'d host buffer; `cp_addr` is the pointer value
itself. The software `CommandProcessor`'s `dram_read`/`dram_write` hooks
route by registered host region: an address inside a host region hits the
`malloc`'d buffer, otherwise the sim RAM. The CP thus moves bytes between
"host" and "device" identically to hardware.

## A.6 NVIDIA/AMD alignment

The 6 calls are the per-platform ring/register/pinned-memory hooks —
the same seam `amdgpu` draws between its per-ASIC backend and its common
core. The allocator, ring protocol, and queue/event layer live in the
common core, matching the `amdgpu`/TTM shared core. `host_mem_alloc` is
precisely a real driver's pinned-memory allocator for the ring.

---

# Addendum (May 2026) — Virtual Memory: an MMU-aware CP DMA

**Status:** design. Supersedes the host-side `page_table_walk` model — which
the `callbacks_t` collapse removed when it moved `mem_alloc` + DMA into the
common core and the CP.

## VM.1 What was wrong with the old model

Vortex VM today: the host runtime builds page tables in device memory, and
`upload`/`download` did a **software `page_table_walk` on the host** to turn a
VA into a PA before DMA-ing. That only worked because the host had direct
access to the simulator's RAM. It is wrong twice over:

- It does not match real GPUs — on NVIDIA (GMMU + Copy Engines) and AMD
  (GPUVM + SDMA) the copy/DMA engines are **MMU-aware**; the driver never
  translates addresses on their behalf.
- It does not fit the CP-sole-DMA architecture: the CP owns every transfer,
  and it cannot call back into a host software walker.

## VM.2 Principle — the copy engine is MMU-aware

On modern GPUs: the **driver builds page tables and never walks them at run
time**; **every hardware engine — compute units *and* copy/DMA engines —
translates in hardware** through those same page tables; software works
purely in **virtual addresses**. One address space, hardware translation.

Vortex's CP already has a DMA engine (`VX_cp_dma`) — it *is* Vortex's copy
engine. The correct design makes it MMU-aware, exactly like SDMA / a CE.

## VM.3 Design

```
  host driver (vx::Device + VMManager)      the CP / its DMA engine
  ──────────────────────────────────       ───────────────────────────
  mem_alloc:  PA <- global_mem_             CMD_MEM_* device operand
              VA <- phy_to_virt_map(PA)       = a VIRTUAL address
  builds PTEs in a host shadow              VX_cp_dma walks the page table
  flushes dirty PT pages via                 (HW walker + small TLB) -> PA,
  CMD_MEM_WRITE(physical)                     then bursts AXI
  runtime API is VA-only — no walks         shares the kernel's PTs + SATP
```

1. **Host driver (common core).** `vx::Device` owns the `VMManager` page-table
   builder. `mem_alloc` allocates a PA from `global_mem_`, installs PTEs,
   returns a **VA**; `mem_free` unmaps. The runtime API is VA-only — the
   driver's job and *only* the driver's job, like `amdgpu`'s GPUVM manager.
2. **Page-table writes.** `VMManager` keeps a host shadow of the page tables
   (already its design) and flushes dirty PT pages with
   **`CMD_MEM_WRITE(physical)`** — one bulk transfer per dirty PT page.
3. **CP DMA = MMU-aware copy engine.** The device-side operand of every
   `CMD_MEM_*` is a **VA**; `VX_cp_dma` translates it via a hardware
   page-table walker + small TLB, using the kernel's page tables. The host
   never translates.
4. **`physical` command flag.** `CMD_MEM_*` carries a `physical` bit in its
   header; set → the CP DMA skips translation. Used for PT bootstrap (writing
   the first PT pages before the table exists) and the PT region itself.
5. **SATP to the CP.** The host programs the page-table root + mode into a CP
   regfile register (`CP_SATP_LO/HI`) at VM init so the CP DMA walker knows
   where the table is. Compute cores still receive SATP via the kernel's boot
   `csrw` — unchanged.
6. **Compute cores — unchanged.** Their per-core MMU walks the same tables.
7. **Simulators.** The shared C++ `CommandProcessor` model gains the same
   MMU-aware behavior — a page-table walk in its device-memory path — so
   simx / rtlsim / gem5 match FPGA semantics. (This models *the CP* walking,
   mirroring hardware — not a host-side shortcut.)

## VM.4 Why it is correct and efficient

- **Correct:** one page table, hardware-walked by every engine; the host is
  translation-free — the AMD SDMA / NVIDIA CE model verbatim.
- **No `callbacks_t` impact:** VM lives entirely in the common core (PT
  builder) and the CP (translation); the 6-function transport HAL is
  untouched — a backend never sees VM.
- **Efficient:** host-shadow + batched flush (one DMA per dirty PT page, not
  per PTE); the CP-DMA TLB amortizes walks across the 4 KB burst chunks
  `VX_cp_dma` already uses; `install_identity_map` already emits **megapage**
  PTEs where alignment permits (fewer walk levels, less TLB pressure);
  VA-only host API → zero host-side translation cost, no host/HW page-table
  coherency race.

## VM.5 Address-space map

| Region | In device VA space? | CP DMA access |
|---|---|---|
| Command ring / DMA staging | no — host memory (`m_axi_host`) | untranslated |
| Page-table region | physical | `CMD_MEM_*(physical)` |
| IO / COUT `[0, USER_BASE)` | identity-mapped | translate → self |
| Kernel image (`mem_reserve`) | identity-mapped | translate → self |
| User buffers (`mem_alloc`) | virtual | translate |

## VM.6 Phasing

- **Phase 1 — simulators, no RTL.** Make the `CommandProcessor` C++ model
  MMU-aware (walker in its device-memory path) + add the `physical` flag and
  `CP_SATP` to the model; `vx::Device` owns the `VMManager`. → VM correct on
  simx / rtlsim / gem5; validates the whole runtime + protocol end to end.
- **Phase 2 — RTL.** `VX_cp_dma` gains a hardware page-table walker + TLB;
  `VX_cp_axil_regfile` gains `CP_SATP`; the `CMD_MEM_*` decoder honors the
  `physical` flag. → VM correct on xrt / opae.

Validate each phase with sgemm + tex on the respective backends.

## VM.7 — Phase 1 implementation checklist (file-by-file)

**`sim/common/cmd_processor.{h,cpp}` — make the CP model MMU-aware**
- `#include <vm_types.h>` (guarded by `VX_CFG_VM_ENABLE`).
- New CP-internal regs `CP_SATP_LO = 0x028`, `CP_SATP_HI = 0x02C`; `mmio_write`
  stores them into a `uint64_t satp_` member.
- New private `uint64_t cp_translate(uint64_t va) const`: with VM disabled or
  `satp_` mode `BARE`, return `va`; else do an Sv32/Sv39 walk using
  `SATP_t`/`vAddr_t`/`PTE_t` and `hooks_.dram_read` to read PTEs — megapage-aware
  (a leaf PTE at level > 0 contributes the low VA bits).
- In the `CMD_MEM_*` handler: read `physical = cur_cmd_.flags & CP_MEM_FLAG_PHYSICAL`.
  Translate the **device** operand(s) once at the base, before the chunk loop —
  `MEM_WRITE`→`arg0`, `MEM_READ`→`arg1`, `MEM_COPY`→both; skip when `physical`.
  (A buffer is one contiguous PA allocation, so a single base translate covers
  the whole transfer.)

**`sw/runtime/common/{vortex2_internal.h,device.cpp}` — host driver owns VM**
- `Device` owns `std::unique_ptr<vortex::VMManager> vm_mgr_` and a nested
  `CpMemIO : vortex::DeviceMemIO` whose `read`/`write` call `cp_submit_mem_*`
  with the `physical` flag set (page-table I/O is PA-direct).
- `cp_submit_mem_` gains a `bool physical` param → sets `CP_MEM_FLAG_PHYSICAL`
  in the CL header `flags` byte. `dev_write/read/copy` pass `physical=false`
  (the CP translates); `CpMemIO` passes `physical=true`.
- `cp_init`: after `cp_enabled_`, if VM → construct `VMManager`, `init()`, then
  program `CP_SATP_LO/HI` via `cp_reg_write` from the SATP value.
- `mem_alloc` (VM, non-`VX_MEM_HOST`): `global_mem_.allocate()`→PA;
  `VX_MEM_PHYS`→`install_identity_map` (keep PA); else `phy_to_virt_map`→VA.
- `mem_free` (VM): `page_table_walk` VA→PA, then `global_mem_.release(PA)`.
- `mem_reserve` (VM): `global_mem_.reserve` + `install_identity_map`.
- `dev_write/read/copy`: **no** host-side translation — pass VAs straight to
  `cp_submit_mem_*`; the CP DMA translates.

**`sw/runtime/{simx,rtlsim}/vortex.cpp` — drop the dead VM stubs**
- Delete the `#ifdef VX_CFG_VM_ENABLE` blocks (`RamMemIO`, `vm_mgr_`, `dev_io_`,
  the `init()` VM call) — VM is wholly common-core now.

**Build + validate:** a `VX_CFG_VM_ENABLE=true` build (`VX_config.toml`), then
sgemm + tex on simx and rtlsim. The `CP_MEM_FLAG_PHYSICAL` bit + the SATP regs
are also what Phase 2's `VX_cp_dma` RTL walker will consume — identical wire
protocol.
