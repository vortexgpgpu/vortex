# CP v3 Critical Review

Status: review findings, action items
Branch: `feature_cp`
Related:
- [command_processor_proposal.md](command_processor_proposal.md)
- [cp_rtl_impl_proposal.md](cp_rtl_impl_proposal.md)
- [cp_runtime_impl_proposal.md](cp_runtime_impl_proposal.md)
- [cp_pure_v2_callbacks_proposal.md](cp_pure_v2_callbacks_proposal.md)
- [cp_xrt_integration_plan.md](cp_xrt_integration_plan.md)

## 1. Scope

A critical, line-level review of the v3 Command Processor stack as it stands
today. Scope:

- Public API: [sw/runtime/include/vortex2.h](../../sw/runtime/include/vortex2.h)
- Common dispatcher: [sw/runtime/common/](../../sw/runtime/common/) (device,
  queue, event, buffer, module, vm, legacy wrappers, callbacks contract)
- XRT runtime backend: [sw/runtime/xrt/](../../sw/runtime/xrt/)
- XRT AFU integration: [hw/rtl/afu/xrt/](../../hw/rtl/afu/xrt/)
- CP RTL: [hw/rtl/cp/](../../hw/rtl/cp/) (all `VX_cp_*.sv`)
- C++ functional twin: [sim/common/cmd_processor.{h,cpp}](../../sim/common/)

About 11 KLoC of C++, SystemVerilog, and packaging TCL.

Each finding cites file and line on both sides where applicable. Findings are
grouped by severity, not by layer, so a reader can triage from the top.

This document does *not* propose an architectural redesign; it is a punch list
of concrete defects and short, scoped improvements. Items that imply a larger
rework are flagged as "Architectural" and pointer to the natural follow-up.

## 2. Bottom line

The shape is right. A doorbell-driven host-resident command ring driving a
small in-order CP that owns DMA, dispatch, cache flush, and event slots is the
same blueprint as AMD HSA / NVIDIA host channels. The runtime API surface
([vortex2.h](../../sw/runtime/include/vortex2.h)) is clean and the layering
(transport HAL → common dispatcher → CP) is well held. The legacy v1 wrapper
is a thin translation, not a parallel implementation.

But four things must land before this can carry real workloads:

1. **Use-after-free of `Buffer`/`Module` in queue work lambdas** (R-1) — hits
   the first time a user writes OpenCL-shaped code.
2. **`VX_cp_completion` silently drops simultaneous retires** (C-1) — the host
   hangs forever the moment multi-queue becomes real.
3. **XRT backend will crash under sustained load** — the unguarded
   `host_bos_` map (X-2) plus the uncaught XRT exceptions (X-1) plus the
   missing memory barrier / `bo::sync` (R-2 + deployment dependency).
4. **COUT producer-consumer deadlock** (O-1) — any kernel that prints more
   than the ring size per hart in a single launch hangs the host forever, on
   stock XRT, with no watchdog. Companion: a SCOPE serial-bus desync (O-2)
   wedges the entire runtime via an AXI-Lite read with no timeout.

Separately, the C++ functional twin in `sim/common/cmd_processor.cpp` has
**four wire-protocol divergences from the RTL** (P-W1, P-W3, P-W4, P-S10) that
mean a SimX pass is not evidence of an RTL pass.

And three architectural smells (A-1 to A-3) put a hard ceiling on throughput
regardless of how fast the CP runs; they should be settled before scaling
matters at all.

## 3. Architectural smells

These are not bugs — they are design choices that prevent the stack from
scaling. Listing first because they reframe the bug-fix priorities below.

### A-1. "Multi-queue" is a software fiction

The API advertises multiple `vx_queue_h` ([vortex2.h:319-326](../../sw/runtime/include/vortex2.h#L319)),
each with its own worker thread ([queue.cpp:28](../../sw/runtime/common/queue.cpp#L28)),
but every command from every queue serializes behind two mutexes:
- `Device::cp_mu_` ([device.cpp:316](../../sw/runtime/common/device.cpp#L316))
- per-Queue `Queue::enqueue_mu_` ([queue.cpp:605](../../sw/runtime/common/queue.cpp#L605))

Worse, **launch state is global**. A launch programs ~12 `VX_DCR_KMU_*`
registers via 12 separate `CMD_DCR_WRITE`s before `CMD_LAUNCH`
([queue.cpp:329-360](../../sw/runtime/common/queue.cpp#L329)). Two queues
cannot interleave at all: Q1's DCR sequence would corrupt Q0's mid-launch.
`enqueue_mu_` is not a perf miss; it is a correctness requirement of the
launch encoding.

The CP RTL is multi-queue-shaped (`NUM_QUEUES`, per-Q ring/seqnum, four
arbiters), but the engine collapses to one-command-at-a-time per CPE and a
single DCR proxy ([VX_cp_dcr_proxy.sv](../../hw/rtl/cp/VX_cp_dcr_proxy.sv)).
It implements concurrency mechanics that the launch encoding precludes.

**Fix direction:** model launches as a self-contained QMD-style descriptor
carried inline in the 64-byte `CMD_LAUNCH` (PC + grid[3] + block[3] + lmem
size + args pointer + flags fits comfortably). Drop the DCR-poking dance
entirely. The CP delivers the descriptor to KMU atomically. Only then is
per-queue parallelism real.

### A-2. Spin loops at every level

- **Runtime**: busy poll on `Q_SEQNUM` over AXI-Lite
  ([device.cpp:342-363](../../sw/runtime/common/device.cpp#L342)).
- **CP RTL**: `CMD_EVENT_WAIT` is a hardware busy poll
  ([VX_cp_event_unit.sv:120-125](../../hw/rtl/cp/VX_cp_event_unit.sv#L120))
  that continuously rereads the host counter slot over `m_axi_host` — forever.
- **AFU**: `interrupt` is a 1-cycle pulse, no IP_ISR latch
  ([VX_afu_wrap.sv:337-339](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L337)).
- **XRT backend**: no `xrt::ip::interrupt` is constructed.

Composed: a 100 ms host event blocks the runtime AND saturates PCIe with the
CP's polls AND burns a CPU on `xrt::ip::read_register`.

**Fix direction:** (a) level-sensitive interrupt with IP_ISR/IER/GIER decode
in [VX_afu_ctrl.sv](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv) (the file was
deliberately stripped of these; restore them);
(b) wire `xrt::ip::interrupt` in the XRT backend;
(c) in `CMD_EVENT_WAIT`, exponential-backoff host reads;
(d) publish per-queue seqnum into a host-coherent mailbox so the runtime polls
host memory instead of MMIO.

### A-3. Per-transfer host allocation on the DMA path

`Device::cp_submit_mem_write`/`cp_submit_mem_read`
([device.cpp:490-522](../../sw/runtime/common/device.cpp#L490)) allocates a
fresh CP-visible staging buffer, memcpys, posts `CMD_MEM_*`, then frees —
every single transfer. On XRT that is an `xrt::bo(HOST_ONLY)` ioctl + mmap +
munmap per transfer.

Downstream costs:
- `Queue::enqueue_fill_buffer` issues this 256× for a 16 MB fill
  ([queue.cpp:596](../../sw/runtime/common/queue.cpp#L596)).
- `Module::load_bytes` issues two per `.vxbin` load.
- Kernel-arg upload issues one per `vx_enqueue_launch`.

**Fix direction:** a sized staging-buffer pool. The kernel-args slot pool
([device.cpp:104-134](../../sw/runtime/common/device.cpp#L104)) is the right
model — generalize it across all transfer sizes. Also: skip staging entirely
for `CMD_MEM_WRITE` when the source is already in CP-visible host memory (a
Buffer with `VX_MEM_HOST`).

## 4. Correctness bugs — block production

### 4.1 Runtime dispatcher

**R-1. Use-after-free of `Buffer`/`Module` in worker lambdas.**
Every `Queue::enqueue_*` captures `Buffer*` / `Kernel*` as raw pointers
([queue.cpp:187](../../sw/runtime/common/queue.cpp#L187),
[queue.cpp:207](../../sw/runtime/common/queue.cpp#L207),
[queue.cpp:228](../../sw/runtime/common/queue.cpp#L228),
[queue.cpp:281](../../sw/runtime/common/queue.cpp#L281)) with no `retain()`.
The work runs on the worker thread after waits resolve — possibly long after
the caller called `vx_buffer_release`. Wait events ARE retained at
[queue.cpp:120-126](../../sw/runtime/common/queue.cpp#L120); apply the same
pattern to every captured Buffer / Module / Kernel.
**Fix:** retain at enqueue, release inside the work lambda.

**R-2. No memory barrier between ring write and doorbell.**
`Device::cp_submit_cl_` does `memcpy(ring, cl, 64)` then `cp_reg_write(Q_TAIL_*)`
([device.cpp:323-334](../../sw/runtime/common/device.cpp#L323)) with no
`_mm_sfence` or `std::atomic_thread_fence(memory_order_release)`. On any
platform where the host BO is mapped WB, the CP can read stale ring bytes.
The XRT backend amplifies this by also lacking `bo::sync(TO_DEVICE)`.
**Fix:** insert a release fence before every `Q_TAIL` write; document the
coherence contract in [callbacks.h](../../sw/runtime/common/callbacks.h)
explicitly.

**R-3. `~Device` frees ring/head/cmpl host buffers without quiescing the CP.**
[device.cpp:80-100](../../sw/runtime/common/device.cpp#L80) frees the host
memory the CP could still be DMAing. **Fix:** write `CP_REG_CTRL=0` and
`CP_Q_CONTROL=0`, poll for quiescent, then free.

**R-4. `cp_submit_dcr_read` races on `Q_LAST_DCR_RSP`.**
[device.cpp:427-431](../../sw/runtime/common/device.cpp#L427): after
`cp_submit_cl_` releases `cp_mu_` and returns, another submitter can post
their own `CMD_DCR_READ` before the first caller reads `LAST_DCR_RSP`. The
RTL has the same hole — `last_dcr_rsp` is a single global register
([VX_cp_axil_regfile.sv:73](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L73)).
**Fix (runtime):** hold `cp_mu_` across the SEQNUM wait + LAST_DCR_RSP read.
**Fix (RTL, longer term):** make DCR responses per-queue or carry them inline
in the completion record.

**R-5. `Event::signal` device-write race.**
[event.cpp:44-58](../../sw/runtime/common/event.cpp#L44) drops `mu_` before
calling `dev_write` on the mirror slot. Two concurrent signalers with
different `value`s can reorder so the device slot ends up below `counter_`.
**Fix:** do the device write under `mu_`, or only the writer that committed
the highest counter performs the DMA.

**R-6. `Module::get_kernel` raw-pointer cache race.**
[module.cpp:170-191](../../sw/runtime/common/module.cpp#L170) caches `Kernel*`
without retaining. Window between `refs_.fetch_sub(1)==1` in `release()` and
`~Kernel` acquiring `kcache_mu_` is wide enough for another `get_kernel` to
retain a corpse. **Fix:** atomic "weak retain" — cache holds owning refs and
calls `release()` itself; or rebuild around `std::shared_ptr<Kernel>`.

**R-7. `legacy_remember_last_event` self-deadlock.**
[device.cpp:774-778](../../sw/runtime/common/device.cpp#L774) holds
`Device::mu_`, calls `Event::release()`; if refcount hits 0, `~Event` calls
`Device::mem_free` which takes the same non-recursive `mu_`.
**Fix:** release outside the lock.

**R-8. `drain_cout` stack-overflow risk.**
[device.cpp:701](../../sw/runtime/common/device.cpp#L701) declares
`char data[RING]` on the stack with no bound check; `VX_MEM_IO_COUT_RING` is
config-dependent. **Fix:** heap-allocate or cap.

**R-9. `vx_event_wait_values` doesn't share a deadline across events.**
[event.cpp:171-176](../../sw/runtime/common/event.cpp#L171) passes the same
`timeout_ns` to each iteration, so total wait can be `n × timeout_ns`.
**Fix:** compute one absolute deadline; subtract elapsed each iteration.

**R-10. C++ exceptions can propagate across `extern "C"` boundaries.**
`Queue::create` calls `new Queue(...)` which may throw `std::system_error`
from `std::thread`. `vx_queue_create` does not catch. UB at the ABI surface.
**Fix:** `try`/`catch` in every `extern "C"` entry that constructs C++ state.

### 4.2 XRT runtime backend

**X-1. Uncaught XRT exceptions across the C ABI.**
`xrt::device(idx)`, `xrt::ip(...)`, and every `write_register`/`read_register`
call can throw `xrt_core::system_error`. None caught. Any missing xclbin or
AXI-Lite timeout aborts the host. **Fix:** wrap every entry in `try`/`catch`
returning a non-zero code.

**X-2. `host_bos_` map is not mutex-protected.**
Concurrent `host_mem_alloc`/`host_mem_free` from multiple queue workers will
corrupt the `std::map`. Likely the most reproducible crash under sustained
load. **Fix:** `std::mutex host_bos_mu_` around `emplace`/`find`/`erase`.

**X-3. `host_only` BO with hardcoded `group_id=0`.**
Wrong on shells where the host-only bank index is not 0. The canonical lookup
needs `xrt::kernel::group_id()`, but the backend uses `xrt::ip` which doesn't
expose it. Silent allocation in the wrong bank → CP reads garbage.
**Fix:** either keep a stub `xrt::kernel` for metadata or accept the
constraint and assert/document it.

**X-4. `host_only` flag silently downgraded on unsupported shells.**
No `bo.flags()` post-construction check.

**X-5. `xrt.ini.in` is a dead template.**
No substitution rule in the Makefile; the file is never produced or installed.
Test makefiles reference `xrt.ini` which therefore does not exist at runtime.

### 4.3 AFU integration

**A-A1. `m_axi_host_awsize/awburst/arsize/arburst` are never declared or driven.**
The CP's intended 64 B/beat bursts go out with inferred-constant sideband;
SmartConnect tolerates it but performance and (worst case) correctness depend
on inference defaults. Same issue on the bank-0 arbiter `VX_axi_arb2` which
has no sideband ports at all.
[VX_afu_wrap.sv:305-329](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L305) and
[VX_afu_wrap.sv:332-335](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L332).
**Single highest-impact AFU bug.**

**A-A2. Vitis `ap_ctrl-hs` contract is half-implemented.**
`ap_done`/`ap_ready` are always 0; `ap_start` is silently dropped;
`interrupt` is a one-cycle pulse with no IP_ISR/IER/GIER block — yet
`package_kernel.tcl` still advertises those registers and exports the kernel
as standard `ap_ctrl_hs`. Any host using stock `xrt::kernel`/`xrt::run` hangs.
**Fix:** either restore the IP_ISR block and drive `ap_done` from CP
completion, or repackage as user-managed (`ap_ctrl_none`) and update the host
contract document.

**A-A3. AW/W routing race in the CTRL bus split.**
[VX_afu_wrap.sv:191-203](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L191): the W mux
uses `route_cp_w_r` latched at AW handshake, but AXI-Lite permits W before AW.
If the host ever sends W first, the W beat routes to the *previous*
transaction's slave. Works today only because XRT happens to send AW first.

**A-A4. CP regfile address truncation to 12 bits.**
[VX_afu_wrap.sv:194](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L194) and
[VX_afu_wrap.sv:213](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L213). With
`VX_CP_NUM_QUEUES > 16` (per-queue block stride 0x40), regfile aliases
`CP_CTRL` and corrupts global state. Today's `NUM_QUEUES=1` is safe; this is
a tripwire.

### 4.4 CP RTL

**C-1. `VX_cp_completion` silently drops simultaneous retires AND on FIFO full.**
[VX_cp_completion.sv:64-99](../../hw/rtl/cp/VX_cp_completion.sv#L64): when two
queues retire on the same cycle, only the lowest-QID is enqueued; the higher
QID's seqnum is lost because the engine's `S_RETIRE` is one-cycle. **The host
then polls the completion slot forever.** This is the most dangerous silent
failure in the entire stack.
**Fix:** per-source one-shot FIFO with ready/ack back to the engine; engine
holds `retire_evt` until completion accepts it.

**C-2. `VX_cp_dma` writes past buffer end on non-CL-aligned sizes.**
[VX_cp_dma.sv:121](../../hw/rtl/cp/VX_cp_dma.sv#L121) rounds size up to 64 B;
a 65-byte copy writes 128 bytes. `wstrb` is always all-ones.
**Fix:** compute tail `wstrb` on the last beat, OR reject unaligned sizes at
the engine.

**C-3. `VX_cp_dcr_proxy` ignores `gpu_if.dcr_req_ready` backpressure.**
[VX_cp_dcr_proxy.sv:129](../../hw/rtl/cp/VX_cp_dcr_proxy.sv#L129).
Transitions out of `S_REQ` are unconditional; if Vortex ever deasserts
`dcr_req_ready` (it does in some configs), the write is lost.
**Fix:** gate transition on `dcr_req_ready`.

**C-4. `CMD_EVENT_WAIT` holds the EVENT arbiter grant for its entire wait.**
[VX_cp_event_unit.sv:41-43](../../hw/rtl/cp/VX_cp_event_unit.sv#L41)
and L120-125. The "round-robin across queues" comment is wrong — the arbiter
never rotates off a held grant. A stalled WAIT on Q0 starves all other
queues' EVENT commands indefinitely. This is also the deadlock that motivated
the `cp_mu_` release dance in [device.cpp:307-336](../../sw/runtime/common/device.cpp#L307).
**Fix:** release the arbiter grant between poll attempts; re-bid each cycle.

**C-5. `VX_cp_axil_regfile`: first AXI-Lite read after reset returns stale data.**
[VX_cp_axil_regfile.sv:354-377](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L354).
One-cycle-off: `read_reg(rd_addr_buf)` samples the pre-update value.
**Fix:** add a register-stage between AR-fire and R-data.

**C-6. `r_ring_size_log2` writes not range-checked.**
[VX_cp_axil_regfile.sv:386-387](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L386).
Host can program any 8-bit value; mask is silently truncated, producing
aliased ring addresses. Writing 0 yields mask=0 → fetch reads the same CL
forever.
**Fix:** range-check the write; clamp to `[6, RING_SIZE_LOG2_MAX]`.

**C-7. `CMD_FENCE` is a no-op.**
[VX_cp_engine.sv:105-108](../../hw/rtl/cp/VX_cp_engine.sv#L105) classifies it
under the default skip case; `arg0`'s fence mask is ignored. Single-queue
ordering is preserved by FIFO; cross-queue ordering between Q0's DMA and Q1's
LAUNCH on the same memory has no fence primitive at all.
**Fix:** implement `FENCE_DMA_BIT`/`FENCE_GPU_BIT` honoring against
in-flight resources.

**C-8. `cp_busy` lies during in-flight DMA.**
[VX_cp_core.sv:430-438](../../hw/rtl/cp/VX_cp_core.sv#L430) reads `*_grant`,
which deasserts mid-transfer. Host status poll says "idle" while DMA is still
running. **Fix:** OR each resource module's per-resource busy signal.

### 4.5 Observability paths — COUT and SCOPE

The print and trace paths are debug surfaces, but both can hang production
runs on XRT today. Listed together because the failure mode in each case is a
host-side wait that the rest of the stack cannot break out of.

**O-1. COUT can hard-deadlock the host on any long-printing launch.**

Full chain, kernel → host:

1. Kernel hart writes more than `VX_MEM_IO_COUT_RING` bytes (default 512)
   into its slot during a single launch.
2. Producer hits the back-pressure spin at
   [vx_print.S:41-43](../../sw/kernel/src/vx_print.S#L41):
   `bgeu a2, RING, 1b`. The hart spins on `rd[slot]`.
3. The spinning hart's warp stays active → `vx_busy` stays high.
4. CP launch FSM sits in `S_WAIT_DRAIN`
   ([VX_cp_launch.sv](../../hw/rtl/cp/VX_cp_launch.sv)). `CMD_LAUNCH` never
   retires.
5. Host loops forever in the `Q_SEQNUM` poll inside
   [cp_submit_cl_](../../sw/runtime/common/device.cpp#L342) — no timeout, no
   watchdog.
6. `drain_cout` is only ever called *after* `cp_submit_launch` returns
   ([device.cpp:393](../../sw/runtime/common/device.cpp#L393)). It never runs.
7. `rd[slot]` is never advanced → step 2 spins forever.

Threshold: 512 B per hart per launch (pre-newline), reduced when multiple
harts collide on `hartid & 63` and share a slot. The comment block at
[device.cpp:351-356](../../sw/runtime/common/device.cpp#L351) explicitly
acknowledges "A kernel that overruns its lossless COUT ring within one launch
therefore back-pressures until the launch ends" — but if the back-pressuring
kernel *is* what's keeping the launch alive, "until the launch ends" is
"forever".

**Industry-aligned fix: lossy ring + overflow indication, matching CUDA / HIP
`printf`.**

NVIDIA's `printf` (FIFO size set by `cudaDeviceSetLimit(cudaLimitPrintfFifoSize)`,
default 1 MiB) and AMD HIP's `printf` both use a fixed-size circular buffer
in device memory; the kernel-side write tries once and **silently drops** the
record when the buffer is full, optionally setting an overflow flag the host
surfaces as "Buffer overflow, output may be truncated". The kernel **never**
blocks. This kills the producer/consumer dependency at the root.

Concretely for Vortex:

- Replace the spin at [vx_print.S:41-43](../../sw/kernel/src/vx_print.S#L41)
  with a conditional store: if `wr - rd >= RING`, skip the byte and atomically
  bump a per-slot `lost_bytes[slot]` counter (one extra word in the COUT
  region; the host prints "[slot N: lost K bytes]" alongside drained
  output).
- This is a ~30-line kernel + ~10-line runtime change. No CP, AFU, or RTL
  changes. Eliminates the deadlock unconditionally; ring size becomes a
  tuning knob, not a correctness boundary.

A complementary structural fix — moving the COUT drain off the post-launch
path so the host drains *during* a long-running kernel — costs more (needs
either a second CP queue or a sideband DMA path the runtime can poll without
contending for the launch's CP ring). The two fixes compose: the lossy ring
makes the bound a soft signal; the concurrent drain makes the bound rarely
relevant in practice.

**O-2. SCOPE drain can wedge the entire runtime on serial-bus desync.**

Mechanism:

1. AFU read-side gates the AXI-Lite response on
   `rvalid_stall = is_scope_raddr && ~scope_rdata_valid`
   ([VX_afu_ctrl.sv:182](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv#L182)).
2. If the bit-serial bus ever desyncs — reset between writing SCP_0 and
   SCP_1, a tap that fails to respond, a switch ring that drops the command —
   `scope_rdata_valid` never asserts.
3. The host AXI-Lite read in `read_reg`
   ([scope.cpp:232-236](../../sw/runtime/common/scope.cpp#L232)) blocks
   indefinitely. `xrt::ip::read_register` has no internal timeout.
4. `vx_scope_drain` runs inside the CP `Q_SEQNUM` poll
   ([device.cpp:357-361](../../sw/runtime/common/device.cpp#L357)). The
   wedged read halts the entire poll loop.
5. Even if the kernel and CP completed cleanly, the host never observes the
   retire; `cp_submit_cl_` never returns; the post-launch `drain_cout` never
   runs either. The whole runtime appears wedged.

Defensive code already in place doesn't cover the failure:
- `SCOPE_MAX_OCCUPANCY` at [scope.cpp:251-255](../../sw/runtime/common/scope.cpp#L251)
  catches a bogus *count* (already-decoded data), not a hung *read*.
- The auto-stop thread at
  [scope.cpp:373-377](../../sw/runtime/common/scope.cpp#L373) calls
  `vx_scope_stop`, which immediately tries to take `g_stop_mutex`
  ([scope.cpp:397](../../sw/runtime/common/scope.cpp#L397)) — already held
  by the wedged drain thread ([scope.cpp:385](../../sw/runtime/common/scope.cpp#L385)).
  Auto-stop joins the hang instead of breaking it.

**Industry-aligned fix: bounded HW timeout + non-blocking stop.**

Every shipping observability bus (ARM CoreSight ITM/STM, Xilinx ILA over
AXI-Lite, Intel SignalTap II, AXI Firewall IP) bounds the host-visible
stall: the slave commits to responding within a max number of cycles or
returns SLVERR with a sentinel data pattern. Two layers, mirroring how
production debug stacks handle this:

1. **AFU bounded stall.** In `VX_afu_ctrl`, count cycles `rvalid_stall` is
   high; if it exceeds a parameterized threshold (suggestion: `4 * 64` =
   256 clocks, i.e. four serial-bus round-trips), force
   `scope_rdata_valid=1` with a sentinel payload (e.g.
   `0xDEAD_DEAD_DEAD_DEAD`), clear `cmd_scope_reading` / `scope_bus_ctr`,
   and return `rresp=2'b10` (SLVERR) on the matching AXI read. Same
   structure as Xilinx AXI Firewall: stall budget → graceful error response.
2. **Runtime watchdog + non-blocking stop.**
   - Make `g_running` a `std::atomic<bool>` and clear it in `vx_scope_stop`
     *before* taking `g_stop_mutex`, so a wedged drain holder cannot block
     a stop indefinitely.
   - In `vx_scope_drain`, check `g_running.load()` between taps so a stop
     request takes effect within one tap of latency.
   - In `read_reg`, treat the SLVERR / sentinel response as "skip this tap
     this drain" rather than fatal — the next pass retries.

Together: a failed tap or desynced serial bus degrades to "SCOPE drops
samples on that tap" instead of "host hangs the whole runtime." Matches
exactly how a CoreSight trace fault degrades to dropped packets, not a CPU
hang.

## 5. Efficiency

**E-1. ~6-cycle latency floor per command + no fetch prefetch.**
Engine FSM is 3 states minimum; fetch is single-CL-outstanding
([VX_cp_fetch.sv](../../hw/rtl/cp/VX_cp_fetch.sv)). With CL holding 5
commands and ~100 cycle host RTT, ~95 cycles per CL are wasted. Add a 2-to-4
deep CL prefetch FIFO.

**E-2. DMA holds the arbiter for the entire transfer.**
[VX_cp_dma.sv](../../hw/rtl/cp/VX_cp_dma.sv) does not re-bid per chunk. A
1 MiB copy = 256 × ~70 cycles = 18000 cycles of held grant; concurrent
queues' DMAs cannot interleave. Real GPU copy engines re-arbitrate every
4 KB.

**E-3. `enqueue_launch` issues ≥12 round-trips per launch.**
Each `WR(...)` macro at [queue.cpp:329-360](../../sw/runtime/common/queue.cpp#L329)
is a full `cp_submit_cl_` cycle (memcpy + 2 MMIO writes + spin-poll
seqnum). A QMD-style packet fuses all of this into one `CMD_LAUNCH`. See A-1.

**E-4. `enqueue_fill_buffer` has no native CP support.**
[queue.cpp:570-605](../../sw/runtime/common/queue.cpp#L570) chunks into
64 KiB host writes. Add `CMD_MEM_FILL` to the CP wire protocol.

**E-5. `drain_cout` runs an unconditional `CMD_MEM_READ` per launch even when nothing was printed.**
[device.cpp:676-718](../../sw/runtime/common/device.cpp#L676). Should be
debug-only, or piggyback on the launch's completion record.

**E-6. `enqueue_signal` doubles host-side and CP-side signaling.**
[queue.cpp:697-716](../../sw/runtime/common/queue.cpp#L697): host already
wrote the slot via `dev_write`, then posts `CMD_EVENT_SIGNAL` that writes the
same value again. Pick one path.

**E-7. RTL completion engine is single-outstanding.**
[VX_cp_completion.sv:100-119](../../hw/rtl/cp/VX_cp_completion.sv#L100). N
queues' retirements bottleneck on 1/cycle. AXI4 supports multi-outstanding
writes per tag.

**E-8. Event slots are 64-byte device allocations for 8 bytes of data.**
[event.cpp:27-42](../../sw/runtime/common/event.cpp#L27) mints a separate
allocation per event, rounded up to `CACHE_BLOCK_SIZE`. Pool them like args
slots.

**E-9. Cycle counter is duplicated in `VX_cp_axil_regfile` and `VX_cp_profiling`.**
Wire one through the other.

**E-10. Per-MMIO read/write overhead under XRT is ~1 µs.**
The completion-polling loop is dominated by `xrt::ip::read_register` syscall
cost. The CP could write each retire's seqnum into a host-coherent mailbox
(over `m_axi_host`) and the runtime would poll cache-coherent host memory
instead.

## 6. Architectural alignment with mainstream GPUs

Where Vortex diverges from NVIDIA / AMD CP conventions, and the cost:

1. **No QMD-style atomic launch packet** → A-1, E-3.
2. **No host-coherent completion mailbox** → completion path is always MMIO
   polling.
3. **No interrupt path with proper status latching** → A-2.
4. **No per-queue context in the CP** — DCR state is global. → A-1.
5. **`CACHE_FLUSH` sweeps cores sequentially via DCR reads.** AMD
   `ACQUIRE_MEM` broadcasts in parallel via fabric writes; this design is
   O(num_cores) per flush AND the runtime issues one after *every* launch
   unconditionally ([device.cpp:380-395](../../sw/runtime/common/device.cpp#L380)).
   Won't scale past ~16 cores. Skip when no host-visible reads follow.
6. **No native `FILL` / `MEMSET` / DMA-with-pattern.** → E-4.
7. **No copy-engine pipelining**: DMA's R→W are sequential per chunk
   ([VX_cp_dma.sv](../../hw/rtl/cp/VX_cp_dma.sv) `S_READ → S_REQ_AW`). Real
   copy engines double-buffer.
8. **`bid_priority` in `VX_cp_arbiter` is declared but unused** — queue
   priorities from the API don't reach hardware.
9. **`head_addr` writeback is unimplemented** — the host must AXI-Lite poll
   head, can't read it coherently from host memory. The regfile carries the
   field but never DMAs to it.

## 7. C++ sim parity with RTL (`sim/common/cmd_processor`)

Audit of `sim/common/cmd_processor.{h,cpp}` against `hw/rtl/cp/VX_cp_*.sv`,
cross-checked against the runtime wire-protocol constants in
[device.cpp](../../sw/runtime/common/device.cpp) and the caps decoder in
[caps.h](../../sw/runtime/common/caps.h).

The two sides agree on the easy parts: opcode encoding, per-opcode command
sizes (sans `F_PROFILE` trailer), 64-byte cache-line geometry,
`MAX_CMDS_PER_CL = 5`, header byte layout, zero-header termination,
CL-boundary overflow handling, all global regfile offsets (`CP_CTRL`,
`CP_DEV_CAPS`, `CP_CYCLE_*`, caps windows), all per-queue regfile offsets
(`RING_BASE_*`, `HEAD_ADDR_*`, `CMPL_ADDR_*`, `RING_SIZE_LOG2`, `CONTROL`,
`TAIL_*`, `SEQNUM`, `ERROR`, `LAST_DCR_RSP`), `gpu_dev_caps` /
`gpu_isa_caps` bit packing, the atomic-LO-then-HI tail-commit rule,
`EVENT_WAIT` comparison-op encoding, `EVENT_SIGNAL` 64-bit write semantics,
`CACHE_FLUSH` per-core sweep using `arg0=num_cores` against
`VX_DCR_BASE_CACHE_FLUSH`, `LAUNCH` busy-rise/busy-fall semantics, ring
fetches of one CL at a time, `DCR_WR/RD` argument encoding, and the default
`CP_RING_SIZE_LOG2 = 16`.

The two sides disagree on the things that matter most for VM- or
unaligned-DMA workloads.

### P-W1. `CP_DEV_CAPS.VM_ENABLED` (bit 24) exists only in the C++ sim
- C++: [cmd_processor.cpp:117-122](../../sim/common/cmd_processor.cpp#L117)
  sets `vm_enabled = 1u << 24`.
- RTL: [VX_cp_axil_regfile.sv:182-185](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L182)
  leaves bits `[31:24]` as zero. There is no VM_ENABLED bit produced by the
  RTL.
- Runtime: [device.cpp:225-227](../../sw/runtime/common/device.cpp#L225) reads
  `dev_caps & (1u<<24)` to discover VM.
- **Consequence:** every FPGA build reports VM disabled regardless of the
  actual hardware build. The runtime never programs `CP_SATP_*` on RTL; any
  kernel relying on VM uses raw PAs.

### P-W3. `F_MEM_PHYSICAL` (0x04) flag is undocumented in `VX_cp_pkg.sv` and ignored by RTL DMA
- C++: [cmd_processor.h:138](../../sim/common/cmd_processor.h#L138) defines
  `MEM_FLAG_PHYSICAL = 0x04`, honored at
  [cmd_processor.cpp:440-451](../../sim/common/cmd_processor.cpp#L440).
- RTL: [VX_cp_pkg.sv:81-82](../../hw/rtl/cp/VX_cp_pkg.sv#L81) defines only
  `F_PROFILE` and `F_FENCE_PRE`. [VX_cp_dma.sv:247](../../hw/rtl/cp/VX_cp_dma.sv#L247)
  explicitly marks `cmd.hdr.flags` as `UNUSED_VAR`.
- **Consequence:** every `CMD_MEM_*` on RTL is "always physical"; on SimX
  with VM enabled it does a PT walk. A VM-enabled kernel that works on SimX
  writes to the wrong device address on RTL.

### P-W4. RTL DMA rounds `arg2` up to 64 B; C++ DMA is byte-exact
- RTL: [VX_cp_dma.sv:121](../../hw/rtl/cp/VX_cp_dma.sv#L121)
  `rem_beats <= (cmd.arg2 + 64'd63) >> 6;`. Always moves whole CLs.
- C++: [cmd_processor.cpp:452-462](../../sim/common/cmd_processor.cpp#L452)
  uses `cur_cmd_.arg2` exactly.
- **Consequence:** RTL writes up to 63 trailing bytes beyond what the runtime
  asked for, potentially clobbering memory after `dst` and exposing data past
  `src`. Silent data corruption never reproduced on SimX. (Cross-references
  C-2.)

### P-S10. Completion-record seqnum off-by-one between sides
- C++: [cmd_processor.cpp:478-483](../../sim/common/cmd_processor.cpp#L478)
  increments `seqnum` then publishes the post-incremented value to
  `cmpl_addr`.
- RTL: [VX_cp_engine.sv:202-203](../../hw/rtl/cp/VX_cp_engine.sv#L202) emits
  `retire_seqnum = seqnum_r` *before* the non-blocking `<=` takes effect, so
  the FIFO carries the pre-incremented value. The regfile mirror at
  [VX_cp_axil_regfile.sv:204](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L204) is
  one cycle later, so it sees the post-incremented value.
- The runtime only polls `Q_SEQNUM`, never reads `cmpl_addr`, so the bug is
  dormant. A future fast-path that polls the completion slot will see RTL
  "one behind" SimX.

### Missing in C++ sim (RTL features unmodeled)
- **Multi-queue.** [cmd_processor.h:169](../../sim/common/cmd_processor.h#L169)
  has a single `Queue q0_`; the regfile only decodes `0x100..0x140`. The RTL
  parameterizes `NUM_QUEUES`. Dormant today because the runtime is
  single-queue.
- **DMA dual-port routing (host vs device).** C++ collapses both via opaque
  hooks; SimX cannot exercise the AFU host/device port split.
- **AXI-Lite DECERR.** RTL returns `bresp=2'b11` for undecoded addresses
  ([VX_cp_axil_regfile.sv:284](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L284));
  C++ silently ignores writes and returns `0xDEADBEEF` for unknown reads.
- **Per-cmd profiling pulses** (`F_PROFILE` writeback). RTL has them; C++
  doesn't. Dormant.
- **`Q_CONTROL` priority/profile-enable bits, four arbiters, completion FIFO
  backpressure, interrupt output, 4 KB-boundary AXI burst handling** — all
  RTL-only.

### Missing in RTL (sim features absent)
- **`CP_SATP_LO/HI` regs at `0x028/0x02C`.** C++ implements them
  ([cmd_processor.cpp:75-76](../../sim/common/cmd_processor.cpp#L75)); RTL
  has no decode for them ([VX_cp_axil_regfile.sv:177-235](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L177)).
  Dormant because of P-W1, but tied to it: fixing W1 *requires* adding the
  SATP regs.
- **VM page-table walk in CP DMA.** C++ implements Sv32/Sv39 walks
  ([cmd_processor.cpp:157-201](../../sim/common/cmd_processor.cpp#L157)); RTL
  DMA has no MMU.
- **`F_MEM_PHYSICAL` flag decode.** See P-W3.

### Other semantic notes
- `S9` — C++ unpack does not honor the `F_PROFILE` 8-byte trailer
  ([cmd_processor.cpp:236-249](../../sim/common/cmd_processor.cpp#L236) vs
  [VX_cp_pkg.sv:163-181](../../hw/rtl/cp/VX_cp_pkg.sv#L163)). Dormant; would
  silently mis-frame the CL the first time the runtime sets `F_PROFILE`.
- `S5` — `CACHE_FLUSH` per-core tag encoding differs in delivery (C++ passes
  `cid` as the hook arg; RTL writes `cid` into the DCR data word's
  `mpm_target_cid` field). Functionally equivalent provided the per-side
  Vortex DCR adapter agrees with its CP.
- `S7` — C++ DMA goes through hook callbacks, not separate host/device AXI
  masters. A bug where RTL `MEM_COPY` mis-routes a port would not surface in
  SimX.

### Bottom line on parity

**A kernel that passes on SimX cannot today be trusted to pass on RTL,** and
the reverse is also unsafe in the multi-queue / DECERR / interrupt-driven
dimensions. Until P-W1, P-W3, P-W4, and the SATP/VM gap (R-1..R-3 in §7) are
reconciled — either by adding VM to the RTL or by hiding VM behind a flag
that flips both sides together — the C++ model is not a faithful reference
for the RTL CP. Silent SimX↔FPGA disagreements should be expected on any
VM-enabled or odd-size DMA workload.

## 8. Minor / notes

- **`load_backend_once`** ([device.cpp:33-65](../../sw/runtime/common/device.cpp#L33))
  uses a global without `std::call_once`; race on the first concurrent
  `vx_device_open`. Use `call_once`.
- **`CHECK_ERR`** ([common.h:44](../../sw/runtime/common/common.h#L44)) prints
  unconditionally — stderr spam when errors are expected (try-grow paths).
- **`Buffer::create`** allocates device memory before `new Buffer`; if `new`
  throws, the allocation leaks. Wrap in unique_ptr / RAII.
- **`Module::~Module`** comment is right; verify against the actual refcount
  flow.
- **Symbol-table footer parser** in [module.cpp:71-166](../../sw/runtime/common/module.cpp#L71)
  trusts attacker-controlled offsets; OK for trusted local binaries, but
  document this assumption.
- **Per-event 64-byte device allocation** — see E-8.
- **`VX_cp_pkg.cmd_size_bytes` default = 4** ([VX_cp_pkg.sv:178](../../hw/rtl/cp/VX_cp_pkg.sv#L178))
  silently mis-frames unknown opcodes. Reject instead.
- **`VX_cp_arbiter.bid_priority`** is wired but ignored — see §6.
- **`VX_cp_engine.no_resource`** ([VX_cp_engine.sv:231](../../hw/rtl/cp/VX_cp_engine.sv#L231))
  is dead state.
- **`VX_cp_pkg.cpe_state_t::head_addr`** is dead — the CP never writes it.
- **`MMIO_CTL_ADDR` / `MMIO_SCP_ADDR` / `CP base 0x1000`** are duplicated in
  both runtime and AFU sources with no shared header — drift risk.
- **`xsim.tcl`** logs every signal — multi-GB `.wdb` files in CI. Gate behind
  env var.

## 9. Prioritized fix list

Ordered for maximum risk reduction with minimum coupling. Items in the same
phase are independent and can be parallelized.

### Phase 1 — unblock production correctness
1. R-1: retain Buffer / Module / Kernel in queue work lambdas.
2. C-1: per-source one-shot FIFO in `VX_cp_completion`.
3. X-1, X-2, R-10: catch C++ exceptions across every C ABI boundary; mutex on
   `host_bos_`.
4. R-2 + X memory barrier + `bo::sync(TO_DEVICE)`: explicit ordering between
   ring write and doorbell on every backend.
5. O-1: convert COUT to a lossy ring (CUDA/HIP shape) — drop on full, atomic
   overflow counter, host prints "[#N: lost K bytes]". Kills the
   producer/consumer deadlock with no CP or RTL changes.
6. O-2: bounded HW timeout on the SCOPE serial bus in `VX_afu_ctrl` (SLVERR
   + sentinel after `~4 × TX_DATAW` clocks) + non-blocking
   `vx_scope_stop` via `std::atomic<bool>` for `g_running`. Removes the
   one runtime-wide hang that's independent of the CP.

### Phase 2 — close silent-corruption paths
7. C-2 / P-W4: byte-exact DMA in RTL (tail `wstrb`) OR reject unaligned size.
8. C-6: range-check `r_ring_size_log2` writes.
9. C-3: gate DCR proxy on `dcr_req_ready`.
10. R-4: serialize `cp_submit_dcr_read` LAST_DCR_RSP read with the SEQNUM
    wait.
11. R-3: quiesce CP before freeing host buffers in `~Device`.
12. R-5: device write under `Event::mu_`.
13. R-7: release legacy_last_event outside `Device::mu_`.

### Phase 3 — restore CP-RTL / C++-sim parity
14. P-W1: add VM_ENABLED bit to RTL `CP_DEV_CAPS` (or remove it from C++).
15. P-W3: define `F_MEM_PHYSICAL` in `VX_cp_pkg.sv` and route through
    `VX_cp_dma`; or document that RTL has no VM and gate the entire path in
    the runtime.
16. P-S10: align completion-record seqnum (publish post-increment on both
    sides).
17. Add SATP regs to RTL regfile (or reject the writes on the runtime side).

### Phase 4 — efficiency
18. A-3: sized staging-buffer pool generalizing `args_pool_`.
19. E-3 / A-1: QMD-style `CMD_LAUNCH` packet; retire DCR-driven launch state.
20. E-2: per-chunk DMA arbiter re-bid.
21. E-1: 2-to-4 deep CL prefetch in `VX_cp_fetch`.
22. E-4: native `CMD_MEM_FILL` opcode.
23. C-4 / A-2: arbiter release between EVENT_WAIT polls; interrupt path
    end-to-end.
24. E-8: pool event slots.

### Phase 5 — AFU + packaging cleanup
25. A-A1: declare and drive all AXI sideband on `m_axi_host` and bank-0
    arbiter; thread sideband through every master.
26. A-A2: either restore IP_ISR/IER/GIER + drive `ap_done` from CP, or
    repackage as `ap_ctrl_none`.
27. A-A3: hold W mux until AW seen.
28. X-5: real `xrt.ini.in` substitution rule.

### Phase 6 — architectural follow-ups (own proposal)
29. Concurrent COUT drain via second CP queue or sideband DMA path —
    complement to O-1 once the lossy ring is in.
30. Multi-queue across runtime + RTL + cmd_processor.cpp.
31. Host-coherent completion mailbox (head/seqnum writeback).
32. Cross-queue `CMD_FENCE` semantics.
33. Priority arbitration through `VX_cp_arbiter.bid_priority`.

## 10. Out of scope

- LLVM / compiler-side changes (kernel ABI, register footer for per-kernel
  occupancy — already tracked in [llvm_vortex_v3_proposal.md](llvm_vortex_v3_proposal.md)).
- gem5 integration ([gem5_v2_cp_migration_proposal.md](gem5_v2_cp_migration_proposal.md)).
- OPAE backend ([cp_opae_integration_plan.md](cp_opae_integration_plan.md)).
- Per-block helper RTL (TEX/RASTER/OM/DXA) — owned by their subsystem
  proposals.
- Vulkan / RT pipeline programmability ([vulkan_support_proposal.md](vulkan_support_proposal.md)).
