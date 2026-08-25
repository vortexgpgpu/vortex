# The `aved` backend — AMD Alveo V80 driver architecture

This document describes the `aved` runtime backend at
[`sw/runtime/aved/vortex.cpp`](../../sw/runtime/aved/vortex.cpp), which runs
Vortex on the **AMD Alveo V80** through the SLASH platform stack. It covers
what the backend is responsible for, how it satisfies the runtime's transport
contract, the three memory paths it has to support, and where it deliberately
diverges from its sibling backends.

It is the **backend deep-dive**. The device-side command protocol lives in
[`command_processor.md`](command_processor.md), the AFU wrapper in
[`fpga_afu_shell.md`](fpga_afu_shell.md), and the platform's address map in
[`../aved_address_map.md`](../aved_address_map.md); none of that is repeated
here. The SLASH stack itself (three-PF topology, `vrtd` wire protocol, VRT
API, `vrtbin`/`system_map.xml`, buddy allocator, connectivity language) is
documented in [`../kb/03_slash_architecture.md`](../kb/03_slash_architecture.md).

The V80 is **not an XDMA shell**, so XRT does not apply to it. That single fact
is why this backend exists rather than being a variant of `xrt`.

---

## 1. Where the backend sits

### 1.1 The transport contract

Every Vortex runtime backend is a **pure transport HAL**. The dispatcher
(`sw/runtime/stub/`) dlopens `libvortex-<name>.so`, resolves `vx_dev_init`, and
receives a `callbacks_t` with exactly five entry points
([`callbacks.h`](../../sw/runtime/common/callbacks.h)):

| Entry point | Responsibility |
|---|---|
| `dev_open` / `dev_close` | device lifecycle |
| `cp_reg_write` / `cp_reg_read` | 32-bit window into the Command Processor regfile |
| `host_mem_alloc` / `host_mem_free` | CP-visible host memory (command ring + DMA staging) |

Everything else — device-memory allocation, DMA, capability decoding, the
command ring, queues, events — lives in the common core
([`sw/runtime/common/device.cpp`](../../sw/runtime/common/device.cpp)). The CP
is the sole memory engine; a backend never moves data itself.

`aved` implements that contract and nothing beyond it. The extra volume in the
file relative to its siblings is not extra responsibility — it is the cost of
making those five operations work on this platform.

### 1.2 Backend comparison

| | `opae` | `xrt` | `aved` |
|---|---|---|---|
| Lines | 255 | 370 | ~1170 |
| Platform | Intel CCI-P | Xilinx XDMA/XRT | AMD V80 / SLASH |
| Device handle | `fpga_handle` | `xrt::device` + `xrt::ip` | `vrt::Device` + `vrt::Kernel` |
| Sim variant | `opaesim` | `xrtsim` (`XRTSIM`) | `avedsim` (`AVEDSIM`) |
| CP regfile base | `0x1000` | `0x1000` | `0x1000` |
| CP-visible host memory | `fpgaPrepareBuffer` (CCI-P shared) | `xrt::bo` host-only | **three paths** — see §4 |
| Exception wrapper | none (C API) | `XRT_TRY`/`XRT_CATCH` | `VRT_TRY`/`VRT_CATCH` |

The structural conventions are shared deliberately: the same `CP_BASE`, the
same `CHECK_HANDLE` / `CHECK_ERR` idiom, the same dual-API split between a
hardware C++ path and an in-process C simulation path, the same
`std::map<uint64_t, …>` of host buffers keyed by the device-visible address
that `host_mem_free` is later given, guarded by a mutex because queue workers
allocate from arbitrary threads.

---

## 2. Build targets

`TARGET` selects what executes the RTL
([`sw/runtime/aved/Makefile`](../../sw/runtime/aved/Makefile)):

| `TARGET` | Links | What runs | Use |
|---|---|---|---|
| `avedsim` | `libavedsim.so` | Verilator model **in this process** | default; fast functional iteration |
| `sim` | real VRT | VRT's simulation platform (xsim, out of process) | platform-level checks |
| `hw` | real VRT | the V80 | silicon |
| `emu` | — | rejected at configure time | see below |

`TARGET=emu` is refused with a hard `$(error)`: the V80 emulation platform is a
behavioural C-model that does not accept RTL kernels, so a build that appeared
to succeed would fail confusingly at run time.

`CPP_API` is defined for everything except `avedsim`. Under `avedsim` the
model runs in-process and shares this process's memory, so the C shim
([`sim/avedsim/vrt_c.cpp`](../../sim/avedsim/vrt_c.cpp)) maps the VRT calls
onto `xrt_sim` — the *same* Verilator harness the `xrt` backend uses. That
sharing is why `avedsim` and `xrt` track each other cycle-for-cycle, and why a
divergence between them is a real signal.

---

## 3. Device lifecycle

`init()` does markedly more than its siblings, and each step exists because of
a specific failure this platform can produce.

### 3.1 Opening the device

`vrt::Device(bdf, vbin_path, program)` opens the board and optionally
reprograms the PL; `vrt::Kernel` then parses the vbin's `system_map.xml`.

`VORTEX_AVED_NO_PROGRAM=1` constructs with `program=false`. This matters
because each open that programs the PL goes through `vrtd`'s design writer,
which runs a reset sequence whenever the requested shell differs from the
current one — and that toggles a **secondary bus reset** on the card's root
port. With the design already resident (loaded over JTAG), skipping the
reprogram removes that path entirely.

### 3.2 The transport gate

Before anything else, `init()` reads `CP_CYCLE_LO` twice.

This gate must come **before** the device reset, because the reset handshake
cannot detect a dead bus: it breaks on `ctl & CTL_AP_IDLE`, and an AXI DECERR
or PCIe completion timeout substitutes `0xFFFFFFFF` on the way back — which has
that bit set. A dead bus would therefore certify itself as healthy, and the
first real symptom would appear much later and much further away (historically:
the common core decoding all-ones `CP_DEV_CAPS` as `VM_ENABLED` and spinning on
65,536 PTEs, which presents as a hang rather than a bus error).

`CP_CYCLE_LO` free-runs — `VX_cp_axil_regfile` increments it every clock
unconditionally — so two reads that differ prove three things at once: reads
reach the register file, they return real data rather than a bus artifact, and
the AFU clock is live.

### 3.3 Parking the CP

`init()` clears `Q_CONTROL.enable` and `CP_CTRL`, then polls `CP_STATUS.busy`
until the CP's fetch has parked and in-flight commands have drained. The RTL is
explicit that this is the intended mechanism: *"To stop a queue, the host clears
Q_CONTROL.enable and the fetch parks in IDLE while in-flight commands drain
naturally"* ([`VX_cp_core.sv`](../../hw/rtl/cp/VX_cp_core.sv)).

A process that crashed, was killed, or was cut short by a host reset ran no
teardown at all, so entry must assume the device was left running. A master
must never be reset while it has outstanding transactions.

If the CP is still busy after this, `init()` **fails** rather than proceeding:
opening anyway would queue every new command behind a stuck one and stall at
the first poll, far from the cause.

### 3.4 The device reset is off by default

Writing `CTL_AP_RESET` is **measurably fatal on the V80 compute shell**. With
the CP parked and `CP_STATUS` reading `busy=0`, the next register read returns
the all-ones no-completion signature and the card leaves the PCIe bus until it
is JTAG-reloaded and the host rebooted. Offset `0x00` is the AFU control
register rather than the CP, so the entire AXI-Lite slave goes down with it —
and no secondary bus reset is involved anywhere.

Nothing needs the write. `VORTEX_AVED_RESET=1` restores it for a platform that
does.

This is the sharpest divergence from `xrt` and `opae`, both of which reset
unconditionally in `init()` and rely on it.

### 3.5 Resuming the CP's position

The common core adopts the CP's current `Q_SEQNUM` at open rather than assuming
the device starts at zero
([`device.cpp`](../../sw/runtime/common/device.cpp), `cp_init`).

The CP's head and retire counters **survive a process exit** — nothing clears
them. `VX_cp_core` decodes the regfile's `q_reset_pulse` and discards it
(`UNUSED_VAR`), so `Q_CONTROL.reset` and `CP_CTRL.reset_all` are both no-ops,
and the only thing that ever cleared the counters was the AFU device reset,
which §3.4 rules out.

The fetch gate is `head < tail`, both absolute byte counts, so a second process
restarting its tail at 0 never advances past the first one's head and its queue
silently never runs — no error, no `Q_ERROR`, no CP-side timeout. On an idle
device `head == tail == retired × CP_CL_BYTES`, so the retire counter is enough
to resume exactly where the previous process stopped.

This is a **driver workaround for a hardware gap**, not a repair. A working
software reset in the CP would remove the need for it.

---

## 4. CP-visible host memory — three paths

This is where `aved` genuinely diverges, and the reason for most of its size.

`callbacks.h` states the contract: *"The region must be coherent with the CP's
`m_axi_host` view (no explicit sync callback)."* `opae` satisfies it with a
CCI-P shared buffer and `xrt` with a host-only BO. On the V80 compute shell,
**one of the three paths cannot**.

### 4.1 Path A — the QDMA slave bridge (contract-native)

When `m_axi_host` is tagged `HOST`, it reaches host DRAM through the QDMA slave
bridge, and `vrtDevice_.allocHostBuffer()` serves exactly what the contract
describes. This is the `xrt`-equivalent path and needs no sync.

**It does not work on this compute shell.** An AXI master bound to `:HOST`
never sees its reads complete: a minimal HLS kernel whose only distinguishing
feature is `sp=<kernel>.m_axi_gmem0:HOST` sits at `ap_start` indefinitely on a
loop that should retire in about a microsecond, while the byte-identical build
with that one line changed to `:HBM0` completes in under 0.1 s. That isolates
the fault to the HOST path rather than to mastering, the AFU, or the build flow.

### 4.2 Path B — staged in device memory (the hardware path)

The ring, head/completion cachelines and DMA staging buffers are instead placed
in **device memory** (HBM), which `m_axi_host` can reach. The bytes then need
an explicit publish/refresh, because device memory is not coherent with the
host — at exactly two moments the CP protocol already supplies:

* **The doorbell** (`Q_TAIL_LO`) is the last write before the CP reads the ring
  → publish host→device.
* **`Q_SEQNUM`** is polled after the CP has written its results back
  → refresh device→host.

Both moments pass through `cp_reg_write` / `cp_reg_read`, so the whole
mechanism stays inside the backend and the common core keeps the coherent
contract it documents.

Three rules make it correct:

1. **The ring is excluded from refresh.** It is ours to write: the core fills
   the host shadow and the bytes only reach the device at the next doorbell.
   Refresh runs from the `Q_SEQNUM` poll, which can land between an append and
   that doorbell, so pulling the ring back would overwrite freshly appended
   descriptors with a stale device copy and silently drop commands.
2. **Head and completion lines are excluded from publish** once the queue is
   live — they belong to the CP, and pushing them would clobber its writes.
   They are included exactly once, for the one-shot seeding on the first
   doorbell, because simulated/device memory starts undefined and a CP reading
   garbage there would compute a bogus pending count.
3. **Frees are deferred.** A `vrt::Buffer` destructor hands its device memory
   straight back to the buddy allocator, while a descriptor in the ring may
   still name that address. Regions freed by `host_mem_free` park until the
   next doorbell fixes the retire count that covers them, then wait for
   `Q_SEQNUM` to reach it.

Self-configuring, not a build flag: `portMemoryConfig()` reads the connection
map out of the vbin's `system_map.xml` and throws when the port has no memory
target. A vbin built with `HOST_TAG=HOST` therefore keeps Path A untouched and
one built with `HOST_TAG=HBM1` stages, with no way for the two to disagree.

### 4.3 Path C — the simulation model

Under VRT's simulation platform the CP's view of host memory is a model living
in the xsim process. The `sim_*` apparatus mirrors Path B's publish/refresh, but
transfers through the ZMQ server instead, and ships only the ring sub-range
appended since the last doorbell (every byte is a simulated AXI burst).

Under `avedsim` there is nothing to sync at all — the Verilator model shares
this process's memory — so the correct behaviour is the plain path.

### 4.4 Testing the untestable path

Path B is otherwise hardware-only, which makes the riskiest code in the file
first-executed on silicon. `VORTEX_AVED_FORCE_STAGE=1` runs it against the
simulator instead, so the ordering, the ring exclusion and the deferred-free
lifetime all get exercised without a board.

---

## 5. Diagnostics

Two facilities exist here that the sibling backends do not have, both added
because a failure on this platform can take the host down with the card.

**`VORTEX_AVED_MMIO_TRACE=<path>`** records every hardware register access,
fsync'd. That is deliberate: the failure mode under investigation wedges the
card and hard-resets the host within about a second, and a buffered log does not
survive. Identical consecutive accesses are coalesced into `xN` records so a
spin loop cannot push the interesting records out of reach, while every
*transition* is recorded exactly. A read of `0xFFFFFFFF` is the PCIe
completion-timeout signature on this platform, never data, and the trace flags
that transition once.

**`VORTEX_CP_TRACE=1`** (common core) names every ring command as it is
appended, so a stall maps to an opcode rather than to an index.

Neither is on by default; both gate on one already-initialised bool.

---

## 6. Design invariants

1. **The backend never moves data.** Only the CP does. `host_mem_alloc`
   returns a pointer and an address; every transfer is a CP command issued by
   the common core.
2. **`cp_reg_*` take CP-internal offsets.** The backend adds `CP_BASE`. This
   is uniform across `opae`, `xrt` and `aved`, and the common core is written
   against the CP-internal map only.
3. **No VRT exception may cross the `callbacks_t` boundary.** Propagating one
   through `extern "C"` is UB; `VRT_TRY`/`VRT_CATCH` convert to `-1` and log
   once. `VRT_CATCH_RC` is the variant that records the failing access in the
   MMIO trace before unwinding.
4. **A master is never reset with transactions outstanding.**
5. **Device state is adopted, not assumed.** The CP's counters persist across
   processes; the driver reads them rather than presuming zero.
6. **AXI-Lite decodes in 16-byte blocks.** A partially populated block DECERRs
   on *every* word, including implemented ones. Adding a CP register means
   padding its block to four words — enforced by `check-axil-blocks` at build
   time.

---

## 7. Divergences from the sibling backends

Recorded so they are deliberate rather than accidental:

| Divergence | Why | Status |
|---|---|---|
| Staged CP memory (§4.2) breaks the "no explicit sync" contract | the `HOST` slave bridge does not work on this shell | workaround; revisit if a shell fixes `:HOST` |
| Device reset off by default (§3.4) | the write takes the card off the PCIe bus | permanent for this platform |
| Resume from `Q_SEQNUM` (§3.5) | the CP has no working software reset | workaround for an RTL gap |
| Transport gate + CP park in `init()` (§3.2–3.3) | the reset handshake cannot detect a dead bus | keep |
| Reaches VRT internals — `vrt::detail::reserveFakePhysAddr`, `getHandle()->getZmqServer()` | no public API for the simulation memory windows | **coupling risk**; would break on a VRT refactor |
| `sim_*` and `staged_*` are near-duplicate publish/refresh engines | different transports, same protocol moments | candidate for unification |

`opae` has a latent issue the others do not: its `host_bos_` map has no mutex,
while queue workers call `host_mem_alloc`/`host_mem_free` from arbitrary
threads. `xrt` and `aved` both guard it.

---

## 8. Open directions

1. **A software reset for the CP.** `q_reset_pulse` is decoded in the regfile
   and discarded in `VX_cp_core`. Wiring it would remove the §3.5 workaround
   and make a hung kernel recoverable without reconfiguring the partition —
   today a kernel that never completes leaves the CP unusable.
2. **Unify the two sync engines** (§4.3) behind one publish/refresh interface
   with a pluggable transport.
3. **Drop the VRT-internal reach** once VRT exposes the simulated memory
   windows and the ZMQ server through public API.
4. **Gate the build on timing closure.** The V80 flow packages a bitstream
   that does not meet timing without saying so; see
   [`../reports/v80_timing_closure.md`](../reports/v80_timing_closure.md).
