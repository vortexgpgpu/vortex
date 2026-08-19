# Vortex on Alveo V80 — `TARGET=hw` bring-up: status, post-mortem, and completion plan

**Revision 3 — 2026-08-19.** Supersedes revisions 1 (2026-08-04) and 2
(2026-08-14) in their entirety. Those revisions contained four claims that were
later measured to be false; they are preserved as errata in §7 rather than
deleted, because the *way* they were wrong is the most useful thing in this
document.

**Status:** the transport works. `TARGET=hw` executes the full CP command path
on silicon. Three of five ladder rungs remain unrun, blocked on board recovery
rather than on any known defect.

| | |
|---|---|
| Scope | `sw/runtime/aved/vortex.cpp`, `sw/runtime/common/device.cpp`, `hw/rtl/cp/VX_cp_axil_regfile.sv`, `hw/syn/xilinx/aved/`, `~/dev/v80/` |
| Affects | `TARGET=hw`. `TARGET=sim` and `TARGET=avedsim` are green and must stay green. |
| Blocking | Board is off the PCIe bus (root port `0000:00:01.1` absent). BIOS AER severity unchanged. |

---

## 1. Executive summary

Three weeks were spent on what turned out to be **two independent hardware
faults, stacked, with the first masking the second**, on a platform whose
iteration cost was ~45 minutes per experiment and whose failure mode was
occasionally a host hard-reset.

Both faults are now found and fixed:

| # | Fault | Symptom | Fix | Date |
|---|---|---|---|---|
| **A** | AXI-Lite register reads DECERR | 6 of 9 CP global registers unreadable; runtime decoded `0xFFFFFFFF` as capabilities and hung | Pad every 16-byte block in `VX_cp_axil_regfile.sv` to 4 words | 2026-08-14 |
| **B** | `m_axi_host` → `HOST` (QDMA slave bridge) never completes reads | CP accepts the ring, reports no error, and never fetches a descriptor | Route `m_axi_host` to `HBM1`; stage CP memory in device memory with explicit publish/refresh | 2026-08-19 |

With both fixed, on 2026-08-19 08:19:

```
[VXDRV] m_axi_host targets device memory; staging CP memory there (HBM port 1)
[minimal] queue_create → buffer_create → write → readback → wait → verify
PASSED!
```

`minimal -l` drives `MEM_WRITE` and `MEM_READ` descriptors through the CP ring
and verifies the returned bytes. **The CP fetches, executes, and retires
commands on real silicon.** That result stands and does not need re-proving.

What remains is four ladder rungs (`minimal` full, `demo`, `sgemv`, `sgemm`), a
board that is currently unenumerated, and a BIOS setting that should be changed
before any further reset is attempted.

---

## 2. Where the project actually is

### 2.1 Hardware

| Artefact | State |
|---|---|
| `build32_aved_hw/bin/vortex_afu.vbin` | Aug 14 03:03 — block-padding RTL fix, `m_axi_host:HOST`. **Superseded.** Keep for the HOST-vs-HBM comparison. |
| `hbm1_aved_hw/bin/vortex_afu.vbin` | Aug 19 03:36 — block-padding fix + `m_axi_host:HBM1`. **This is the image to run.** |
| `hbm1_aved_hw/config.cfg` | `shell=compute`, `sp=…m_axi_mem_0:MEM`, `sp=…m_axi_host:HBM1`, `krnl freqhz=200000000` |

The two hw trees are separate build directories, not rebuilds of one, so both
survive on disk and `VBIN_DIR` in `run_hw_test.sh` selects between them without
a 37-minute resynthesis.

### 2.2 Source changes (all uncommitted — see §8, Phase 0)

| File | Change |
|---|---|
| `hw/rtl/cp/VX_cp_axil_regfile.sv` | +39: pad blocks `0x00`, `0x20`, and each queue's `0x30` block to 4 words; document the measured 16-byte-block rule |
| `sw/runtime/aved/vortex.cpp` | +250: staged-CP-memory path (`staged_probe/publish/refresh/xfer`), `VORTEX_AVED_NO_PROGRAM`, `#ifdef CPP_API` guards that had broken `avedsim` |
| `sw/runtime/common/device.cpp` | +132: reject all-ones `CP_DEV_CAPS`; factor `cp_poll_seqnum_()`; 10-second stall diagnostic dumping the whole queue block; `VORTEX_CP_POLL_TIMEOUT_S` |
| `sw/runtime/common/vortex2_internal.h` | +14: `cp_poll_seqnum_` declaration |
| `hw/rtl/dxa/VX_dxa_smem_wr.sv` | +90/−? — unrelated to this work, review separately before committing |

Untracked: `docs/kb/` (3 knowledge bases), this document,
`~/dev/v80/{hostprobe,hostprobe_hbm,hw_ladder.sh,v80_rescan_load.sh}`.

### 2.3 Board

```
root port 0000:00:01.1   ABSENT
lspci -d 10ee:           (empty)
slash / ami              not loaded
last boot                ended 09:40:28 by hard reset during TOGGLE_SBR
```

The board needs recovery before anything can run. See §8 Phase 1 — and note
that the obvious recovery (JTAG shell load) is the one that arms the crash.

### 2.4 Test status

| Rung | Command | `sim` | `hw` |
|---|---|---|---|
| 1 | `minimal OPTS="-n4 -l"` | pass | **PASS 2026-08-19** |
| 2 | `minimal OPTS="-n1"` | pass | not run |
| 3 | `demo OPTS="-n64"` | pass | not run |
| 4 | `sgemv OPTS="-n32"` | pass | not run |
| 5 | `sgemm` (default) | pass, `instrs=336912` | not run |

---

## 3. Fault A — AXI-Lite reads DECERR (found 2026-08-14)

### 3.1 What was measured

An 80-address sweep (`~/dev/v80/sweep.cpp`) on a fresh bitstream and a healthy
board:

```
RESPOND (16):  0x010 0x014 0x018 0x01C      <- globals
               0x100 .. 0x12C               <- all 12 per-queue registers
DECERR  (64):  everything else
```

Six addresses that `is_decoded()` claims to implement returned `0xFFFFFFFF`:
`0x000`, `0x004`, `0x008`, `0x020`, `0x024`, `0x130`.

### 3.2 The actual rule

A **16-byte-aligned block answers reads if and only if all four of its words are
decoded.** A partially-populated block DECERRs on *every* word, including the
words that are implemented.

```
block 0x000   3/4 implemented (0x0C missing)      -> whole block DECERR
block 0x010   4/4                                  -> OK
block 0x020   2/4 (0x20, 0x24 only)                -> whole block DECERR
block 0x100/0x110/0x120   4/4                      -> OK
block 0x130   1/4 (0x130 only)                     -> whole block DECERR
```

80/80 addresses fit. This is not a Vortex bug and not an RTL logic bug: the
AXI-Lite path between the shell's `S_AXILITE_INI` and the AFU's `s_axi_control`
resolves at 128-bit granularity and rejects a block it cannot fully resolve. The
per-word decode inside `VX_cp_axil_regfile` is never consulted.

### 3.3 The fix

Pad the incomplete blocks. `0x0C`, `0x28`, `0x2C` read as zero; the queue block
gains `0x34/0x38/0x3C`. Nine lines of `is_decoded()`, three lines of
`read_reg()`, and a long comment so nobody re-derives it.

Note that `0x28`/`0x2C` are `CP_SATP_LO/HI` in the software ABI. This CP has no
MMU (`grep -rn satp hw/rtl/cp` is empty), correctly reports `VM_ENABLED=0`, and
the runtime never writes them. Reading zero is the honest answer.

**Standing rule for this RTL: when you add a register, pad its 16-byte block out
to four words.** A lint check for this is in the roadmap (§8 Phase 4).

---

## 4. Fault B — `HOST` mastering does not work on this shell (found 2026-08-19)

### 4.1 The controlled experiment

Two builds. The HLS source is byte-identical (`diff -q` verified). One line of
`config.cfg` differs.

| build | `sp=` target | AP_CTRL trace | outcome |
|---|---|---|---|
| `~/dev/v80/hostprobe` | `HOST` | stuck at `0x1` (`ap_start`, not idle, not done) for ~15 min | never completes |
| `~/dev/v80/hostprobe_hbm` | `HBM0` | `0x4` → `0x1` → `0xe` | `sum=524800` correct, < 0.1 s |

Argument registers were verified correct in the failing case (`src=0xfea9a000`,
`size=0x100`). The workload is a 256-iteration II=1 loop that should retire in
about a microsecond.

This isolates the fault to the `HOST` path — not to mastering in general, not to
the AFU, not to the build flow.

`HOST` resolves to the QDMA slave bridge, sinking at
`/qdma_slave_bridge_noc/S00_AXI` (`slashkit/emit/hw/tcl_gen.py:86`). It is a
designed, documented feature with kernel-driver backing
(`driver/slash_hostbuf.c` allocates DMA-coherent host memory precisely so a
kernel can pull a command ring from it). It simply does not function on the V80
compute shell as installed here.

### 4.2 Why neither simulator caught it

- **avedsim** — the Verilator model runs in this process and shares our memory
  directly. There is no bus.
- **sim** — the `sim_*` path in `vortex.cpp` explicitly copies host memory into
  the model over ZeroMQ.

Both bypass the exact mechanism that fails. **A host-mastering path can only be
validated on hardware.** This is now recorded in KB-1 §8.3 and KB-3 §9.

### 4.3 The fix

Route `m_axi_host` to `HBM1` and stage the CP's memory there. The runtime
detects which variant it was handed rather than taking a build flag:

```cpp
void staged_probe() {
  staged_cfg_.reset();
  if (sim_mode_) return;
  try {
    staged_cfg_ = vrtKernel_.portMemoryConfig(HOST_PORT_NAME);   // "m_axi_host"
    // ... device memory: stage
  } catch (const std::exception&) {
    // target="HOST": the slave bridge. Leave the existing path alone.
  }
}
```

`portMemoryConfig()` reads the connection map out of the vbin's
`system_map.xml` and throws when the port has no memory target. So a
`HOST_TAG=HOST` vbin keeps the old behaviour and a `HOST_TAG=HBM1` vbin stages,
with no way for the two to disagree. One binary serves both.

Staging then needs the same explicit publish/refresh the simulation path needs,
for the same reason (device memory is not coherent with the host) and at exactly
the same two moments: **publish at the doorbell** (`CP_Q_TAIL_LO` write, before
the CP reads) and **refresh on a seqnum change** (after it has written back).

Three correctness details in that code, each of which would be a silent
data-corruption bug:

1. **`staged_refresh()` must exclude the ring.** The ring is host-written; the
   refresh runs from the `Q_SEQNUM` poll, which can land between a descriptor
   append and its doorbell. Pulling the ring back device→host would overwrite
   freshly appended descriptors with the stale device copy and silently drop
   commands. `sim_refresh()` skips it for the same reason — which is where this
   was caught, by reading the existing code before spending a reset.
2. **`staged_publish()` must exclude the head and completion cachelines once the
   queue is live.** Those belong to the CP; pushing them would clobber its
   writes. `include_cp_owned` is set only for the one-shot seed on the first
   doorbell.
3. **Round every allocation up to 4096 bytes.** VRT's `MediumBlockSuperblock` is
   `BuddySuperblockBase<12, 21>`, so 2^12 is the allocator's floor. The head and
   completion regions are one cacheline each and throw
   `Size too small for MediumBlockSuperblock` out of `vx_device_open` otherwise.

---

## 5. Driver and software architecture — corrected

The original proposal had no architecture section. It reasoned about registers
without a model of what sat between them and the host. That gap is what made
Fault A take ten days. This is the model.

### 5.1 The path a register read actually takes

```
  Device::cp_reg_read(0x008)                       sw/runtime/common/device.cpp
    └─ vortex_aved::cp_reg_read(off)               sw/runtime/aved/vortex.cpp
         └─ read_register(CP_BASE + off)
              └─ vrt::Kernel::read(addr)           libvrt
                   └─ libvrtd++ / libvrtd          AF_UNIX SOCK_SEQPACKET
                        └─ vrtd                    daemon, holds the privileged fd
                             └─ libslash ctldev
                                  └─ slash.ko PF2 (10EE:50C2)  GET_BAR_FD → dma-buf
                                       └─ mmap'd BAR, DMA_BUF_IOCTL_SYNC bracketed
  ─────────────────────────── PCIe ───────────────────────────
                                            CPM5 PCIe controller 1, Gen5 x8
                                              └─ PF0 BAR0 window 0x201_0000_0000 [256 MB]
                                                   └─ NoC → S_AXILITE_INI
                                                        └─ SmartConnect / AXI crossbar
                                                             ◀── FAULT A LIVES HERE
                                                             └─ AFU s_axi_control (64 K)
                                                                  └─ VX_afu_wrap addr[12] split
                                                                       ├─ 0: legacy ctrl block
                                                                       └─ 1: VX_cp_axil_regfile
```

Fault A was in the crossbar layer — the one layer with no source in this repo,
no model in either simulator, and no mention in the original proposal.

### 5.2 The path the CP takes to reach its ring

```
  VX_cp_* fetch engine
    └─ m_axi_host  (AXI4 master out of the AFU)
         ├─ HOST_TAG=HOST → /qdma_slave_bridge_noc/S00_AXI
         │                    └─ CPM5 slave bridge → PCIe → host DRAM
         │                         (dma_alloc_coherent'd by slash_hostbuf.c)
         │                    ◀── FAULT B: reads never complete
         │
         └─ HOST_TAG=HBM1 → NoC → HBM controller, channel 1
                              └─ device memory; host reaches it via QDMA PF1
                                   (vrt::Buffer::sync, explicit publish/refresh)
```

Two directions, two independently-failing mechanisms, one of which is invisible
to both simulators. **The original proposal covered only the first.** Its §7
ladder even described rung 1 (`minimal -l`) as "CP DMA reaches device memory. No
kernel launch at all — cannot stall the AFU even if the aperture is wrong." That
is exactly backwards: `-l` writes a pattern and reads it back *through the CP
command ring*, so it is the rung that most directly exercises Fault B's path. It
was the most dangerous rung, presented as the safest.

### 5.3 Platform facts the runtime now depends on

| Fact | Consequence in code |
|---|---|
| `0xFFFFFFFF` is the PCIe completion-timeout / DECERR signature | `device.cpp` rejects all-ones `CP_DEV_CAPS`; any poll must reject `~0` before testing bits |
| AXI-Lite resolves in 16-byte blocks | `VX_cp_axil_regfile.sv` pads every block to 4 words |
| VRT's allocator floor is 4096 bytes | `STAGED_MIN_ALLOC` in `vortex.cpp` |
| `vrt::Kernel::wait()` never times out | `VORTEX_CP_POLL_TIMEOUT_S`; poll `AP_CTRL` directly in probes |
| A PDI write succeeds only on a freshly reset device | `VORTEX_AVED_NO_PROGRAM=1` → `vrt::Device(bdf, vbin, false)` |
| `portMemoryConfig()` throws for `target="HOST"` | used as the staging probe — detect, don't flag |
| `std::cout` is block-buffered under `runuser`/`tee` | all probe progress goes to `std::cerr` |

---

## 6. Why three weeks

Stated as causes, in descending order of how much time each cost.

### 6.1 I did not make the experiment cheap before running experiments

This is the largest single cause and the one entirely within my control.

The per-experiment cost was: PDI load → a second load in the same session
*always* fails → AMC to `NO_AMC` → JTAG recovery → reboot → sometimes root port
absent → more recovery. Call it 45 minutes, occasionally a host hard-reset, and
twice an unplanned outage.

The fix was one constructor parameter — `vrt::Device(bdf, vbin, /*program=*/false)`
— which was available from day one and which I found on **day 15**. With it, a
five-rung ladder runs on one reset. Without it, the ladder as written in the
original proposal would have consumed five recovery cycles.

Three weeks of paying a 45-minute tax per experiment, when a day spent reading
`vrt/device.hpp` and `flash_worker.c` would have removed it.

### 6.2 Two faults stacked, and the first hid the second

Fault A blocked the runtime before it ever mastered anything, so Fault B could
not present. Fixing A required a resynthesis (~37 min) and a fresh board; only
then did B become visible, and B required a *second* resynthesis with a
different connectivity config. That is structurally two serial debug cycles on a
platform with an expensive turnaround, and there was no way to parallelise them
because the second was unobservable until the first was done.

This is the part I would not have compressed much. But it accounts for maybe a
week, not three.

### 6.3 Both simulators are green and neither can see either fault

`TARGET=sim` passed `sgemm` with `instrs=336912` throughout. That is genuinely
useful — it proves the kernels and the host code are right — but it created a
false sense that the remaining gap was small. Both faults live in layers neither
simulator models: an AXI crossbar (A) and a PCIe slave bridge (B). Every real
bug was hardware-only, at the slowest possible feedback rate.

### 6.4 I reasoned from logs instead of running controls

The original proposal contains the line *"Static log analysis produced five
wrong conclusions tonight."* It then went on to produce more. The retraction
list:

- "There is no evidence AXI-Lite has ever worked" — wrong, it worked.
- "The bitstream does not match the source tree" — wrong, that was preprocessing.
- "`(addr[4] | addr[8]) == 0` predicts DECERR" — retracted, then re-derived two
  paragraphs later on the same ten points, and wrong both times.
- "`orcas2` hard-resets from a cache-parity MCE on core 1" — wrong; the resets
  are triggered by PCIe link transitions, not by the CPU.

Every one of those was a pattern fitted to observational data with no control
experiment. By contrast, the two experiments that actually resolved something
were both minimal controls with a single variable: the 80-address sweep (day 11)
and `hostprobe` vs `hostprobe_hbm` (day 15). Both were cheap. Both should have
come first.

### 6.5 I lacked the substrate knowledge to know which layer to suspect

Having now written the three knowledge bases, all three faults read as textbook:

- 16-byte block granularity is normal AXI crossbar behaviour — a 128-bit segment
  resolves at 128-bit granularity.
- `0xFFFFFFFF` meaning "no completion" is universal PCIe, not a Xilinx quirk.
- A slave bridge is an independent path requiring independent verification.

The original proposal's hypothesis table eliminated six candidates — RTL logic,
stale bitstream, timing, runtime software, NoC merge, transport-dead — and the
true cause was in none of those categories, because every category was *inside
the Vortex design*. The hypothesis space never included the shell-to-AFU AXI
path's own decode behaviour, and never included the reverse direction at all.

### 6.6 Board fragility, some of it self-inflicted

Root-port disappearance, JTAG recoveries, `Shell: unknown` forcing an SBR, two
host hard-resets. A real fraction of the elapsed time was restoring the board
rather than testing on it.

And I made it worse. My "JTAG-load the shell before rebooting" advice does fix
enumeration — but it leaves firmware reporting `Shell: unknown`, which
*guarantees* a shell switch on the next design load, which runs the SBR that
twice reset the host. I gave that advice without tracing its consequence, and
then flagged the consequence only after the user had already rebooted.

---

## 7. Errata — every incorrect claim in revisions 1 and 2

Kept deliberately. The failure mode is more instructive than the conclusions.

| # | Claim | Where | Verdict | What was actually true | Root error |
|---|---|---|---|---|---|
| 1 | "There is no evidence AXI-Lite to the AFU has ever worked." | r1 §3.1 | **False** | The cycle counter at `0x010` advances; `GPU_DEV_CAPS` at `0x018` decodes coherently. 16 of 80 addresses always worked. | Treated a read *value* as data before establishing the read *completed*. `0xFFFFFFFF` is the no-completion signature. |
| 2 | "The bitstream does not match the source tree." | r2 §11.4 | **Retracted** | The build preprocesses RTL per target. After normalising comments, whitespace, and `` `UNUSED_VAR `` scaffolding: 0 real differing lines in the regfile. | Diffed generated artefacts as if they were source. |
| 3 | "Every decoded address fails iff `(addr[4] \| addr[8]) == 0`." | r2 §12.1b | **False** | The rule was fitted to 10 hand-picked points. An 80-address sweep disproves it. The real rule is 16-byte block completeness. | Pattern-fitted to insufficient data — and, having explicitly retracted the same rule in §12.1, restated it two paragraphs later. |
| 4 | "`orcas2` hard-resets from a fatal cache-parity MCE on physical core 1; pin runs away from it." | r1 §8 | **False** | Every fatal MCE is triggered by `slash.ko` PCIe rescans / link transitions. The CPU is fine. | Correlation with single-threaded runs mistaken for causation. Three weeks of `taskset` against a non-problem. |
| 5 | "Rung 1 (`minimal -l`) cannot stall the AFU — no kernel launch at all." | r1 §7 | **False** | `-l` writes and reads back *through the CP command ring*, so it exercises the host-mastering path directly. It is the rung Fault B kills. | Confused "launches no kernel" with "touches no bus". |
| 6 | Exit criterion 1: "`axil_probe` reads `0x00061001` from `0x1008`." | r1 §9 | **Unmeetable** | `0x1008` sits in a partially-populated block and could not respond regardless of its value. | Wrote an exit criterion against a register whose reachability was the thing under test. |
| 7 | "`slashkit` has no console script" filed as an aside. | r2 §12.3 | **Misprioritised** | It made the hw image unbuildable from a clean checkout (`make` Error 127). | A P0 build blocker recorded as a footnote. |
| 8 | Six hypotheses "eliminated". | r2 §12 | **Incomplete** | All six were inside the Vortex design. The true causes were in the AXI crossbar and the PCIe slave bridge. | Hypothesis space drawn too narrowly; no layer-by-layer decomposition of the path. |
| 9 | The entire document treats "the transport" as AXI-Lite only. | r1–r2, throughout | **Incomplete** | There are two directions. The reverse one (CP → host memory) blocked everything after Fault A was fixed. | No architecture model; see §5. |
| 10 | The plan assumes hardware runs are repeatable at will. | r1 §7 | **False** | One PDI write per device reset. The second always fails and costs a JTAG recovery. | No cost model for the iteration loop. |

---

## 8. Roadmap to completion

Five phases. Phases 0 and 1 are prerequisites; 2 is the actual remaining work
and takes about an hour.

### Phase 0 — off-board work — **DONE 2026-08-19**

Committed as `6579edce6`, `6a4c6bc48`, `553e78609`, `820996f9e`, `55f30697b`.

| Item | Status |
|---|---|
| Commit the tree | done — 5 commits, nothing pushed |
| Deferred-free fix (use-after-free in batch mode) | done, builds clean, **runtime-unvalidated** |
| Transport gate | done, validated under `avedsim` |
| Block-padding lint (`hw/scripts/check_axil_blocks.py`) | done; retrodicts the exact three failing blocks |
| BIOS PCIe fatal-error severity | **not done — user action** |
| `slashkit` console-script defect | not done |

Two files deliberately left uncommitted: `README.md` carries an accidental
regression (drops the SLASH setup-guide link, adds trailing whitespace), and
`hw/rtl/dxa/VX_dxa_smem_wr.sv` is unrelated RTL work needing separate review.

**A use-after-free was found by reading, not by burning a board cycle.** In
batch mode `host_free` published the staging region and then destroyed the
`vrt::Buffer`, returning the HBM block to VRT's allocator while a ring
descriptor still named it. The non-batch path is safe only by accident —
`cp_submit_cl_` rings the doorbell and polls to completion before `host_free`
is reached — which is exactly why `minimal -l` passed and why rung 2, the
first rung that launches a kernel, would not have. Regions are now held in
`staged_pending_free_` until `staged_refresh()` observes a `Q_SEQNUM` advance.

Also discovered: **`TARGET=avedsim` did not compile at HEAD** (unguarded
`sim_mode_`). The `#ifdef CPP_API` guards fix it. It now builds and runs, and
the transport gate executes and passes there; `minimal -l` still stalls after
`CONFIGS` under avedsim, which is pre-existing (there is no HEAD baseline to
regress from) and off the hardware path.

### Phase 0b — original Phase 0 text, retained for reference

**0.1 Commit the tree.** Five files of hard-won fixes are uncommitted and one
host reset from being lost. A truncated `libvortex-aved.so` has already been
produced this way once.

```
hw/rtl/cp/VX_cp_axil_regfile.sv        block padding + the measured rule
sw/runtime/aved/vortex.cpp             staged CP memory, NO_PROGRAM, CPP_API guards
sw/runtime/common/device.cpp           all-ones rejection, poll timeout, stall dump
sw/runtime/common/vortex2_internal.h   cp_poll_seqnum_ decl
docs/kb/, docs/proposals/              knowledge bases + this document
```

`hw/rtl/dxa/VX_dxa_smem_wr.sv` is unrelated to this work — review and commit it
separately.

Also move `~/dev/v80/{hostprobe,hostprobe_hbm,hw_ladder.sh,v80_rescan_load.sh,
v80_oneshot.sh}` into the repo under `hw/syn/xilinx/aved/tools/`. They are the
only reproduction path for both controlled experiments and they currently live
outside version control.

**0.2 Change the BIOS PCIe error severity.** *This is the gate for everything
that follows.* Until a fatal PCIe error on root port `0000:00:01.1` stops being
escalated to a platform reset, every recovery attempt risks taking the host
down, and the two crashes on 2026-08-19 will recur.

Look for, in order of likelihood:
- *AMD CBS → NBIO* → PCIe uncorrectable/fatal error severity → non-fatal
- *PCIe AER Support* → disabled, or severity downgraded
- *System Error Severity* / *System Error on PCIe fatal* → disabled

Verify afterwards with `lspci -vv -s 00:01.1 | grep -A3 UESvrt` — fatal-severity
bits should be clear for the classes an SBR provokes.

**0.3 Add a build-time lint for the block rule.** A five-line script over
`is_decoded()` that fails the build if any 16-byte block is partially populated.
Fault A cost ten days; it must not be reintroduced by the next register added.

**0.4 Fix the `slashkit` console-script defect properly** rather than via the
`~/.local/bin/slashkit` shim, so the hw image builds from a clean checkout.

### Phase 1 — recover the board (one time, minimum risk)

Root port `0000:00:01.1` is absent, so no rescan can help (KB-2 §8.1).

**Preferred: cold power cycle.** Full shutdown, mains off, power on. The card
boots from OSPI, wins the POST race, enumerates, and — critically — comes up
reporting a *known* shell rather than `Shell: unknown`. That avoids the shell
switch, which avoids the SBR, which avoids the crash path entirely.

*The user has said cold power cycling is no longer available. If that still
holds, the fallback is the JTAG shell load — but only after Phase 0.2, and with
the understanding that the next design load will force an SBR.*

Verify before proceeding:

```bash
ls /sys/bus/pci/devices/0000:00:01.1        # must exist
lspci -d 10ee: -nn                          # 50b4, 50c1, 50c2
sudo ami_tool overview                      # state READY, not NO_AMC
v80-smi list                                # PF0 / PF1 / PF2 / VRTD all pass
v80-smi list | grep Shell                   # must NOT read "unknown"
```

Exit criterion: all five pass. If `Shell` reads `unknown`, stop and expect one
SBR on the next program — do not run the ladder until Phase 0.2 is done.

### Phase 2 — run the ladder on ONE reset (the remaining work)

```bash
bash ~/dev/v80/hw_ladder.sh          # minimal -l → minimal → demo → sgemv → sgemm
```

The script already implements the correct discipline: rung 1 programs the PDI
and runs `-l` only; every later rung sets `VORTEX_AVED_NO_PROGRAM=1` and reuses
the resident design. Set `PROGRAM_FIRST=0` when the vortex design is already
loaded — a load that fails costs a JTAG recovery and there is nothing to gain
from reloading a design that is already there.

Ordering is cheapest-first so a break is localised:

| # | Rung | Proves | If it fails |
|---|---|---|---|
| 1 | `minimal -n4 -l` | CP fetches and retires `MEM_WRITE`/`MEM_READ`; staging publish/refresh is correct | Stop. Everything after is noise. Read the 10-second stall dump — it distinguishes "setup writes never landed" from "doorbell never landed" from "CP cannot master to that address". |
| 2 | `minimal -n1` | A kernel launches, retires, signals completion | Fault is in launch/completion, not in the ring |
| 3 | `demo -n64` | Multi-warp execution + typed verification | Fault is in warp scheduling or memory ordering |
| 4 | `sgemv -n32` | `float4` vector loads, strided access | Fault is in vectorised load path |
| 5 | `sgemm` | Tiled GEMM, LMEM + global traffic | The headline workload |

Timeouts are already bounded: `HW_TIMEOUT=900`, `VORTEX_CP_POLL_TIMEOUT_S=60`.
A wedged AFU aborts instead of hanging until JTAG.

### Phase 3 — verification, not just green ticks

1. **Instruction-count cross-check.** Hardware `sgemm` must report
   `instrs=336912`, the same count `sim` and `avedsim` produce. A `PASSED` line
   alone does not prove the hardware executed the same program; a matching
   retire count does, and a mismatch localises the divergence immediately.
2. **Re-run the ladder a second time on the same reset** to confirm
   `program=false` reuse is stable and there is no state leak between runs.
3. **Run the `HOST` variant once** (`VBIN_DIR=…/build32_aved_hw/bin`) and
   confirm it still fails at rung 1. That keeps Fault B falsifiable rather than
   folklore, and gives AMD a reproducible case (§ Phase 5).
4. **Confirm `TARGET=sim` and `TARGET=avedsim` are still green.** The
   `#ifdef CPP_API` guards in `vortex.cpp` were added because unguarded
   `sim_mode_` references broke the `avedsim` build outright.

### Phase 4 — harden so this cannot recur

1. **Transport gate in `init()`.** Before the `ap_reset` write, read a register
   with a known non-trivial constant and fail loudly on `0xFFFFFFFF` or
   `0x00000000`. This was proposed in revision 1 §6 and is *still not
   implemented*; the all-ones rejection in `device.cpp` catches the specific
   `CP_DEV_CAPS` case but there is no general gate. One clear error line versus
   a 65,536-PTE spin.
2. **Harden every poll against `~0`.** The `ap_idle` check
   (`vortex.cpp` ~line 190) breaks out on `ctl & (1<<2)`, and `0xFFFFFFFF` has
   bit 2 set. It has been self-certifying since day one.
3. **Land the block-padding lint** from Phase 0.3.
4. **CI rung.** Add `minimal -l` under `TARGET=hw` to the nightly, gated on the
   board being healthy, so a regression surfaces in a day rather than a month.

### Phase 5 — upstream the platform defect

The `HOST` / QDMA-slave-bridge path is a documented SLASH feature with kernel
support that does not work on the V80 compute shell. `hostprobe` vs
`hostprobe_hbm` is a ten-line, single-variable reproduction. Report it with both
builds attached — it is the kind of defect that will cost the next user three
weeks too.

---

## 9. Exit criteria

1. Phase 0 committed; BIOS AER severity changed and verified.
2. Board enumerates with a known shell; all five Phase 1 checks pass.
3. Ladder rungs 1–5 pass under `TARGET=hw` on a single device reset.
4. Hardware `sgemm` reports `instrs=336912`, matching `sim` and `avedsim`.
5. `TARGET=sim` and `TARGET=avedsim` still green.
6. Transport gate and block-padding lint in place.
7. `hostprobe`/`hostprobe_hbm` reproduction filed upstream.

---

## 10. Standing rules for this project

Distilled from the three weeks. Each one cost something.

1. **Make the experiment cheap before running experiments.** If an iteration
   costs 45 minutes, spend a day removing the cost first.
2. **Run a control, don't fit a pattern.** Minimal design, one variable, on
   hardware. Both faults were resolved this way; nothing was resolved by log
   analysis.
3. **`0xFFFFFFFF` is not data.** Reject it before testing any bit.
4. **Decompose the path by layer before hypothesising.** The bug is often in the
   layer with no source in your repo.
5. **A green simulator does not validate a bus.** Anything crossing PCIe — in
   either direction — is hardware-only.
6. **Program once per session.** `vrt::Device(bdf, vbin, false)` thereafter.
7. **Pad every 16-byte block.**
8. **Round allocations up to 4096.**
9. **Unbind every driver before any link transition**, and prefer not to have a
   link transition at all.
10. **Commit before touching the board.** Two host resets so far; one already
    truncated a shared library mid-build.

---

## 11. References

- [`docs/kb/01_xilinx_fpga_driver_development.md`](../kb/01_xilinx_fpga_driver_development.md) — PCIe substrate, DMA/IRQ/ABI patterns, XDMA/QDMA/CPM5, reset and AER
- [`docs/kb/02_v80_fpga_configuration.md`](../kb/02_v80_fpga_configuration.md) — PDI/PLM/OSPI, segmented configuration, NoC address map, recovery
- [`docs/kb/03_slash_architecture.md`](../kb/03_slash_architecture.md) — SLASH stack, kernel ABI, vrtd, VRT API, `config.cfg` grammar
- `docs/proposals/aved_afu_proposal.md`, `docs/proposals/slash_v80_bringup_report.md` — earlier design work
- `docs/aved_address_map.md` — the address-map investigation
- Memory: `v80-hw-decerr`, `v80-pdi-load-reset`, `v80-pcie-root-port`,
  `v80-aperture-fixes`, `orcas2-cpu-mce`, `slash-framework`
