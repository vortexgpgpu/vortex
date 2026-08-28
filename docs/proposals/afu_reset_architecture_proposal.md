# The AFU reset defect: root cause, fix, and what remains

**Status: ACCEPTED ON SILICON** (2026-08-27). `minimal` runs twice in one boot
with no JTAG reload, both resets honoured, on the minimal and the full
configuration; the sequencer's refusal path is also confirmed against a
genuinely stuck master (§8.2's mis-built bitstream provided the test case).
Reloading an AFU over a used one now needs no reboot. Remaining: hung-kernel
recovery via `Q_CONTROL.reset` (§8 item 8).
**Scope:** `hw/rtl/afu/common/` — `VX_afu_axil_demux.sv`, `VX_afu_axi_drain.sv`,
`VX_afu_reset_seq.sv`, `VX_afu_req_gate.sv` (all new), `VX_afu_wrap.sv`,
`VX_afu_ctrl.sv`; `hw/rtl/cp/` — `VX_cp_core.sv`, `VX_cp_fetch.sv`,
`VX_cp_engine.sv`, `VX_cp_axil_regfile.sv`; `sw/runtime/aved/vortex.cpp`;
`hw/syn/xilinx/aved/platforms.mk`, `hw/syn/xilinx/aved/Makefile`;
`hw/unittest/afu_axil_demux/`, `hw/unittest/afu_reset_seq/`

---

## 1. Summary

Writing `CTL_AP_RESET` to AFU control offset `0x00` took the V80 off the PCIe
bus. **The reset was never the problem.** The fatal defect was in the AFU's
AXI-Lite address demux, and it had nothing to do with resetting anything — the
`ap_reset` pulse the write was asking for never even fired.

The demux routed the AXI-Lite **W** (write-data) beat using a register that is
only updated when the **AW** (write-address) beat handshakes. So a write to the
legacy window (`addr[12]=0`) whose predecessor went to the CP window
(`addr[12]=1`) had its **AW delivered to `VX_afu_ctrl` and its W delivered to
the CP regfile**. `VX_afu_ctrl` then waited forever for a W beat the CP had
already swallowed, and no `BRESP` was ever produced. With a write outstanding
and unacknowledged, the shell's AXI-Lite master stopped making progress, and
every subsequent host access died of a PCIe completion timeout — the all-ones
signature.

The driver's `CTL_AP_RESET` write to `0x00` was the **only** access the runtime
ever made below `0x1000`, and it always followed the CP quiesce writes. So the
one write that could trigger the bug was the one write that always did.

This is now:

* **reproduced** in simulation from the exact recorded MMIO sequence
  (§3), and
* **fixed** by `VX_afu_axil_demux`, with a regression test that fails when
  the defect is re-injected (§5).

Two further defects sat behind it, both real and both now fixed (§6): the
reset had **no drain logic**, so it would have abandoned in-flight AXI
transactions the moment it became reachable; and **the CP had no software
reset** at all, because `q_reset_pulse` was decoded in the regfile and
discarded in `VX_cp_core`.

`VORTEX_AVED_RESET` is gone — the reset is unconditional again, as in `xrt`
and `opae` (§7).

---

## 2. The evidence that overturned the earlier explanation

An earlier version of this document claimed `ap_reset` reset AXI masters
mid-transaction and that the U55C's XDMA decoupler absorbed the dangling
transaction. **Both halves were wrong.** The V80 has a `dfx_decoupler_0` of its
own (529 LUTs in its utilization report), and the recorded traces rule out
outstanding transactions entirely.

### 2.1 The card never left the bus

`~/dev/v80/logs/prev_1758/sample.tsv`, sampled at 1 Hz across the wedge:

```
16:51:55  crumb=rung:minimal-loopback  link=32.0 GT/s PCIe/8  pf0=up
16:51:56  crumb=rung:minimal-loopback  link=32.0 GT/s PCIe/8  pf0=up
...
```

The link stayed **up at full speed and full width** the entire time. No AER
fatal or non-fatal error was recorded. Whatever happened was not a link event
and not a device-level failure.

### 2.2 It happened on a completely idle device

`~/dev/v80/logs/prev_1758/mmio_minimal.tsv` — the whole process, three accesses:

```
read   0x00001010  0x04012e41   ok
read   0x00001010  0x041edf4b   ok
write  0x00000000  0x00000010   <- CTL_AP_RESET
read   0x00000000  0xffffffff   *** 65 ms later: no completion ***
```

The queue was never enabled, no kernel ran, no DMA was issued. There was
nothing in flight to be reset out from under. The outstanding-transaction
hypothesis is falsified by this trace alone.

### 2.3 The failure is a completion timeout, not a hang of the reset domain

Post-wedge reads are spaced ~55 ms apart and all return `0xFFFFFFFF`. That is a
PCIe completion timeout being served on every access (the default Range-B CTO
is 50–100 ms). The read that times out targets offset `0x00`, which is served
by `VX_afu_ctrl` — a module that is **not** in the soft-reset domain
(`VX_afu_wrap.sv` gives it `reset` alone, deliberately). So "the reset knocked
out the slave" cannot explain it either.

### 2.4 The second trace names the trigger

`mmio_minimal_loopback.tsv` shows the full driver sequence:

```
read   0x00001010   ok      CP window
read   0x00001010   ok      CP window
write  0x0000111c   ok      CP window  (Q_CONTROL.enable = 0)
write  0x00001000   ok      CP window  (CP_CTRL = 0)
read   0x00001004   ok      CP window  (CP_STATUS -> busy = 0)
write  0x00000000   0x10    LEGACY window  <- first sub-0x1000 access
read   0x00000000   0xffffffff             <- dead
```

Every access before the fatal one is in the CP window. The fatal one is the
first — and only — access in the legacy window.

---

## 3. The defect

### 3.1 The two slaves disagree about W

`VX_afu_ctrl` uses a strict three-state write FSM
(`VX_afu_ctrl.sv:225,236-247`):

```systemverilog
assign s_axi_wready = (wstate == WSTATE_DATA) && ~wready_stall;
// WSTATE_ADDR -> (aw_fire) -> WSTATE_DATA -> (w_fire) -> WSTATE_RESP
```

It accepts a W beat only **after** its AW has landed.

`VX_cp_axil_regfile` buffers the two channels independently
(`VX_cp_axil_regfile.sv:279-280`):

```systemverilog
assign axil_s.awready = !wr_addr_buf_valid;
assign axil_s.wready  = !wr_data_buf_valid;   // no AW required
```

It accepts a W beat **whenever its data buffer is empty**, with or without a
matching AW. That is legal, and it is why a misrouted W beat disappears
silently instead of stalling visibly.

### 3.2 The demux routed W by a stale register

The original demux in `VX_afu_wrap.sv`:

```systemverilog
always @(posedge clk) begin
    ...
    if (s_axi_ctrl_awvalid && s_axi_ctrl_awready) begin
        route_cp_w_r <= is_cp_aw;          // updated only at AW handshake
        ...
assign lg_wvalid  = s_axi_ctrl_wvalid && !route_cp_w_r;   // W routed by the
assign cp_axil.wvalid = s_axi_ctrl_wvalid && route_cp_w_r; //   OLD value
assign s_axi_ctrl_wready = route_cp_w_r ? cp_axil.wready : lg_wready;
```

AW routing is combinational from the incoming address; W routing is by a
register that does not reflect that address until the *next* cycle.

### 3.3 The deadlock, cycle by cycle

Xilinx AXI-Lite masters present AW and W in the same cycle, and AXI4
explicitly forbids a master from waiting for `AWREADY` before asserting
`WVALID` — precisely to avoid deadlocking slaves that want data first. So this
is not an unusual case; it is the normal one.

Cycle *N*, write to `0x0000` with `route_cp_w_r` still `1` from the preceding
`0x1000` write:

| Channel | Routed by | Goes to | Result |
|---|---|---|---|
| AW `0x0000` | `is_cp_aw = 0` (combinational) | `VX_afu_ctrl` | accepted; `wstate → WSTATE_DATA` |
| W `0x10` | `route_cp_w_r = 1` (stale) | CP regfile | accepted into `wr_data_buf` |

Cycle *N+1*: `route_cp_w_r` finally becomes `0`. But the master has already
seen `WREADY` and dropped `WVALID`. `VX_afu_ctrl` sits in `WSTATE_DATA`
waiting for a W beat that will never come, so `lg_bvalid` stays low **forever**.
`ap_reset` is gated on `s_axi_w_fire` in `VX_afu_ctrl`, so the reset never
fires. Meanwhile the CP regfile holds an orphan W beat with no matching AW, so
its `wready` is stuck low too — the next CP write would also hang.

The host-side consequence follows: an AXI-Lite write that never returns
`BRESP` leaves the shell's AXI-Lite master unable to retire it, so nothing
after it makes progress and every read times out. That is exactly the ~55 ms
all-ones cadence in §2.3.

### 3.4 Reproduced

`hw/unittest/afu_axil_demux` drives the recorded sequence into the **real**
`VX_afu_ctrl` plus a model of `VX_cp_axil_regfile`'s write channel. Against the
original demux:

```
2. legacy write follows CP-window writes (AVED ordering)
  write 0x111c (CP) returns BRESP                            PASS
  write 0x1000 (CP) returns BRESP                            PASS
  read 0x1004 (CP) returns RRESP                             PASS
  write 0x0000 returns BRESP                                 *** FAIL ***
  ap_reset pulsed                                            *** FAIL ***
  no orphaned W beat left in the CP slave                    *** FAIL ***
```

Deterministic, and it matches the hardware trace access for access.

---

## 4. Why `driver=xrt` on a U55C never hit it

Same RTL — `hw/rtl/afu/xrt/vortex_afu.v` and `hw/rtl/afu/aved/vortex_afu.v`
differ only in port names and whitespace, and both instantiate the same
`VX_afu_wrap`. The defect was present on the U55C the whole time. **Two
independent properties of the XRT path kept it from ever being triggered**, and
either one alone is sufficient.

### 4.1 XRT writes `ap_reset` before it touches the CP

`sw/runtime/xrt/vortex.cpp:206` issues `CTL_AP_RESET` immediately after
`xrt::ip` is opened, before any CP-window access exists. `route_cp_w_r` is
therefore still `0`, which matches `is_cp_aw = 0`, so AW and W route together.

`sw/runtime/aved/vortex.cpp` puts the same write **after** the CP quiesce block
(`Q_CONTROL`, `CP_CTRL`, `CP_STATUS` — all in the CP window). That ordering is
what creates the mismatch.

### 4.2 XRT resets the AFU on every open; AVED deliberately does not

`sw/runtime/xrt/vortex.cpp:166` calls `xrtDevice.load_xclbin(...)` on every
process open, which re-asserts the kernel's `ap_rst_n` and clears
`route_cp_w_r` to `0`.

The aved backend opens with `program = false`
(`sw/runtime/aved/vortex.cpp:219`) because a V80 design write only succeeds on
a freshly reset device. Nothing re-asserts `ap_rst_n`, so `route_cp_w_r`
survives from the previous process — which always ended with CP-window writes.

This also explains the observed history exactly: the first run after a JTAG
reload survived the `ap_reset` write (fresh `reset`, `route_cp_w_r == 0`), and
every run after it died on the same write.

### 4.3 Why no simulator caught it

`xrtsim` runs the xrt backend, so it inherits §4.1's ordering and never
produces the mismatch. Every simulation also starts from a fresh `reset`, so
`route_cp_w_r` begins at `0`. The bug needs a *stale* routing latch, which is
a state only a second process on real silicon could reach.

### 4.4 It was not V80-specific

Nothing about the Versal NoC, the QDMA bridge, or the compute shell is required
to trigger this. A U55C running the aved ordering — or an XRT build with
`SCOPE` enabled, which writes `MMIO_SCP_ADDR = 0x28` in the legacy window after
CP traffic — would deadlock identically. The V80 was simply the platform whose
driver used the ordering that exposes it.

---

## 5. The fix

`VX_afu_axil_demux` replaces the inline demux. The rule: **W routing must never
depend on a register that only updates at the AW handshake.**

```systemverilog
wire wr_route_known = wr_pending || aw_fire;
wire wr_route       = wr_pending ? wr_route_r : sel_aw;

assign s_wready = wr_route_known && (wr_route ? m1_wready : m0_wready);
```

Three cases, all correct:

| Situation | Route source |
|---|---|
| a write is already pending | its latched route |
| AW is firing this cycle | **falls through** from the incoming address |
| W arrived before its AW | route unknown → `wready` held low until AW lands |

This is the standard structure — the same fall-through that `axi_lite_demux`
implementations expose as a `FallThrough` parameter, for exactly this reason.

Two latent defects in the same block are fixed alongside it:

* **AW/AR are now stalled while a transaction of that direction is pending.**
  The original forced a new AW to the *previous* write's slave
  (`route_aw = route_cp_w_valid ? route_cp_w_r : is_cp_aw`), which would
  mis-route outright if the master ever pipelined two writes.
* **B and R are qualified by the pending flag**, so a response can never be
  presented from a slave that owns no outstanding transaction.

The cost is one cycle of turnaround between back-to-back writes on a control
path that runs at a few accesses per millisecond, plus a combinational
dependency of `s_wready` on `s_awvalid` and the address decode. That direction
is explicitly permitted — AXI4 forbids a *master* from waiting for `AWREADY`
before `WVALID`, not a slave from waiting for `AWVALID` before `WREADY` — and
it is a short path on a 32-bit control interface. It should be checked against
the timing report on the next build all the same, since the design already runs
close to closure.

### 5.1 Regression test

`hw/unittest/afu_axil_demux` — registered in `hw/unittest/Makefile`, so
`make -C hw/unittest run` covers it. Seven groups: the XRT ordering, the AVED
ordering (§3.4), the reverse direction, W-before-AW, W-after-AW, sixteen
alternating writes, and alternating reads checked for correct slave data.

All pass against the fix. Re-injecting the original W-routing makes exactly
group 2 fail, which is the property that makes the test worth keeping.

---

## 6. The reset itself is now sequenced

The demux fix made `CTL_AP_RESET` reachable again, but reaching
`VX_afu_ctrl` was never enough on its own: the old path pulsed the reset-delay
shift register straight from the write, resetting `Vortex_axi`, `VX_cp_core`
and `bank0_arb` **with no drain logic at all**. A search of `VX_afu_wrap.sv`
for `outstanding`, `inflight` or `drain` returned zero matches. Any transaction
in flight when the pulse landed would have been abandoned mid-burst.

That never got a chance to bite, because the demux deadlocked first. It is
fixed now rather than left as a latent trap.

### 6.1 Outstanding-transaction tracking

[`VX_afu_axi_drain`](../../hw/rtl/afu/common/VX_afu_axi_drain.sv) sits on every
AXI master — each `m_axi_mem_<i>` and `m_axi_host` — and reports `idle` when
the port owes the interconnect nothing:

```
idle = (aw_count == w_count) && (aw_count == b_count) && (ar_count == r_count)
```

Three pairs of free-running wrapping counters compared for equality, rather
than up/down counters. Equality is right in both directions, so it stays
correct when AXI4 allows write data to be presented before its address — which
would drive an up/down counter negative.

### 6.2 The request gate

New `AW`/`AR` are withheld from the shell while a reset is quiescing, so the
counters reach zero in bounded time even if the core is still running.

The first version of this gate was written the obvious way, and it was wrong:

```systemverilog
assign m_axi_mem_awvalid_a[i] = pre_awvalid_a[i] && !rst_stop_req;   // WRONG
```

AXI4 §A3.2.1 requires that once `VALID` is asserted it stays asserted until the
edge where `VALID` and `READY` are both high. A master may not withdraw an
offer because it changed its mind. That expression does exactly that whenever
`stop_req` rises while a request is already being presented, and an
interconnect is entitled to have latched it — the Versal NoC does. The port is
then permanently out of step with the slave: the transaction is neither
accepted nor retractable, later transfers on the same port never complete, and
the drain counters, which only observe the handshake, report that master busy
forever. A reset sequencer built to avoid corrupting the bus would have been
corrupting it itself.

[`VX_afu_req_gate`](../../hw/rtl/afu/common/VX_afu_req_gate.sv) registers the
block decision, and the register may only change while no offer is outstanding:

```systemverilog
always @(posedge clk) begin
    if (reset) blocked <= 1'b0;
    else if (!(out_valid && !out_ready)) blocked <= stop_req;
end
assign out_valid = in_valid && !blocked;
assign in_ready  = out_ready && !blocked;
```

A request already presented stays presented until it is accepted; the next one
is held off. Quiescing therefore takes effect *within one transaction* rather
than instantly — bounded by the slave's own acceptance latency, and exactly
what quiescing ought to mean.

`W`, `B` and `R` are never gated: a burst whose address the interconnect has
already accepted must be allowed to finish. Gating at the AFU boundary rather
than inside Vortex means no change to the core — it simply stalls, and it is
about to be reset anyway.

### 6.3 The sequencer

[`VX_afu_reset_seq`](../../hw/rtl/afu/common/VX_afu_reset_seq.sv):

| State | Action | Exit |
|---|---|---|
| `IDLE` | — | `ap_reset` → `QUIESCE` |
| `QUIESCE` | assert `stop_req` | all masters idle → `ASSERT`; timeout → `ERROR` |
| `ASSERT` | one cycle: reload the reset-delay shift register | → `RELEASE` |
| `RELEASE` | hold `stop_req` until the delayed reset has drained | → `IDLE` |
| `ERROR` | raise a sticky status bit; **no reset is asserted** | → `IDLE` |

The timeout matters more than the happy path. Resetting a master that will not
drain is what breaks the interconnect, so the sequencer refuses and reports it.
A device that says "I could not reset" is strictly more useful than one that
silently corrupts the bus — the software already learned this lesson and
refuses to proceed when `CP_STATUS.busy` will not clear.

The platform reset keeps its direct path to the shift register. It must always
work, sequencer or not.

`busy` drives `ap_idle`, so the runtime's existing poll observes the whole
sequence unchanged; it simply became truthful. The refusal is reported in a new
`ap_ctrl` read bit 5 (`CTL_RESET_ERROR`).

### 6.4 The CP queue reset now exists

`q_reset_pulse` was decoded in `VX_cp_axil_regfile` and discarded in
`VX_cp_core` (`UNUSED_VAR`), so `Q_CONTROL.reset` and `CP_CTRL.reset_all` were
no-ops. The register map advertised a capability the hardware did not
implement, which is worse than not offering it.

It is wired now, with the same quiesce-then-clear discipline:

* the pulse is latched as a **pending** request, not applied immediately;
* while pending, `VX_cp_fetch` stops issuing new reads (`stop_req`), so the
  CPE drains on its own;
* the clear is applied in the single cycle when both the fetch and the engine
  report `idle`, so an AXI read is never abandoned in flight;
* `head_r`, `seqnum_r`, **and the host-programmed tail and enable bit** clear
  together. That last part is not optional: the fetch gate is `head < tail` on
  absolute byte counts, so a head cleared against a stale tail would
  immediately refetch the whole ring.

### 6.5 Tests

`hw/unittest/afu_reset_seq` drives the real `VX_afu_reset_seq` and
`VX_afu_axi_drain`, plus the same shift register `VX_afu_wrap` uses: reset
withheld while a read is outstanding, withheld while a write burst still owes
`WLAST` or `BRESP`, refused with `timeout_error` when a master never drains,
the sticky error cleared by a later successful request, and the platform reset
working regardless. Injecting the old "reset without draining" behaviour fails
six of those assertions.

`hw/unittest/cp_core` gained the queue-reset scenario end to end: run a
command, confirm `Q_SEQNUM` advanced, pulse `Q_CONTROL.reset`, then confirm
seqnum, head, tail and enable all read back zero. Re-injecting the discarded
pulse fails it.

---

## 7. Software consequences

* **`VORTEX_AVED_RESET` is gone.** The aved backend now writes `CTL_AP_RESET`
  unconditionally in `init()`, as `xrt` and `opae` do. The variable only ever
  existed to dodge the demux bug; keeping an escape hatch for a fixed defect
  just preserves a second, less-tested code path.
* The runtime checks `CTL_RESET_ERROR` after the `ap_idle` poll and fails the
  open with a specific diagnostic if the device declined to reset.
* **Do _not_ reorder the aved `init()` to match xrt.** Moving `CTL_AP_RESET`
  ahead of the CP quiesce block would have sidestepped the demux issue the way
  xrt does, but that ordering is the *worse* one: it resets the CP and its AXI
  masters without first parking them. XRT is not doing this right; it is
  getting away with it because `load_xclbin` leaves it a freshly reset device
  with nothing in flight. The aved order — quiesce, confirm
  `CP_STATUS.busy == 0`, then reset — is correct and stays.
* `cp_quiesce_()` in `device.cpp` stays. Quiescing before teardown is right
  regardless, and it is what makes `ASSERT` reachable quickly.
* The runtime's resume-from-`Q_SEQNUM` workaround (`205160014`) can stay — it
  is harmless and cheap — or be removed now that `Q_CONTROL.reset` works.
  Removing it should wait until the reset is confirmed on silicon.
* Recovery from a hung kernel becomes `Q_CONTROL.reset` instead of
  `jtag_load_vortex.sh`.

---

## 8. Validation status

1. ~~Reproduce the demux failure in simulation.~~ **Done** (§3.4).
2. ~~Fix it, with a test that fails against the defect.~~ **Done** (§5.1).
3. ~~Track outstanding transactions and refuse to reset a master that will not
   drain.~~ **Done** (§6.1–6.3), tested in `hw/unittest/afu_reset_seq`.
4. ~~Wire `q_reset_pulse`.~~ **Done** (§6.4), tested in `hw/unittest/cp_core`.
5. ~~Elaborate and run.~~ **Done** — `xrtsim` and `avedsim` build clean and
   pass `demo` with the reset unconditional.
6. **On silicon — partly done** (`rst2c`, 2026-08-27). Confirmed on the board:

   | Behaviour | Observed |
   |---|---|
   | `CTL_AP_RESET` completes instead of wedging the card | ✅ |
   | `ap_ctrl` reads back afterwards (no completion timeout) | ✅ |
   | Reset succeeds on a drained device, `reset_error` clear | ✅ |
   | Reset **refused** on an undrained master, `CTL_RESET_ERROR` set | ✅ |
   | `Q_SEQNUM` cleared by the queue reset | ✅ |

   The demux fix, the sequencer's success path, its refusal path, and the CP
   queue reset all work on silicon. What that bitstream could **not**
   demonstrate is `minimal` twice in one boot, because it carried an unrelated
   build-configuration defect (§8.2) that hung the CP before the second run.
7. **`minimal` twice in one boot** — **DONE, ACCEPTED** (2026-08-27, boot of
   22:24). Both runs passed with the reset honoured (`ap_ctrl=0x00000004`,
   `CTL_RESET_ERROR` clear) and zero all-ones reads, on the minimal config
   (`rstmin`, 1 core) **and** on the full graphics config (`rst3`), which then
   passed vecadd/sgemv/sgemm; `demo` still shows its pre-existing wrong-results
   failure, which is tracked separately and does not involve the reset.
   Loading `rst3` over a *used* `rstmin` needed no reboot and no recovery —
   the region drains now. Scripted as
   [`hw/syn/xilinx/aved/tools/reset_acceptance.sh`](../../hw/syn/xilinx/aved/tools/reset_acceptance.sh),
   which runs the test twice and reads each run's register trace: the reset
   must have been issued, honoured (`ap_idle` set, `CTL_RESET_ERROR` clear),
   and the card must never return `0xFFFFFFFF`. A refused reset fails the test
   even when the binary passes.
8. **Hung-kernel recovery** — not done. Launch a kernel that never completes,
   recover with `Q_CONTROL.reset`, and run a passing test with no JTAG reload.
   This is the capability the whole exercise was for, and it can only be
   demonstrated on hardware.

### 8.1 Risks carried into that build

* **Timing.** Measured on `rst2c`: WNS −0.312 ns, TNS −1570 ns,
  15,349 / 1,214,122 failing endpoints, against gfx2c's −0.261 ns. All ten
  worst paths are in the TCU (`wgmma`/`tbuf`/`bbuf`, the `fedp` pipeline).
  **None involves `rst_stop_req` or the request gate**, so the reset logic did
  not cost closure. Utilisation is essentially unchanged: 648,356 LUTs
  (25.19%), 1,089 RAMB36, 657 DSP.
* **The gate is on a hot path.** `m_axi_mem_awvalid` now passes through
  `VX_afu_req_gate`. It is one AND from a register, but it is in the memory
  request path of every bank. See the timing note above.
* **Counter width.** `COUNT_WIDTH = 10` — the comparison is exact only while
  fewer than 1024 transactions are outstanding on a port. That is far beyond
  what `PLATFORM_MEMORY_ID_WIDTH` (6 bits ⇒ 64 IDs) implies, but it is an
  assumption rather than a proof.

### 8.2 The defect that cost the `rst2c` hardware run

`rst2c` was built with `sp=vortex_afu_0.m_axi_host:HOST`, which routes the CP's
command-ring master to the QDMA slave bridge. Reads through that bridge never
return a response on this compute shell. The CP therefore hung on its very
first ring fetch, and — this is the part worth internalising — **reported
nothing wrong**:

```
Q_CONTROL=0x1  Q_TAIL=0x40  Q_ERROR=0  CP_STATUS=0  head=0 (forever)
```

Every register read back armed and correct. The following run's reset was then
refused with `CTL_RESET_ERROR`, *correctly*, because that read really was still
outstanding. The sequencer was reporting the truth and it was read as a fault
in the sequencer.

The cause was not RTL at all. `platforms.mk` pinned `MEM_TAG` but never
`HOST_TAG`, so the Makefile's `HOST_TAG ?= HOST` won and the working value
survived only on the command line of whoever last built a bitstream — one
build. Fixed by pinning `HOST_TAG = HBM1` beside `MEM_TAG` and adding a
parse-time guard that refuses `HOST_TAG=HOST` outright, so the mistake costs a
second rather than a synthesis run.

The A/B that settled it was two `config.cfg` files already sitting on disk. It
was diagnosed instead by hardware experiments costing three reboots. **Diff the
generated link configuration against the last known-good build before
attributing a hardware symptom to RTL** — it is free, and here it was the whole
answer.

---

## 9. What this changes about how the stack should be read

Three confident explanations for this failure were wrong before the right one:
outstanding AXI transactions, the XDMA decoupler, and multiplier timing. Each
survived because it was plausible and none was tested. What settled it was the
recorded MMIO trace plus the 1 Hz link sample — both of which existed for days
before they were read carefully.

The same pattern then repeated on the follow-up build (§8.2): a hardware
symptom was attributed to new RTL and chased across three reboots, when the
answer was a one-word difference between two `config.cfg` files already on
disk. The lesson did not transfer the first time it was learned, which is
itself the argument for writing it down here.

Four practical consequences:

* **`hw/unittest` is where AFU-level integration logic belongs.** The demux was
  twelve lines of glue inside a 770-line wrapper and no test could reach it. It
  is now a module with a test that fails when it regresses, and so is the reset
  sequencer.
* **"It works on XRT" is not evidence.** The U55C ran the demux defect for as
  long as the CP has existed and never showed it, purely because its driver
  happened to order two register writes the other way round.
* **A test that has never failed has not been validated.** All three new tests
  were checked by re-injecting the original defect and confirming they go red.
  The first attempt at that check was itself wrong — the build failed on lint
  and the stale binary reported a pass — which is the same class of mistake as
  judging a hardware run by its exit code.
* **A test harness that does not instantiate the thing it is testing proves
  nothing.** `afu_reset_seq` drove the drain counters' handshake inputs
  directly and never contained a request gate, so the AXI4 violation in §6.2
  sat outside the DUT entirely and all 33 checks passed over it. The gate is
  inside the harness now. When a module is extracted specifically so it can be
  tested, the extraction is only half the work.
* **Cheap evidence first.** Diffing two generated build configurations costs
  seconds; a hardware bisect costs reboots and a bitstream. §8.2 was the
  second, and should have been the first.
