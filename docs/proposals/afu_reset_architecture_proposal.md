# The AFU reset defect: root cause, fix, and what remains

**Status:** root cause identified and reproduced; RTL fix implemented, not yet
run on silicon
**Scope:** `hw/rtl/afu/common/VX_afu_axil_demux.sv` (new),
`hw/rtl/afu/common/VX_afu_wrap.sv`, `hw/rtl/cp/VX_cp_core.sv`,
`sw/runtime/aved/vortex.cpp`, `hw/unittest/afu_axil_demux/`

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

A second, unrelated defect stands: **the CP still has no software reset**
(§6). That one is real, and the rest of this document's original proposal for
it survives.

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

## 6. The CP still has no software reset

Independent of the above, and still open.

`VX_cp_axil_regfile` decodes `Q_CONTROL.reset_pulse` and drives
`q_reset_pulse[]`. `VX_cp_core.sv:498` throws it away:

```systemverilog
// To stop a queue, the host clears Q_CONTROL.enable and the fetch parks
// in IDLE while in-flight commands drain naturally.
`UNUSED_VAR (q_reset_pulse[q])
```

Enable-based quiescing is the right way to *stop* a queue. **But stopping is
not clearing.** The head and retire counters keep their values, so nothing
returns the queue to seqnum 0 and a fresh process inherits them. That is the
bug the runtime works around by resuming from `Q_SEQNUM` at open
(`205160014`).

The register map advertises a capability the hardware does not implement, which
is worse than not offering it.

### 6.1 Proposed

1. **Outstanding-transaction counters** per AXI master in `VX_afu_wrap`
   (`aw_pending`, `ar_pending`, `w_burst_active`), so `master_idle` is
   observable. A search for `outstanding|inflight|drain` in `VX_afu_wrap.sv`
   still returns zero matches.
2. **A reset sequencer** replacing the unconditional shift-register reload:
   `IDLE → QUIESCE` (stop issuing new AXI requests, wait for idle) `→ ASSERT`
   (`VX_CFG_RESET_DELAY` cycles) `→ RELEASE`. A timeout in `QUIESCE` sets an
   error bit and does **not** reset — refusing is better than resetting a
   master that will not drain.
3. **Wire `q_reset_pulse`**: on the pulse, after the CPE's own quiesce, clear
   `head_r`, `seqnum_r` and the completion state for that queue.

With (3), `Q_CONTROL.reset` does what the register map says and a hung kernel
becomes recoverable without reconfiguring the partition — today that costs
about three minutes of JTAG and is impossible in a deployed setting.

Note that (1) and (2) are now *prudence*, not a fix for an observed failure.
Resetting a master with transactions in flight is a protocol violation whether
or not it is what killed this card, and it is worth closing before the first
kernel actually hangs.

---

## 7. Software consequences

* **`VORTEX_AVED_RESET` should go back to defaulting on** once the fix is
  confirmed on silicon. It is currently off by default
  (`sw/runtime/aved/vortex.cpp:357`) because the write was destructive; with
  the demux fixed, the write reaches `VX_afu_ctrl` and pulses `ap_reset` as
  intended.
* **Do _not_ reorder the aved `init()` to match xrt.** It is tempting — moving
  `CTL_AP_RESET` ahead of the CP quiesce block would sidestep the demux issue
  the way xrt does — but that ordering is the *worse* one. It resets the CP
  and its AXI masters without first parking them, which is the protocol hazard
  §6 exists to close. XRT is not doing this right; it is getting away with it
  because its device is freshly reset by `load_xclbin` and has nothing in
  flight. The aved order (quiesce, confirm `CP_STATUS.busy == 0`, then reset)
  is correct and should stay. Fix the demux, not the caller.
* `cp_quiesce_()` in `device.cpp` stays. Quiescing before teardown is right
  regardless.
* The entry-side error message in `sw/runtime/aved/vortex.cpp` that tells the
  user to JTAG-reload should be revisited once §6 lands, since
  `Q_CONTROL.reset` will be the answer.

---

## 8. Validation plan

1. ~~Reproduce the failure in simulation.~~ **Done** (§3.4).
2. ~~Fix and prove the fix with a test that fails against the defect.~~
   **Done** (§5.1).
3. **Elaborate the wrapper** — `xrtsim` / `avedsim` build with the extracted
   demux, confirming the refactor did not change integration.
4. **Run the aved test suite on `avedsim`** with `VORTEX_AVED_RESET=1`, which
   now exercises the previously fatal path in a model.
5. **On silicon**: rebuild the bitstream, then run `minimal` twice in a row in
   the same boot with `VORTEX_AVED_RESET=1`. The second run is the one that
   used to kill the card. That measurement decides whether this worked;
   everything before it is evidence.
6. **Then** §6, whose validation plan is unchanged: force outstanding
   transactions, pulse `ap_reset`, assert reset is withheld until the counters
   reach zero; stall a slave and assert the sequencer times out rather than
   resetting; and confirm `Q_CONTROL.reset` returns `Q_SEQNUM` to 0.

Step 5 requires a synthesis run and a JTAG reload, so it is the user's call
when to spend it.

---

## 9. What this changes about how the stack should be read

Three confident explanations for this failure were wrong before this one:
outstanding AXI transactions, the XDMA decoupler, and multiplier timing. Each
survived because it was plausible and none was tested. What settled it was the
recorded MMIO trace plus the 1 Hz link sample — both of which existed for days
before they were read carefully.

Two practical consequences:

* **`hw/unittest` is where AFU-level integration logic belongs.** The demux was
  twelve lines of glue inside a 770-line wrapper and no test could reach it. It
  is now a module with a test that fails when it regresses.
* **"It works on XRT" is not evidence.** The U55C ran this defect for as long
  as the CP has existed and never showed it, purely because its driver happened
  to order two register writes the other way round.
