# Proposal: a correct AFU reset architecture

**Status:** proposal — not implemented
**Scope:** `hw/rtl/afu/common/VX_afu_wrap.sv`, `hw/rtl/afu/common/VX_afu_ctrl.sv`,
`hw/rtl/cp/VX_cp_core.sv`, `sw/runtime/aved/vortex.cpp`,
`sw/runtime/common/device.cpp`

---

## 1. Summary

The AFU has **no working reset**. Two defects, and they compound:

1. **`CTL_AP_RESET` resets AXI masters mid-transaction.** On the V80 this
   takes the card off the PCIe bus until it is JTAG-reloaded and the host
   rebooted. The runtime now defaults the write off (`6b41978e7`).
2. **The CP has no software reset.** `q_reset_pulse` is decoded in the regfile
   and discarded in `VX_cp_core` (`UNUSED_VAR`). `Q_CONTROL.reset` and
   `CP_CTRL.reset_all` are no-ops.

Because (1) is unusable and (2) does not exist, **nothing can return the device
to a known state**. The runtime works around this by adopting the CP's
surviving counters at open (`205160014`) — a correct workaround for the command
ring, but it does not reset the Vortex core, and it cannot recover a hung
kernel.

**Yes, this is serious.** A GPU that cannot be reset cannot recover from a hung
kernel, cannot guarantee a clean state between processes, and cannot be made
safe for multi-user or long-running use. Today a kernel that fails to complete
leaves the CP unusable until the FPGA partition is reconfigured — roughly three
minutes of JTAG, and impossible in any deployed setting.

---

## 2. Why `ap_reset` breaks the card

### 2.1 What it actually resets

`VX_afu_ctrl` turns a write of bit 4 to `ap_ctrl` into a one-cycle `ap_reset`
pulse. That reloads a shift register in `VX_afu_wrap`:

```systemverilog
always @(posedge clk) begin
    if (reset || ap_reset) vx_reset_shift_r <= {`VX_CFG_RESET_DELAY{1'b1}};
    else vx_reset_shift_r <= {vx_reset_shift_r[...], 1'b0};
end
wire subsys_reset = reset || vx_reset;
```

`vx_reset` and `subsys_reset` then reset, **simultaneously and with no drain**:

| Block | Line | Contains |
|---|---|---|
| `Vortex_axi` | 421 | the GPU **and its AXI master interfaces** |
| `VX_cp_core` | 302 | the CP **and its `axi_host` / `axi_dev` masters** |
| `bank0_arb` | 558 | the arbiter merging Vortex bank-0 and the CP onto the shared master |

The AXI-Lite *slave* is deliberately left on `reset` alone so it can complete
the write that triggered the sequence. That part is correct.

### 2.2 The defect

**Nothing tracks outstanding AXI transactions.** A search of
`VX_afu_wrap.sv` for `outstanding`, `inflight`, `drain` or any pending counter
returns **zero** matches.

So if either master has a transaction in flight when `ap_reset` fires — a burst
part-way through its `W` beats, or a read awaiting `R` — the master's state
machine is reset out from under it. The AXI protocol is violated: the
interconnect is left waiting for beats that will never arrive, or holding a
response for a transaction the master has forgotten.

This is a protocol violation regardless of platform. The V80 is simply less
forgiving: those masters feed the shell's HBM NoC and QDMA bridge, and a
dangling transaction there wedges the path — after which even the AXI-Lite
slave, which was never reset, becomes unreachable because the shell stops
servicing the partition.

That matches the measurement exactly: with the CP parked and `CP_STATUS`
reading `busy=0`, the write still killed the card, and offset `0x00` — the AFU
control register, not the CP — returned all-ones on the next read.

### 2.3 Why XRT on a U55C survives it

`sw/runtime/xrt/vortex.cpp:206` writes `CTL_AP_RESET` unconditionally in
`init()` and works. Same RTL, same defect. The difference is tolerance: the
XDMA shell's decoupler isolates the reconfigurable partition, and a dangling
transaction is absorbed rather than wedging the NoC.

**This is important framing.** The bug is ours; the U55C hides it. Any platform
with a stricter interconnect will expose it, so "it works on XRT" is not
evidence the reset is correct.

---

## 3. Why the CP reset does not exist

`VX_cp_axil_regfile` decodes `Q_CONTROL.reset_pulse` (bit 1) and drives
`q_reset_pulse[]`. `VX_cp_core` then throws it away:

```systemverilog
// Reset pulse from regfile (Q_CONTROL.reset / CP_CTRL.reset_all) is
// not propagated to CPEs as a separate signal. To stop a queue, the
// host clears Q_CONTROL.enable and the fetch parks in IDLE while
// in-flight commands drain naturally.
`UNUSED_VAR (q_reset_pulse[q])
```

The reasoning is sound as far as it goes — enable-based quiescing is the safe
way to *stop* a queue, and abruptly resetting a CPE with commands in flight
would repeat the §2 mistake. **But stopping is not clearing.** The head and
retire counters keep their values, so nothing returns the queue to seqnum 0,
and a fresh process inherits them. That is the bug the runtime now works around
by resuming from `Q_SEQNUM`.

The register map advertises a capability the hardware does not implement, which
is worse than not offering it: software reads the map and reasonably assumes a
reset exists.

---

## 4. Proposed architecture

The principle: **a master is quiesced before it is reset, and the reset is
observable.** Three pieces, independently useful.

### 4.1 Outstanding-transaction tracking

Add per-master counters in `VX_afu_wrap` for each AXI interface
(`m_axi_mem_*`, and the CP's `axi_host` / `axi_dev` before arbitration):

```
aw_pending += (awvalid && awready)   ;  aw_pending -= (bvalid && bready)
ar_pending += (arvalid && arready)   ;  ar_pending -= (rvalid && rready && rlast)
```

`master_idle = (aw_pending == 0) && (ar_pending == 0) && !w_burst_active`.

Cheap — a handful of counters wide enough for the outstanding limit implied by
`PLATFORM_MEMORY_ID_WIDTH` (6 bits ⇒ 64 IDs).

### 4.2 A reset sequencer

Replace the unconditional shift-register reload with a small FSM:

| State | Action | Exit |
|---|---|---|
| `IDLE` | — | `ap_reset` pulse → `QUIESCE` |
| `QUIESCE` | assert `stop_req` to CP and Vortex: finish current work, issue **no new** AXI requests | all masters idle → `ASSERT` |
| `ASSERT` | drive `vx_reset` / `subsys_reset` for `VX_CFG_RESET_DELAY` cycles | counter expires → `RELEASE` |
| `RELEASE` | deassert; hold `ap_idle` low until the internal reset has propagated | → `IDLE` |
| — | **timeout** in `QUIESCE` → set an error bit, do **not** reset | → `IDLE` |

The timeout matters: if a master will not drain, resetting anyway recreates the
present failure. Refusing and reporting is strictly better than bricking the
card — the software already learned this lesson and refuses to proceed when
`CP_STATUS.busy` will not clear.

`ap_idle` already means "the reset sequence is not in flight"
(`VX_afu_ctrl.sv:307`), so software's existing poll works unchanged; it simply
becomes truthful.

### 4.3 Interface decoupling during reset

While `ASSERT` is active, force each master's outputs to a quiescent state
(`*valid = 0`, `*ready = 1`) so nothing dangles on the shell side even if a
counter is wrong. This is what the XDMA shell's decoupler does for us on a
U55C, and it is what makes the V80 case survivable rather than fatal.

### 4.4 Wire the CP reset

With §4.1–4.3 in place, `q_reset_pulse` becomes safe to implement: on the pulse,
after the CPE's own quiesce, clear `head_r`, `seqnum_r` and the completion
state for that queue. Then:

* `Q_CONTROL.reset` does what the register map says.
* A hung kernel is recoverable without reconfiguring the partition.
* The runtime's resume-from-`Q_SEQNUM` workaround can stay (it is harmless and
  cheap) or be removed.

---

## 5. Software consequences

Once the hardware sequence is correct:

* `VORTEX_AVED_RESET=1` can go back to being the default, because the write
  stops being destructive.
* `cp_quiesce_()` in `device.cpp` stays — quiescing before teardown is good
  practice regardless, and it is what makes `ASSERT` reachable quickly.
* The entry-side quiesce in `sw/runtime/aved/vortex.cpp` `init()` can be
  simplified: the hardware would guarantee what it currently checks by hand.
* Recovery from a hung kernel becomes `Q_CONTROL.reset` instead of
  `jtag_load_vortex.sh`.

---

## 6. Validation plan

Each step must be provable before the next:

1. **Unit-level, simulation.** Force outstanding transactions on a master, pulse
   `ap_reset`, assert that reset is not asserted until the counters reach zero
   and that no AXI beat is dropped. `avedsim` and `xrtsim` both run the AFU
   wrapper, so this is testable with no board.
2. **Timeout path.** Stall a slave so a transaction never completes; assert the
   sequencer times out, reports the error and does **not** reset.
3. **CP reset.** Enable a queue, advance the seqnum, pulse `Q_CONTROL.reset`,
   confirm `Q_SEQNUM` reads 0 and the queue runs again from zero.
4. **On silicon.** Run `minimal` with `VORTEX_AVED_RESET=1` and confirm the card
   stays on the bus — the single measurement that decides whether this worked.
5. **Hung-kernel recovery.** Launch a kernel that never completes, then recover
   with `Q_CONTROL.reset` and run a passing test, with no JTAG reload.

Step 4 is the one that matters. Everything before it is evidence; that is proof.

---

## 7. Risks and alternatives

* **Area and timing.** A handful of counters and a small FSM. Negligible against
  616 k LUTs, but the design already runs ~5% beyond closure, so it must not
  land on a critical path. Keep the counters out of the AXI handshake path.
* **A drain that never finishes.** Handled by the timeout (§4.2). A master that
  cannot drain is already a broken device; the sequencer should say so rather
  than make it worse.
* **Alternative — leave it alone and keep the workarounds.** Viable only while
  the device is single-user and a JTAG cable is reachable. It rules out any
  deployment, and it means every future user rediscovers that `ap_reset` bricks
  the card. The workaround is also incomplete: it never resets the Vortex core.
* **Alternative — a shell-side decoupler.** Would hide the defect the way XRT
  does, but it is AMD's shell, not ours, and it would leave the protocol
  violation in our RTL for the next platform to expose.

---

## 8. Recommendation

Implement §4.1–4.3 together — they are one change and none is useful alone.
Then §4.4, which is small once the sequencer exists.

The cost is modest and bounded: a few counters, one FSM, and a decouple stage.
The return is a device that can be reset, which is a prerequisite for everything
from multi-user access to CI on real hardware. As it stands, the only reliable
way to return a V80 to a known state is to reconfigure the FPGA.
