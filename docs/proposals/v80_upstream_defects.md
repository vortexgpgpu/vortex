# Upstream defect reports — AMD Alveo V80 / SLASH

Three defects found during the Vortex-on-V80 bring-up, each with a minimal
reproduction. Filed here so they can be sent to AMD without re-deriving them.

Environment: Alveo V80 (XCV80), SLASH compute shell, Vivado/Vitis 2025.1,
`v80-smi` / VRT from `/opt/xilinx/slash`, Ubuntu 24.04, kernel 7.0.0-28.

---

## 1. `sp=<kernel>.<port>:HOST` — AXI reads through the QDMA slave bridge never complete

**Severity: high.** A documented, driver-supported feature that does not work.

### Reproduction

Two builds. HLS source byte-identical (`diff -q` verified). One line of
`config.cfg` differs.

| build | `sp=` target | AP_CTRL trace | outcome |
|---|---|---|---|
| `hostprobe` | `HOST` | stuck at `0x1` (`ap_start`) for ~15 min | never completes |
| `hostprobe_hbm` | `HBM0` | `0x4` → `0x1` → `0xe` | `sum=524800` correct, < 0.1 s |

The kernel is a 256-iteration `II=1` accumulate loop that should retire in about
a microsecond. Argument registers were verified correct in the failing case
(`src=0xfea9a000`, `size=0x100`) by reading them back over AXI-Lite.

### Why this is not a misuse

`HOST` resolves to the QDMA slave bridge, sink pin
`/qdma_slave_bridge_noc/S00_AXI` (`slashkit/emit/hw/tcl_gen.py:86`). The kernel
driver ships explicit support for it: `driver/slash_hostbuf.c` allocates
DMA-coherent host memory precisely so a kernel can pull a command ring from it,
and documents that intent in its header comment.

### Impact

Any design whose accelerator masters into host DRAM is blocked. In our case the
Vortex command processor fetches its descriptor ring over this port, so
`TARGET=hw` could not execute at all until the ring was relocated into HBM.

### Note for the reporter

Neither simulator can catch this: emulation shares process memory with the
C-model, and simulation copies host memory into the model over ZeroMQ. Both
bypass the mechanism that fails. A host-mastering path is hardware-only
verification.

**Open question we did not resolve:** whether the slave-bridge aperture requires
an address offset that VRT's `getPhysAddr()` does not apply. Our control
experiment changed the target *and* therefore the address, so it does not
isolate the address dimension. Worth checking before AMD treats this as a pure
fabric defect.

---

## 2. `v80-smi write-static-shell --jtag` cannot recover a card that is off the PCIe bus

**Severity: high.** The documented recovery path has a circular dependency on
the thing it is meant to recover.

### Reproduction

With the card not enumerated (root port absent, `lspci -d 10ee:` empty):

```
$ v80-smi write-static-shell --jtag --shell-type compute -d 01:00 \
      --pdi .../amd_v80_gen5x8_25.1_nofpt.pdi --bash-source ...
Resolving PDI path...
Resolving xsdb Tcl script...
Connecting to VRTD...
Resolving device address...
Resolving VRTD device...
SMI execution failed: Requested resouce doesn't exist
```

`--jtag` resolves the device through VRTD before doing anything, and VRTD only
knows devices enumerated on PCIe. So the JTAG recovery path cannot be used in
the one situation that requires it.

### Workaround

`share/v80-smi/versal_flash_pdi.tcl` has no VRTD dependency and works standalone:

```sh
PDI_PATH=/path/to/amd_v80_gen5x8_25.1_nofpt.pdi xsdb versal_flash_pdi.tcl
```

This succeeded as an unprivileged user with the card off the bus, and a
subsequent reboot enumerated all three functions.

### Suggested fix

`--jtag` should skip VRTD resolution entirely and take the target from the JTAG
chain (or from `V80_TARGET_ID`), as the Tcl already does.

### Related usability issue

Before any of the above, `hw_server` reports an **empty target list** whenever
`ftdi_sio` has claimed the FT4232H interfaces — which it does after every
reboot, producing `/dev/ttyUSB0..3`. That is indistinguishable from a dead card
and sent us down the wrong path for an afternoon. A userspace `USBDEVFS_RESET`
makes udev re-apply the shipped Xilinx cable rules and release the JTAG
interface. Two suggestions:

- Have `v80-smi` detect the held cable and say so, rather than failing opaquely.
- Document that the chain must be checked with `jtag targets`, not `targets` —
  the latter needs a running PLM and is empty on an unconfigured device even
  when the chain is perfectly healthy.

---

## 3. AXI-Lite reads resolve at 16-byte granularity, so a partially decoded block DECERRs entirely

**Severity: medium (documentation).** Not a bug so much as undocumented
behaviour that is invisible in simulation and expensive to discover.

### Observation

An 80-address sweep of a kernel's `s_axi_control` window on the compute shell:

```
RESPOND (16):  0x010 0x014 0x018 0x01C   and   0x100 .. 0x12C
DECERR  (64):  everything else
```

A 16-byte-aligned block answers reads **iff all four of its words are decoded by
the slave**. A partially populated block returns `0xFFFFFFFF` on *every* word,
including the words that are implemented — the per-word decode inside the kernel
is never consulted. 80/80 addresses fit this rule.

```
block 0x000  3/4 implemented (0x0C missing)   -> whole block DECERR
block 0x010  4/4                               -> OK
block 0x020  2/4 (0x20, 0x24 only)             -> whole block DECERR
block 0x100/0x110/0x120  4/4                   -> OK
block 0x130  1/4 (0x130 only)                  -> whole block DECERR
```

Padding the incomplete blocks with registers that read zero fixed it, confirmed
on silicon.

### Impact

The symptom is a register reading `0xFFFFFFFF` on hardware for no reason
visible in the RTL, and it reproduces in no simulator, because the behaviour
belongs to the shell → `s_axi_control` interconnect rather than to the kernel.
It cost ten days to localise.

### Suggested fix

Document the granularity in the platform guide, and ideally have the linker warn
when a kernel's register map leaves a 16-byte block partially populated.

---

## Reporting checklist

- [ ] Attach `hostprobe` / `hostprobe_hbm` (identical HLS source, one `sp=` line differs)
- [ ] Attach the 80-address sweep output and the before/after register dumps
- [ ] Attach the `v80-smi write-static-shell --jtag` transcript with the card off the bus
- [ ] Note the SLASH revision and `v80-smi --version`
