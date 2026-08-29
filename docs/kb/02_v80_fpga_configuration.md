# AMD Alveo V80 — FPGA Configuration Knowledge Base

**Scope.** How an Alveo V80 gets configured: the silicon's boot architecture,
the flash layout, every path by which a design can reach the device, the two
shells that SLASH uses, and the recovery procedures for each way it can end up
unusable. Sections marked **[field]** are measured on this machine
(`orcas2`, root port `0000:00:01.1`) during the Vortex bring-up, not vendor
documentation.

---

## 1. The board

| Property | Value |
|---|---|
| Device | AMD Versal HBM **XCV80** adaptive SoC |
| Logic | 2.6 M LUTs, 10,848 DSP slices, 132 Mb BRAM, 541 Mb URAM |
| HBM | 32 GB HBM2e (2 × 16 GB stacks), ~820 GB/s peak |
| DDR | 8 GB DDR4 onboard + DIMM slot supporting up to 32 GB |
| Host interface | PCIe Gen4 ×16 **or dual Gen5 ×8** |
| Networking | 4 × QSFP56 (4 × 200 Gb/s) |
| Expansion | 3 × MCIO (fly-over PCIe5) |
| Config storage | 2 Gb (256 MB) OSPI flash, ×8, 200 MHz |
| Secondary storage | eMMC0, 64 GB, 8-bit |
| Debug | Onboard USB-JTAG |

The device is a *SoC*, not a plain FPGA. Configuration is orchestrated by
firmware running on hardened processors, which is why "programming a V80" is a
software protocol rather than a bitstream shift.

---

## 2. Versal configuration architecture

### 2.1 The players

- **PMC (Platform Management Controller)** — the block that owns boot. Contains
  the BootROM and the PPU (a MicroBlaze-class processor) that runs the PLM.
- **BootROM** — immutable. Reads the boot-mode pins, finds the PDI on the
  selected boot device, validates it, and loads the **PLM** into PPU RAM.
- **PLM (Platform Loader and Manager)** — the software that does everything
  else: decodes and processes the rest of the PDI, loads each partition, and
  then stays resident for run-time platform management, error management,
  partial reconfiguration, and subsystem restart. The PLM can reload images and
  load *partial* PDIs at run time.
- **CIPS** — the Vivado IP block that configures the PS, PMC, and CPM. This is
  where a design declares its PCIe controller mode, BAR layout, boot mode,
  peripherals, and clocks.
- **CPM5** — the hardened PCIe/CCIX module. Two PCIe controllers, two QDMA/bridge
  subsystem instances, high-bandwidth AXI into the NoC.
- **NoC** — the programmable network-on-chip. Every master-to-slave path
  (PCIe→HBM, PMC→DDR, RPU→DDR, PL→anything) is a NoC route with an assigned
  address aperture and bandwidth class.
- **RPU** — two Arm Cortex-R5F cores. On AVED these run the **AMC** management
  firmware.
- **APU** — two Arm Cortex-A72 cores. Unused by the AVED base design.

### 2.2 PDI — Programmable Device Image

A PDI is not a bitstream. It is a *container* of partitions: PLM code, PMC data,
CDO (configuration data object) sequences that program the NoC and memory
controllers, PL fabric configuration, and optionally ELF images for the RPU/APU.
The BootROM understands enough of it to extract the PLM; the PLM understands the
rest.

**Unique Identifiers (UIDs)** are embedded in the PDI. The PLM checks them when
a subsequent (partial) image is delivered, so an incompatible PL image is
rejected rather than loaded into a shell it does not match. This is the
mechanism behind "this vbin does not match the loaded shell" errors.

### 2.3 Segmented configuration — and why it governs enumeration

PCIe requires an endpoint to be ready for link training within ~120 ms of
PERST# release. A full V80 PDI takes on the order of **13.5 s** to load from
OSPI. **[field]**

The resolution is **segmented configuration**: the boot image is structured much
like a Tandem PROM image, so that CPM and the other elements needed for link-up
are programmed and released *first*, followed by the remainder of the design.
Vivado does this by default for CPM5 designs when one or more controllers are in
endpoint mode.

Three consequences that dominate day-to-day V80 work:

1. A design *not* built with segmented configuration will miss the window and
   the card will not enumerate after a warm reboot.
2. **A configuration already resident in the fabric survives PERST#.** This is
   why loading the shell over JTAG and *then* rebooting recovers a card that a
   plain reboot cannot: the fabric is already configured when the reset arrives,
   so it wins the race trivially. **[field]**
3. If the card misses the window, platform firmware does not instantiate the
   root port at all — see §8.

---

## 3. Boot modes

The Versal boot mode is selected by pins, but is *overridable at run time* via a
boot-mode register that JTAG can write. The V80's pins select OSPI.

Reading the current mode over JTAG:

```tcl
xsdb% connect
xsdb% ta 1
xsdb% device status jtag_status
```

Boot mode is bits **[15:12]** of `jtag_status`:

| Value | Mode |
|---|---|
| `0000` | JTAG |
| `1000` (0x8) | OSPI |

Switching to JTAG boot mode, using the script shipped in the AVED deployment
archive:

```tcl
xsdb% source flash_setup/versal_change_boot_mode.tcl
xsdb% device status jtag_status      # confirm bits [15:12] now read 0000
```

After this, Vivado Hardware Manager will accept JTAG programming operations on
the device.

> The upstream documentation states the procedure but does not document what the
> script writes, nor how to restore OSPI mode. Empirically, a **cold power
> cycle** restores the pin-selected mode, which is the reason the AVED
> bootstrap procedure ends with a full shutdown rather than a `reboot`.

---

## 4. OSPI flash layout

### 4.1 Flash Partition Table (FPT)

The first page of the 2 Gb OSPI holds the AVED **Flash Partition Table**, which
lets several images coexist:

**FPT header**

| Field | Notes |
|---|---|
| Magic word | `0x92F7A516` |
| Version | FPT format version |
| Header size | bytes |
| Entry size | bytes per entry |
| Entry count | number of partitions |

**FPT entry** — type (PDI), base address, partition size.

The baseline AVED implementation defines **two partitions**, giving resilient
boot with fallback.

Constraints:

- PDI files must be aligned to **32 KB boundaries** in the OSPI address space.
- A partition must be sized to hold the whole PDI.
- FPT initialisation is normally a one-time operation; changing it requires
  reprogramming over JTAG via Vivado.

### 4.2 Boot behaviour

**Cold boot / power-on reset.** The PMC reads address 0 and finds the FPT
header, whose signature is not a valid PDI signature, so it skips it and loads
the design from the first partition.

**Run time / multiboot.** Versal's multiboot procedure allows reloading from a
*different* partition: write the target offset into the **`PMC_MULTIBOOT`**
register and trigger a system reset. The BootROM restarts, sees a non-zero
multiboot offset, and searches from there. On the V80 this is driven by AMC
commands issued via `ami_tool` / the AMI driver
(`AMI_IOC_DEVICE_BOOT(partition=N)`).

**Fallback.** If the image at the multiboot offset fails validation, the
BootROM increments the multiboot offset and searches the next 32 KB-aligned
address — which is the reason for the 32 KB alignment rule. A "golden" image
placed at a higher offset therefore acts as the last-resort fallback.

---

## 5. CIPS / CPM5 configuration in the AVED base design

The AVED base design is the reference every V80 platform (including SLASH's
shells) is derived from.

### 5.1 CPM5

- Uses **PCIe Controller 1** in **QDMA mode**.
- **Gen5 ×8** — 32 GT/s raw. The design uses 8 lanes, and *"to be in PCI SIG
  compliance, the bottom 8 lanes must be used."*
- Advanced mode with extended large configuration space enabled. The **Extended
  Config Interface** is wired to the Hardware Discovery IP so vendor-specific
  discovery metadata is readable from extended config space.

### 5.2 Physical functions and BARs (AVED baseline)

| PF | Device ID | Class | BAR | Purpose |
|---|---|---|---|---|
| PF0 | `0x50B4` | `12/00` Processing Accelerator | one 64-bit BAR0, 256 MB | Management. AXI bridge master into PL memory space |
| PF1 | `0x50B5` | — | DMA BARs, 8 MSI-X vectors | QDMA data path to DDR and HBM |

PF0's BAR0 maps to PCIe address range
**`0x201_0000_0000` – `0x201_0FFF_FFFF`** (256 MB), translated into PL memory
space. Only a subset of it is used by the base design; the remainder is
available for a design's own registers — which is exactly how a compute shell
publishes its kernel AXI-Lite maps.

> SLASH's compute shell extends this to **three** PFs by adding PF2 (`0x50C2`)
> for BAR MMIO and moving QDMA to `0x50C1`. See KB-3 §3.

### 5.3 PS / PMC peripherals

| Function | Configuration |
|---|---|
| Primary boot | OSPI, 2 Gb / 256 MB, 200 MHz |
| Secondary storage | eMMC0, 64 GB, 8-bit |
| RPU | 2 × Cortex-R5F (runs AMC firmware) |
| APU | 2 × Cortex-A72 |
| UART | UART0 / UART1 on PS MIO — **the primary firmware debug channel** |
| I²C | LPD_I2C0 (power/clock/temperature), LPD_I2C1 (QSFP management) |
| SPI | SPI0 for logging flash |
| RPU→NoC | 128-bit @ 800 MHz |
| PMC→NoC | 128-bit @ 400 MHz |

Clock outputs to PL:

| Output | Frequency | Use |
|---|---|---|
| PL CLK 0 | 100 MHz | AXI interfaces |
| PL CLK 1 | 33.3333 MHz | free-running system clock |
| PL CLK 2 | 250 MHz | PCIe extended configuration interface |

I/O standards: bank 0/3 LVCMOS 1.8 V; banks 1 and 2 LVCMOS 3.3 V.

---

## 6. NoC configuration and the address map

The Versal global address map spans 16 TB and is realised entirely by the NoC.
Masters attach through **NMUs** (NoC Master Units) and slaves through **NSUs**.

| Unit | Data width | Use |
|---|---|---|
| `NMU_512` | 32–512 bit | general-purpose masters |
| `NMU_128` | 128 bit fixed | low-latency, hardened blocks such as CIPS |
| `HBM_NMU` | 32–256 bit | direct HBM access |
| `NSU_512` | 32–512 bit | general-purpose slaves |
| `NSU_128` | 128 bit fixed | low-latency slaves |
| `DDRMC_NSU` | — | DDR memory-controller converter |
| `HBM_NSU` | — | HBM controller converter |

### 6.1 AVED apertures

| Region | IP | Address range | Size |
|---|---|---|---|
| DDR LOW0 | `axi_noc_mc_ddr4_0` | `0x000_0000_0000` – `0x000_7FFF_FFFF` | 2 GB |
| DDR CH1 | `axi_noc_mc_ddr4_0` | `0x500_8000_0000` – `0x500_FFFF_FFFF` | 2 GB |
| DIMM / DDR CH2 | `axi_noc_mc_ddr4_1` | `0x600_0000_0000` – `0x67F_FFFF_FFFF` | 32 GB |
| HBM | integrated in `axi_noc_cips` | — | 32 GB across 16 channels |
| PL / PF0 BAR0 | `axi_noc_cips` | `0x201_0000_0000` – `0x201_0FFF_FFFF` | 256 MB |
| QDMA registers | CPM5 | `0x6_1000_0000` | — |
| PCIe Bridge registers | CPM5 | `0x6_0000_0000` | — |

VRT's fake-address scheme for non-hardware platforms deliberately mirrors this
shape: HBM allocations start at `0x40_0000_0000` and DDR at `0x600_0000_0000`,
so a pointer printed in simulation is recognisable as the same kind of thing it
would be on hardware.

### 6.2 Connectivity and bandwidth classes

`axi_noc_cips` routes: PCIe host → PL management, DDR, DIMM, HBM; PMC → DDR,
DIMM; RPU → DDR.
`axi_noc_mc_ddr4_0`: `S00_INI` carries PCIe/PMC/RPU to MC port 0; `S01_INI`
carries PCIe to MC port 1.
`axi_noc_mc_ddr4_1`: `S00_INI` PCIe/PMC to MC port 0; `S01_INI` PCIe to MC
port 1.

All paths use the **Best Effort** traffic class, with declared bandwidths:

| Path | Declared bandwidth |
|---|---|
| AXI4-Lite (`M0x_AXI`) | 5 MB/s |
| DDR / DIMM INI (`M0x_INI`) | 800 MB/s |
| HBM pseudo-channels | 250 MB/s each |

These are *NoC arbitration hints*, not caps on achievable throughput, but a path
starved in the NoC compiler will underperform regardless of the memory
controller behind it.

> **[field] Both PCIe NMUs must be exercised to get full bandwidth.** The SLASH
> driver picks a NoC channel per queue pair (`mm_channel`: `auto` stripes by
> `qid & 1`, or pin to `0`/`1`). If a split run is no faster than a single
> forced channel, traffic is not actually spreading across both NMUs:
> ```sh
> sudo v80-smi validate -d <BDF> --raw-transfer-test --no-reset --mm-channel 0
> sudo v80-smi validate -d <BDF> --raw-transfer-test --no-reset --mm-channel 1
> sudo v80-smi validate -d <BDF> --raw-transfer-test --no-reset --mm-channel auto
> ```

---

## 7. Configuration paths

There are four ways a design reaches a V80, in decreasing order of privilege and
increasing order of convenience.

### 7.1 JTAG → fabric (volatile)

Vivado Hardware Manager, or `xsdb`, loads a PDI directly into the device. Does
not touch flash. **Survives PERST#** but not a power cycle. This is the recovery
path when the card is not on the PCIe bus, and the only path when the AMC is
wedged.

### 7.2 JTAG → OSPI (persistent)

Vivado Hardware Manager with a configuration memory device:

1. Switch to JTAG boot mode (§3).
2. *Open Hardware Manager* → *Open Target* → *Auto Connect*; the card appears as
   `xcv80_1`.
3. Right-click `xcv80_1` → *Add Configuration Memory Device* → select part
   **`cfgmem-2048-ospi-x8-single`**.
4. Program with `flash_setup/fpt_setup_<vbnv>_<release>.pdi` together with
   `flash_setup/v80_initialization.pdi`; address range **Entire Configuration
   Memory Device**.
5. Wait for *Flash Programming Completed Successfully*.
6. **Full power cycle** (`sudo shutdown -h now`, then power on). A soft `reboot`
   does not re-read the boot-mode pins.

This is the one-time bootstrap for a brand-new board, or the recovery when OSPI
is corrupt and PF0 no longer enumerates.

### 7.3 PCIe → OSPI, via AMC

In deployment, the AMC firmware receives PDI data over PCIe: the host writes the
image into a DDR buffer that is mapped into PF0's BAR, and the AMC writes it
into the target OSPI partition. `ami_tool` drives this from the host, and
`v80-smi write-static-shell --flash` is SLASH's wrapper for writing its static
shell this way.

Requires: `ami` bound to PF0, AMC in `READY` state.

### 7.4 PCIe → fabric, via the runtime

`v80-smi program my_design.vbin -d 03:00`, or implicitly
`vrt::Device device("03:00", "my_design.vbin")`. This extracts the PDI from the
vbin and loads it into the fabric through vrtd. It is the normal
developer-iteration path, and it is the one with the sharp edge documented in
§9.1.

---

## 8. Enumeration failures

### 8.1 The missing root port **[field]**

If the card is not link-trained when platform firmware enumerates at POST,
firmware does not instantiate the root port. On this host:

- `0000:00:01.1` is **absent** from `/sys/bus/pci/devices/`.
- Bus numbers shift, so `00:07.1` (AMD `1022:1556`) takes bus 01.
- `lspci -d 10ee:` is empty.
- **`echo 1 > /sys/bus/pci/rescan` can never help** — there is no bridge below
  which to rescan.

Diagnosis, first command every time:

```bash
ls /sys/bus/pci/devices/0000:00:01.1 || echo "root port ABSENT — rescan is futile"
```

Recovery: JTAG-load the shell into the fabric (§7.1), *then* warm-reboot. The
resident configuration survives PERST# and wins the POST race. A cold power
cycle also works but loses the JTAG-loaded design.

### 8.2 AMC not ready

`ami_tool overview` reports `NO_AMC` instead of `READY`. Causes:

- A failed PDI design write (see §9.1) drove the AMC into this state.
- The RPU firmware crashed; dmesg shows `AMC Heartbeat expired event received`.

Recovery requires **JTAG**, not PCIe: `AMI_IOC_DEVICE_BOOT` reaches the AMC over
GCQ, and a wedged AMC cannot service its own recovery request.

### 8.3 Progressive recovery ladder

AMD's documented escalation, cheapest first:

```
AMI driver reload
  → PCIe hot reset (ami_tool / v80-smi reset)
    → host PCI remove + rescan
      → server warm reboot
        → cold power cycle
```

**[field]** On this platform, insert one more rung between "hot reset" and
"warm reboot": **JTAG shell load**. And note that steps 2–4 all involve a link
transition, which on this host has twice hard-reset the machine (§9.3).

---

## 9. Shells, and the two configuration hazards

### 9.1 One design write per device reset **[field]**

SLASH's vrtd performs `reset_with_ami` **only when the requested shell differs
from the currently reported shell** (`vrt/vrtd/src/flash_worker.c:273`).
Measured 2026-08-18:

```
20:29:58  design_write: shell switch required current=0 required=2
20:29:59  reset_with_ami: AMI_IOC_DEVICE_BOOT(partition=1) OK
20:29:59  removed 01:00.0/.1/.2  →  SBR  →  rescan
20:30:18  Design write completed successfully            ← load #1 OK
20:33:53  Design write submitted (3m35s later)
20:34:03  Failed to transfer design writer payload: Input/output error
```

Once the card reports `Shell: compute`, no further reset ever occurs and **every
subsequent design write fails** — and a failed write takes the AMC to `NO_AMC`,
which costs a JTAG recovery.

**Rule: a PDI design write succeeds only on a freshly reset device.** Waiting
does not substitute for a reset.

**Workaround that makes iteration affordable:**

```cpp
vrt::Device device(bdf, vbin, /*program=*/false);   // skip the PDI load
```

Program once, then run every subsequent test against the resident design. In
this project that is wired to `VORTEX_AVED_NO_PROGRAM=1` (runtime) and
`VRT_NO_PROGRAM=1` (probe utilities).

### 9.2 `Shell: unknown` after a JTAG load **[field]**

A JTAG fabric load leaves the AMC unable to identify what is loaded, so
`v80-smi list` reports `Shell: unknown`. That is *not equal* to the requested
shell, so the next design write **does** trigger a shell switch — which triggers
`reset_with_ami`, which triggers an SBR. See §9.3.

### 9.3 The SBR hazard **[field]**

Twice on 2026-08-19, a `TOGGLE_SBR` on root port `0000:00:01.1` hard-reset the
host. The journal for the dying boot ends mid-sequence with no shutdown record
and no MCE:

```
09:40:27  slash_hotplug: toggle_sbr: bridge=0000:00:01.1 bus=01
09:40:27  pcieport 0000:00:01.1: unlocked secondary bus reset via: slash_hotplug_ioctl
09:40:28  vrtd: reset_with_ami: SBR toggle complete for 0000:01:00.0
09:40:28  slash_hotplug: toggle_sbr: post-SBR settle complete (1000 ms)
09:40:28  slash_hotplug: ioctl: TOGGLE_SBR succeeded         ← log ends
```

For contrast, a deliberate reboot ends with `systemd-shutdown: Shutting down`.
The same SBR succeeded three times in an earlier boot, so this is a race, not a
deterministic fault.

Root cause is a firmware policy: a fatal PCIe error on this root port is
escalated to a platform reset rather than logged. Mitigations, in order:

1. **BIOS** — set PCIe uncorrectable/fatal error severity to non-fatal, or
   disable "System Error on PCIe fatal". On AMD/AMI boards look under
   *AMD CBS → NBIO*, or *PCIe AER Support* / *System Error Severity*.
2. **Unbind everything before any link transition.** AMD says so in its own
   source, `vrt/vrtd/src/reset.c:334`: *"If any function remains bound while the
   bus is reset, the kernel may attempt MMIO or config-space accesses to a
   device whose link is down, which can cause machine checks or system hangs."*
3. **Avoid the reset**: get the shell to `compute` once, then never program
   again in that session.

### 9.4 The recovery cycle to avoid

```
warm reboot  →  card loses POST race  →  root port absent
      ↑                                        ↓
      │                              JTAG-load shell to fix it
      │                                        ↓
      │                              firmware now reports Shell: unknown
      │                                        ↓
      └──── host hard reset ←── SBR ←── first design write needs a shell switch
```

Break it at the third step: `program=false`.

---

## 10. Verification and field procedures

### 10.1 Post-boot health check

```bash
# 1. Root port present?
ls /sys/bus/pci/devices/0000:00:01.1

# 2. All three functions enumerated?
lspci -d 10ee: -nn        # expect 50b4 (PF0), 50c1 (PF1), 50c2 (PF2)

# 3. Link negotiated as expected?
lspci -vv -s 01:00.0 | grep -E 'LnkCap|LnkSta'
ami_tool pcieinfo -d 01:00

# 4. Card firmware alive?
sudo ami_tool overview            # expect state READY, not NO_AMC

# 5. Drivers bound and daemon aware?
v80-smi list                      # PF0 / PF1 / PF2 / VRTD must all pass
```

### 10.2 Memory and DMA validation

```bash
v80-smi validate -d 01:00          # HBM + DDR integrity (pattern i ^ seed) and H2C/C2H bandwidth
v80-smi validate -d 01:00 -j 16    # 16 threads (default 8, max 64)
```

### 10.3 Inspecting what will be loaded

```bash
v80-smi inspect my_design.vbin     # platform, clock, kernels, args, memory connections
tar tzf my_design.vbin             # raw archive contents
v80-smi query -d 01:00             # what YOU last wrote to this BDF
```

`query` carries an explicit caveat in the upstream docs: it reports what the
current user last wrote, not what is physically loaded. There is no way to read
the live design back off the card. Treat it as a hint.

### 10.4 Firmware-level debug

- **UART on PS MIO** is the AMC's own console and is the only channel that
  survives a wedged GCQ.
- `ami_tool debug_verbosity -d <BDF> -l debug` raises AMC message verbosity.
- `echo 1 > /sys/bus/pci/drivers/ami/ami_debug_enabled` raises host driver
  verbosity into dmesg.
- `ami_tool bar_rd` / `bar_wr` for raw register poking when nothing else works.
- The reported `logic_uuid` from `ami_tool overview` should match the UUID in
  the AVED archive's `version.json` — a mismatch means the flashed image is not
  what you think it is.

---

## 11. Bring-up checklist for a fresh board

- [ ] USB-JTAG cable connected; Vivado 2025.1 installed and sourced.
- [ ] AVED deployment archive downloaded (xilinx.github.io/AVED,
      xilinx.com/member/v80.html).
- [ ] `xsdb` → `connect` → `ta 1` → `device status jtag_status` shows OSPI
      (`1000`).
- [ ] `source versal_change_boot_mode.tcl`; re-check → `0000` (JTAG).
- [ ] Hardware Manager: add `cfgmem-2048-ospi-x8-single`, program
      `fpt_setup_*.pdi` + `v80_initialization.pdi`, entire device.
- [ ] **Cold power cycle**, not a reboot.
- [ ] `lspci -d 10ee:50b4` shows one entry per board.
- [ ] Install SLASH stack; `ami` binds PF0.
- [ ] `sudo ami_tool overview` → `READY`, `logic_uuid` matches `version.json`.
- [ ] `v80-smi write-static-shell --flash` to install the SLASH static shell.
- [ ] `v80-smi list` → PF0 / PF1 / PF2 / VRTD all pass.
- [ ] `v80-smi validate -d <BDF>` passes.

---

## Sources

- [AVED — Device Programming](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_2_20240408/AVED+-+Device+Programming.html)
- [AVED V80 — CIPS Configuration](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_2_20240408/AVED+V80+-+CIPS+Configuration.html)
- [AVED V80 — NoC Configuration](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_2_20240408/AVED%2BV80%2B-%2BNoC%2BConfiguration.html)
- [AVED JTAG Boot Recovery](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_1_20231204/AVED+JTAG+Boot+Recovery.html)
- [AVED Debug Techniques](https://xilinx.github.io/AVED/amd_v80_gen5x8_24.1_20241002/AVED+Debug+Techniques.html)
- [AMI — Hot Reset](https://xilinx.github.io/AVED/amd_v80_gen5x8_24.1_20241002/AMI+-+Hot+Reset.html)
- [AVED — Host to Card Communication](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_2_20240408/AVED+-+Host+to+Card+Communication.html)
- [AMD Alveo V80 product page](https://www.amd.com/en/products/accelerators/alveo/v80.html)
- [Alveo V80 Data Center Accelerator Card Installation Guide](https://www.soniccomponents.com/wp-content/uploads/2025/08/1081285820.pdf)
- [Versal Platform Loader and Manager — AMD Wiki](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/2037088327/Versal+Platform+Loader+and+Manage)
- [Vivado Design Tutorials — Versal Segmented Configuration (2025.1)](https://github.com/Xilinx/Vivado-Design-Tutorials/tree/2025.1/Versal/Boot_and_Config/Segmented_Configuration)
- [Boot and Configuration — Embedded Design Tutorials](https://xilinx.github.io/Embedded-Design-Tutorials/docs/2023.1/build/html/docs/Introduction/Versal-EDT/docs/4-boot-and-config.html)
- [Image Selector (ImgSel) Utility — AMD Wiki](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/2662138473/Image+Selector+ImgSel+Utility)
- [MicroZed Chronicles: Versal Address Map and DDR Memory Controller](https://www.adiuvoengineering.com/post/microzed-chronicles-versal-address-map-and-ddr-memory-controller)
- [PCIe_CPM Lab 2 — QDMA AXI MM to NoC and DDR](https://github.com/Xilinx/PCIe_CPM/blob/main/docs/Lab2/Lab2.md)
- Local: `~/dev/SLASH-compute/docs/tutorials/admin/bootstrap-aved.rst`,
  `docs/tutorials/admin/device-management.rst`, `driver/README.md`
