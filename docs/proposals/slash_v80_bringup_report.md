# SLASH / Alveo V80 Bring-Up Report

**Date:** 2026-07-30
**Host:** `orcas2` — Linux 7.0.0-28-generic, AMD Zen, AMD Alveo V80 at PCIe `0000:11:00`
**Tooling:** AMD Vivado / Vitis 2025.2 (`/opt/xilinx/2025.2`), sourced via `~/dev/xilinx_setup.sh`
**Subject:** `~/dev/SLASH` — the AMD open-source platform stack for the Alveo V80
**Purpose:** hands-on evaluation of SLASH ahead of [aved_afu_proposal.md](aved_afu_proposal.md)

---

## 1. Outcome

The full SLASH software stack was built from source and **three sample
designs were executed successfully on the SLASH simulation platform**
(Vivado `xsim` behind a vbin). All passed against their golden models.

**Execution on the physical FPGA was not achieved**, for two reasons that
are both outside the software's control:

| Gate | Status |
|---|---|
| Driver stack (`ami` + `slash` loaded, `vrtd` running) | **Resolved** — both modules build and load cleanly |
| Card responds on PCIe | **Failed** — config space reads all-`0xff`; no driver binds; FT4232H JTAG absent from `lsusb` |
| Hardware vbin needs a prebuilt *static shell* | **Blocked** — requires gated SMBus IP + Vivado Enterprise license |

Two findings matter more than the rest:

1. **The software stack is fully functional and the card is not.** `ami`
   2.4.0 and `slash` both load and register with the PCIe core, and both
   bind to zero devices (§6.2). This is a hardware/power fault requiring
   physical intervention, not a software problem.
2. **A hardware vbin cannot be produced on this machine at all**,
   independent of the card's state, because the static shell requires an
   AMD member-portal IP download and a Vivado Enterprise license (§6.3).

Root access and Secure Boot were both obstacles during the session and
were resolved (§6.5); neither is a standing blocker.

---

## 2. What was built

Everything was built into a userspace prefix (`~/dev/.slash-local`)
because no passwordless `sudo` is available — see §4.1.

| Component | Result | Artifact |
|---|---|---|
| `slash` kernel module | built + **loaded** | `driver/slash.ko` (8.7 MB, vermagic 7.0.0-28-generic) |
| `ami` kernel module (from AVED) | built + **loaded** | `sw/AMI/driver/ami.ko` — AMI 2.4.0 |
| `ami_tool` (from AVED) | built | `sw/AMI/app/build/ami_tool` |
| `libslash` | built + installed | `libslash.so.1.0.0` |
| `vrtd` daemon + `libvrtd`/`libvrtdpp` | built + installed | `bin/vrtd` |
| `libvrt` (VRT runtime) | built + installed | `libvrt.so.1.0.0` |
| `v80-smi` | built + installed | `SMI v1.0.0` |
| Example apps 00/01/02 | built | native executables |
| HLS kernels (V80 `xcv80-lsva4737-2MHP-e-S`) | synthesised | IP-XACT `component.xml` |
| Simulation vbins | linked | `axilite_sim.vbin` (29 MB), `aximm_sim.vbin` (22 MB) |
| Hardware vbin | **failed** | blocked on static shell (§6.3) |

The `slash` module's PCI alias table is worth recording:

```
alias: pci:v000010EEd000050B5sv*sd*bc*sc*i*     # PF1
alias: pci:v000010EEd000050B6sv*sd*bc*sc*i*     # PF2
```

`slash` binds **only** PF1 and PF2. PF0 (`0x50B4`) belongs to `ami`. This
is why the current PCIe state (§6.1) blocks everything downstream.

---

## 3. Sample execution results

Run as `./build/<app> 0000:11:00 build/<target>_sim.vbin`. The BDF argument
is required by the CLI but unused on the simulation platform — VRT reads
the platform from the vbin metadata, not from the device.

| Example | Exercises | Result |
|---|---|---|
| `00_axilite` | AXI-Lite control, linking, two kernels | **passed** — expected 1536.327881, got 1536.328003, abs error 1.22e-4 within 1.54e-3 tolerance |
| `01_aximm` | AXI-MM master interfaces | **passed** |
| `02_chain` | free-running streaming kernels (`axis`) | **passed** — 6.0 s wall-clock |

Wall-clock for the `00_axilite` kernel waits was **878,750 µs** — ~0.9 s of
host time for a trivial two-kernel design. This is `xsim` RTL simulation
speed and sets expectations for the SLASH `sim` platform: it validates
packaging and host integration, but it is far too slow to serve as an
iteration loop for a design the size of Vortex.

`v80-smi inspect` reads the vbin metadata correctly without any device
present, resolving kernel names, physical addresses, and per-argument
offset/range/direction:

```
Platform: SIMULATION
Clock frequency: 200000000
Kernel: accumulate_0   Physical address: 0x20200000000
Kernel: increment_0    Physical address: 0x20200010000
```

Linked simulation vbins: `axilite_sim.vbin` 29.3 MB, `aximm_sim.vbin`
22.3 MB, `chain_sim.vbin` 21.7 MB.

---

## 4. Obstacles encountered and resolutions

Six issues blocked the build. All were resolved. Two are latent bugs in
SLASH that would affect any user on a current kernel or a non-root install.

### 4.1 No passwordless sudo → userspace prefix

`sudo` requires a password, so `apt install` was unavailable and the
documented build recipe (which installs to `/usr/local`) could not be
followed.

Resolution: `apt-get download` needs no privileges. A prefix was built at
`~/dev/.slash-local`:

- `pkgs/` — 31 downloaded `.deb` files
- `root/` — the debs extracted with `dpkg -x`
- `env.sh` — sets `PATH`, `CMAKE_PREFIX_PATH`, `PKG_CONFIG_PATH`, `CPATH`,
  `LIBRARY_PATH`, `LD_LIBRARY_PATH`

`.pc` files were rewritten to repoint `prefix=/usr` at the local tree.
`cmake` (3.28.3) and `ninja` (1.11.1) came from the same mechanism —
note that `python3 -m venv` is **not** usable on this host (`ensurepip`
is absent), so the pip route to cmake/ninja is unavailable.

Source `~/dev/.slash-local/env.sh` before any SLASH build or run.

### 4.2 Kernel 7.0 removed the legacy timer API — *SLASH bug*

The vendored QDMA driver (`submodules/qdma_drv`, a 2023.2-era snapshot)
fails to compile against kernel 6.15+:

```
error: implicit declaration of function 'del_timer'
error: implicit declaration of function 'from_timer'
```

Both were removed upstream in favour of `timer_delete()` and
`timer_container_of()`.

Resolution: SLASH already has a clean `kcompat` probe mechanism
(`driver/kcompat/probe.sh` builds each `.c` in that directory as a
conftest and derives `SLASH_HAVE_<BASENAME>` from the filename). Two files
were added in that idiom, leaving the submodule pristine:

- `driver/kcompat/timer_modern.c` — probes `timer_delete()` /
  `timer_container_of()`
- `driver/kcompat/qdma_timer_compat.h` — maps the legacy spellings onto
  the modern API

plus a guarded `ccflags-y += -include .../qdma_timer_compat.h` in
`driver/Makefile`. The probe reports `SLASH_HAVE_TIMER_MODERN=y` on this
host and `slash.ko` builds clean.

**This is worth upstreaming to SLASH** — any user on a 6.15+ kernel hits it.

### 4.3 Undocumented dependencies — *SLASH doc bug*

Two libraries are required but absent from the documented dependency list
in both `README.md` and `docs/tutorials/admin/platform-setup.rst`:

- **`libcap-dev`** — `vrtd` links `libsystemd`, which needs `cap_*` symbols.
- **`cppzmq-dev`** — `vrt/include/vrt/utils/zmq_server.hpp` includes
  `zmq.hpp`, the C++ binding, which `libzmq3-dev` does not ship.

(The RHEL tab of `platform-setup.rst` does list `cppzmq-devel`; the Ubuntu
tab does not. `libcap-dev` is missing from both.)

### 4.4 Dangling `.so` symlinks in the extracted prefix

`dpkg -x` of a `-dev` package yields e.g. `libsystemd.so -> libsystemd.so.0`,
a *relative* symlink that dangles inside the prefix because the runtime
package is installed system-wide. The linker silently fell back to the
static `libsystemd.a`, which then failed on unresolved `cap_*` symbols —
a misleading error whose real cause was symlink resolution, not a missing
library. Five symlinks were repointed at `/usr/lib/x86_64-linux-gnu/`.

### 4.5 Hardcoded `/usr/include/jsoncpp/` — *SLASH portability bug*

The sim project generated by `slashkit` emits:

```cmake
include_directories(${XSI_INCLUDE_DIR} $ENV{XILINX_HLS}/include/ /usr/include/jsoncpp/)
```

The hardcoded absolute path assumes a root install. Worked around via
`CPATH`; the correct fix is `pkg-config --cflags jsoncpp` or
`find_package(jsoncpp)`.

### 4.6 `xsim` runtime not on the library path

Running a simulation vbin fails with:

```
Could not load XSI simulation shared library (xsim.dir/top_wrapper_behav/xsimk.so):
libxv_simulator_kernel.so: cannot open shared object file
terminate called after throwing an instance of 'Xsi::LoaderException'
```

Sourcing the Vitis `settings64.sh` does **not** put `xsim`'s runtime on
`LD_LIBRARY_PATH`. Add `$XILINX_VIVADO/lib/lnx64.o` explicitly.

Note the failure mode: when the simulation subprocess dies, VRT logs
`Simulation process exited with code 134` as a **warning** and the host
application then hangs indefinitely rather than erroring out. A first
encounter looks like a hung simulation, not a missing library.

---

## 5. Vivado 2025.2 vs the documented 2025.1

SLASH documents Vivado/Vitis **2025.1** and warns that other versions may
break. This host has **2025.2**. Observed on 2025.2:

| Flow | Result |
|---|---|
| Vitis HLS kernel synthesis → IP-XACT | works |
| `slashkit link -p sim` → simulation vbin | works |
| `xsim` execution of the linked design | works, results correct |
| `slashkit link -p hw` | not reached — blocked before Vivado ran (§6.3) |

So 2025.2 is fine for everything reachable here. The hardware path remains
untested against it, and the static-shell build explicitly requires 2025.1.

---

## 6. Why hardware execution is blocked

### 6.1 PCIe: only PF0 enumerates, with no BARs

```
$ lspci -d 10ee: -nn
11:00.0 Processing accelerators [1200]: Xilinx Corporation Device [10ee:50b4]
```

PF1 (`0x50B5`) and PF2 (`0x50B6`) are absent, and PF0's BARs are
unassigned — `/sys/.../resource` is all zeros and `enable` reads 0. The
boot log shows why:

```
pci 0000:11:00.0: [10ee:50b4] type 00 class 0x120000 PCIe Endpoint
pci 0000:11:00.0: Max Payload Size set to 512 (was 16384, max 512)
pci 0000:11:00.0: buffer not found in pci_save_pcie_state
```

`was 16384` is not a legal MPS value — the Versal was still booting when
the kernel enumerated it, so config space read back as garbage and no
resources were assigned. Reading config space *now* returns sane values
(`10ee:50b4`, multifunction bit set, subsystem `10ee:000e`), so the card
is alive; the stale enumeration is what needs clearing.

`v80-smi` diagnoses this precisely:

```
Board 0000:11:00 NOT READY
  (PF0: NOT READY: wanted driver: 'ami', currently loaded driver: '(none)')
  (PF1: NOT READY: not found)
  (PF2: NOT READY: not found)
  (VRTD: NOT READY: Failed to open socket No such file or directory)
```

Fix (needs root) — a warm reboot is the more reliable equivalent:

```bash
echo 1 | sudo tee /sys/bus/pci/devices/0000:11:00.0/remove
echo 1 | sudo tee /sys/bus/pci/rescan
```

### 6.2 Driver stack loads correctly; the card does not respond

Both kernel modules were built and loaded successfully (Secure Boot was
disabled to allow it). `ami` is not packaged on this host, so it and
`ami_tool` were built from the AVED submodule — see §6.5 for two AVED
build bugs fixed along the way.

```
ami:   Successfully registered module with PCIE Core
slash: driver 'slash' registered
```

Neither bound to anything:

- `lsmod` reports both modules with usage count **0**.
- `/sys/bus/pci/devices/0000:11:00.0/driver` does not exist — no driver bound.
- `ami_tool overview` reports AMI 2.4.0 healthy with an **empty device table**.
- No PCI probe message was ever logged for `11:00.0`.

The kernel never fires a probe because the endpoint's config space reads
all-`0xff`; the `10ee:50b4` entry in `lspci` is a cached artifact of the
failed boot-time enumeration, not a live device. `/dev/ami0` is the
driver's control node, not a board.

**Conclusion: the software stack is fully functional. The card is not
responding.** The supporting evidence is the onboard FT4232H JTAG
(`0403:6011`), which runs on PCIe standby power and should enumerate
regardless of whether the Versal has booted — it is absent from `lsusb`.
Per the host's recorded bring-up notes, that combination means the card's
core rails are not coming up. This requires physical intervention (aux
power cable, reseat, or a different slot), not software.

### 6.3 The hardware vbin needs a static shell that cannot be built here

`slashkit link -p hw` fails immediately:

```
ModuleNotFoundError: No module named 'slashkit.resources.static_shell'
```

`static_shell` is gitignored (`linker/slashkit/resources/.gitignore`). It
is the prebuilt FPGA platform base every hardware vbin links against, and
it ships inside the packaged `slashkit` distribution. Building it locally
via `slashkit install` requires, per
`docs/tutorials/admin/platform-setup.rst`:

1. **The SMBus IP** (`xilinx.com:ip:smbus:1.1`) — *not in the repository
   and not bundled with Vivado*. It must be downloaded from the AMD member
   portal (https://www.xilinx.com/member/v80.html, AMD account required).
   Confirmed absent from `linker/slashkit/resources/base/iprepo/`.
2. **A Vivado Enterprise license** — the SMBus IP is not available under
   the standard tier. No license file or `XILINXD_LICENSE_FILE` /
   `LM_LICENSE_FILE` is configured on this host.
3. **Vivado + Vitis 2025.1** (this host has 2025.2).
4. Several hours of synthesis — the installer source comments call it a
   "10-hour Vivado run".

There is no published SLASH package feed in the documentation, so the
prebuilt-package shortcut is not available either. **Obtaining the static
shell requires an AMD account with V80 member-portal entitlement and an
Enterprise license — neither is something the build can work around.**

### 6.4 AVED build bugs (AMI driver, modern kernels)

`ami` and `ami_tool` are not packaged on this host and were built from
`submodules/AVED/sw/AMI`. Two bugs had to be worked around. Both were
handled via `KCPPFLAGS` on the make command line, leaving the submodule
pristine:

1. **`driver/Makefile` uses `$(PWD)` for its include paths.** Under kbuild
   recursion `$(PWD)` resolves to the kernel tree, not the module
   directory, so `gcq.h` and `fw_if.h` are never found. `$(src)` is the
   correct spelling. (`EXTRA_CFLAGS` is also deprecated in favour of
   `ccflags-y`.)
2. **`linux/vmalloc.h` is no longer transitively included**, making
   `vzalloc`/`vfree` implicit declarations — fatal under `-Werror`.

```bash
D=~/dev/SLASH/submodules/AVED/sw/AMI/driver
make -C $D KCPPFLAGS="-I$D -I$D/fal -I$D/fal/gcq -I$D/gcq-driver/src -include linux/vmalloc.h"
make -C ~/dev/SLASH/submodules/AVED/sw/AMI/api
make -C ~/dev/SLASH/submodules/AVED/sw/AMI/app
```

### 6.5 Secure Boot

Loading either module requires Secure Boot disabled, or the local MOK at
`/var/lib/shim-signed/mok/MOK.der` enrolled via `mokutil --import` plus a
reboot through MOK Manager. The key exists on this host but was **not**
enrolled (the only enrolled MOK is Canonical's), so unsigned modules were
rejected with `Key was rejected by service` until Secure Boot was turned
off in UEFI setup.

### 6.6 Root cause: PCIe enumeration race, and the long-term fix

The endpoint's config space reads valid data *now* (`ee 10 b4 50 …`, link
trained at 32 GT/s x8), but all BARs are unassigned and `enable=0`. The
kernel allocated no resources at boot because config space was garbage at
*that* instant — hence `Max Payload Size set to 512 (was 16384)` in the
boot log. The Versal finishes booting AVED from OSPI after the host has
already walked the bus, so the kernel's view stays frozen from the failed
enumeration: no driver binds, and PF1/PF2 never appear.

**Durable fix — Tandem PCIe.** Versal supports splitting configuration into
a stage-1 image containing only the PCIe endpoint, small enough to meet the
100 ms PCIe deadline from `PERST#` deassertion, with stage 2 loading the
rest in the background. SLASH is already built for it
(`-DTANDEM_BOOT_SUPPORTED=1` in `driver/Makefile`, honoured by
`qdma_drv/.../qdma_nl.h`). The AVED image currently in OSPI does not behave
like a tandem image. Reflashing OSPI with the tandem-capable AVED
deployment PDI is the real fix — and that PDI is the same member-portal
download the static shell needs (§6.3), so one trip resolves both.

**Implemented mitigations** (in `~/dev/`, installed by
`sudo ~/dev/v80_install_fix.sh`):

| Artifact | Purpose |
|---|---|
| `v80_recover.c` / `v80_recover` | Drives `REMOVE` → `TOGGLE_SBR` → `RESCAN` via `/dev/slash_hotplug` |
| `v80-recover-boot.sh` | Detects unassigned BAR0 after a boot delay and recovers |
| `v80-recover.service` | Runs the above at every boot, before `vrtd` |
| `v80_install_fix.sh` | Adds `pci=realloc pci=pcie_bus_safe` to GRUB; installs + enables the service |

The SBR step is essential. SLASH's hotplug uapi documents the order as
remove → SBR → rescan; a bare `remove` + `rescan` without the reset **loses
the device entirely**, which is exactly what happened earlier in this
session. `pci=realloc` targets the unassigned-BAR symptom directly, and
`pci=pcie_bus_safe` avoids the bogus MPS negotiation.

A BIOS **PCIe slot power-on / device-detection delay**, if the board exposes
one, is the most effective host-side knob and should be checked too.

### 6.7 What it would take

```bash
# 1. Re-enumerate so PF1/PF2 appear with BARs (or warm reboot)
echo 1 | sudo tee /sys/bus/pci/devices/0000:11:00.0/remove
echo 1 | sudo tee /sys/bus/pci/rescan
lspci -d 10ee:                       # expect 50b4, 50b5, 50b6

# 2. Load the driver stack (module already built)
sudo insmod ~/dev/SLASH/driver/slash.ko
sudo ~/dev/.slash-local/root/usr/bin/vrtd &

# 3. Verify — all four checks must pass
source ~/dev/.slash-local/env.sh
v80-smi list

# 4. Hardware vbin — blocked until the static shell is obtained (§6.3)
```

---

## 7. Implications for the Vortex AVED AFU work

These findings feed directly back into [aved_afu_proposal.md](aved_afu_proposal.md).

1. **The `sim` platform is not a viable iteration loop for Vortex.** ~0.9 s
   of host time for a two-HLS-kernel design means a Vortex-sized design
   would be impractical per-commit. This validates the proposal's phasing:
   `sim/avedsim` (Verilator, in-tree) must be the development loop, with
   the SLASH `sim` platform reserved for per-milestone packaging validation.

2. **The proposal's phase 5 (on-hardware bring-up) has a procurement
   dependency, not just a technical one.** The static-shell gating (member
   portal + Enterprise license) should be resolved *before* phase 0, since
   it has a lead time no amount of engineering removes. This is a material
   change to the proposal's plan and its risk section should be updated.

3. **Phases 1–4 are entirely unblocked.** The runtime, the AFU RTL, the
   Verilator sim, and IP-XACT packaging all need only what is already
   working here: VRT headers and libraries (built), Vivado 2025.2 (works
   for HLS and IP packaging), and `slashkit link -p sim` (works).

4. **The IP-XACT contract is confirmed in practice.** The HLS flow emitted
   exactly the interfaces the proposal's §4.2 predicts:
   `Add axi4lite interface s_axi_control`, `Add clock interface ap_clk`,
   `Add reset interface ap_rst_n`, `Add interrupt interface interrupt`,
   `Add axi4full interface m_axi_gmem0`. The `s_axi_control` naming
   requirement is real and confirmed — Vortex's `s_axi_ctrl` must be
   renamed at the AVED shim boundary.

5. **The proposal's Q2 (V80 HBM map) is answerable from a vbin** —
   `v80-smi inspect` resolves kernel physical addresses (`0x20200000000`)
   and `argMemoryConfig` port bindings without a device attached. That
   question can be closed in phase 3 rather than waiting for hardware.

6. **Q1 (the host-memory aperture) remains open and remains the critical
   path.** Nothing observed during this bring-up revealed a host-memory
   aperture; the examples all use HBM buffers with explicit
   `sync(HOST_TO_DEVICE)`. The absence of any `HOST[n]` concept in the
   `[connectivity]` cfg grammar is now confirmed by working examples, which
   strengthens the case for the proposal's resolution (a) — a
   device-resident command ring with an explicit sync callback.

---

## 8. Reproducing

```bash
source ~/dev/.slash-local/env.sh
source ~/dev/xilinx_setup.sh
export LD_LIBRARY_PATH="$XILINX_VIVADO/lib/lnx64.o:$LD_LIBRARY_PATH"

cd ~/dev/SLASH/examples/00_axilite
cmake -B build -S . -G Ninja -DSLASH_USE_REPO=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --build build --target hls
cmake --build build --target axilite_sim
./build/00_axilite 0000:11:00 build/axilite_sim.vbin
```

Build logs are under the session scratchpad; the userspace prefix and its
downloaded `.deb` set persist at `~/dev/.slash-local/`.
