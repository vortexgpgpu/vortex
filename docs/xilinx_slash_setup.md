# SLASH / Alveo V80 Setup Guide

Build and install the SLASH platform stack, the prerequisite for running
Vortex on an AMD Alveo V80 through the [`aved`](proposals/aved_afu_proposal.md)
backend.

SLASH is the V80's equivalent of XRT: a runtime (VRT), a linker (`slashkit`),
a daemon (`vrtd`), and a kernel driver. It is a separate project from Vortex —
clone it alongside, do not vendor it.

- Upstream: <https://github.com/Xilinx/SLASH>
- Vortex-side integration: [`sw/runtime/aved/`](../sw/runtime/aved/),
  [`sim/avedsim/`](../sim/avedsim/), [`hw/syn/xilinx/aved/`](../hw/syn/xilinx/aved/)

---

## 1. Terminology trap

SLASH renames both emulation tiers relative to Vitis/XRT, and the names
collide:

| Vitis/XRT | SLASH | Meaning |
|---|---|---|
| software emulation | **emulation** (`emu`) | behavioural C-model |
| hardware emulation | **simulation** (`sim`) | RTL Verilog simulation |

RTL kernels — which Vortex is — support only `hw` and `sim`. `emu` is
unavailable; the AVED backend rejects it with a diagnostic rather than
failing obscurely inside the linker.

Vortex's own `sim/avedsim` (Verilator) is a *third*, distinct thing and is
the intended day-to-day iteration loop. See
[proposals/aved_afu_proposal.md §7](proposals/aved_afu_proposal.md).

---

## 2. Prerequisites

**Tools.** Vivado / Vitis. SLASH documents **2025.1**; see §7 for what does
and does not work on 2025.2.

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh   # puts vivado, vitis, v++, xsct on PATH
```

**Packages.** For the package build see [§3.1](#31-build-the-packages) — the
list below is for the source build in [§3A](#3a-building-from-source-maintainers-only).
SLASH's documented list is incomplete; the full set:

```bash
sudo apt install cmake pkg-config ninja-build \
  libxml2-dev libzmq3-dev libjsoncpp-dev zlib1g-dev \
  libsystemd-dev libinih-dev libcli11-dev \
  libcap-dev cppzmq-dev \
  linux-headers-$(uname -r)
```

`libcap-dev` (pulled in by `libsystemd`) and `cppzmq-dev` (provides
`zmq.hpp`, which `libzmq3-dev` does not ship) are **not** in SLASH's README
and both are hard build failures.

**Submodules.**

```bash
git submodule update --init --recursive
```

---

## 3. Installing

SLASH ships Debian and RPM packaging. Build the packages once, install them
once, and nothing further is needed on any boot: the kernel module is built
and reinstalled by DKMS across kernel upgrades, `vrtd` is a socket-activated
systemd service, and udev assigns the device nodes to the daemon.

### 3.1 Build the packages

```bash
sudo apt install --no-install-recommends \
  debhelper dh-dkms cmake libcli11-dev libinih-dev libjsoncpp-dev \
  libsystemd-dev libxml2-dev libzmq3-dev cppzmq-dev ninja-build \
  pkg-config python3-jinja2 python3-pip python3-setuptools python3-venv \
  python3-wheel rsync zlib1g-dev

cd SLASH
SLASH_PKG_SKIP_ROOT_DESIGN_BUILD=1 bash scripts/package-deb.sh --noninteractive
```

`SLASH_PKG_SKIP_ROOT_DESIGN_BUILD=1` skips rebuilding the platform design,
which needs `v++` and the node-locked SMBus IP. The driver, daemon, libraries
and `slashkit` do not depend on it.

Artifacts land in `deb/`.

### 3.2 Install

```bash
cd deb
sudo apt install --no-install-recommends \
  $(ls -1 *.deb | grep -v '^ami_' | sed 's|^|./|')
```

`ami` is built but deliberately left out: nothing in the `slash` dependency
chain needs it, and it is the component that heartbeats the AMC over GCQ.

If a previous **source** install exists, remove its leftovers first or they
win: files under `/etc/systemd/system/` override the packaged units, and
`/etc/ld.so.conf.d/slash.conf` puts the old prefix ahead of the system
libraries.

```bash
sudo rm -f /etc/systemd/system/vrtd.service /etc/systemd/system/vrtd.socket
sudo rm -f /etc/ld.so.conf.d/slash.conf
sudo systemctl daemon-reload && sudo ldconfig
```

### 3.3 Secure Boot

DKMS signs the module with a locally generated MOK. Under Secure Boot that
key must be enrolled or the kernel refuses the module:

```
modprobe: ERROR: could not insert 'slash': Key was rejected by service
# dmesg: Loading of module with unavailable key is rejected
```

Enrol it once, then reboot and complete the enrolment in MokManager
(*Enroll MOK → Continue → Yes →* the password you set):

```bash
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der
```

`mokutil --sb-state` reports whether Secure Boot is on; if it is off, this
step does not apply.

### 3.4 Verify

After the reboot nothing needs starting by hand — the module autoloads from
its PCI aliases and `vrtd.socket` is socket-activated:

```bash
lsmod | grep slash
systemctl is-active vrtd
v80-smi list
```

`v80-smi list` reports **PF0 NOT READY** when `ami` is absent. That is a
readiness *report*, not a requirement: PF1, PF2 and VRTD are what matter.

---

## 3A. Building from source (maintainers only)

The package route above is the supported one. Build in place only when
developing SLASH itself; the result is not what a deployment should run.

Components must be built in dependency order.

```bash
cd SLASH
cd driver && make && sudo insmod slash.ko && cd ..
cd driver/libslash && cmake -S . -B build -G Ninja && cmake --build build \
  && sudo cmake --install build && cd ../..
cd vrt/vrtd && cmake -S . -B build -G Ninja && cmake --build build \
  && sudo cmake --install build && cd ../..
cd vrt && cmake -S . -B build -G Ninja && cmake --build build \
  && sudo cmake --install build && cd ..
cd smi && cmake -S . -B build -G Ninja && cmake --build build \
  && sudo cmake --install build && cd ..
```

An in-place `insmod` of an unsigned module is rejected under Secure Boot;
see §3.3.

---

## 4. Running the examples

```bash
cd examples/00_axilite
cmake -B build -S . -G Ninja -DSLASH_USE_REPO=ON
cmake --build build                       # host application
cmake --build build --target hls          # HLS kernels
cmake --build build --target axilite_sim  # simulation vbin
./build/00_axilite 0000:11:00 build/axilite_sim.vbin
```

**`TARGET=sim` needs neither a board nor `vrtd`.** The BDF argument is only
used to key a metadata cache directory; the device is never opened. This
makes the `sim` platform usable on any machine with Vivado.

Two gotchas:

- **`xsim`'s runtime is not on the library path** after sourcing the Vitis
  settings. Without it a run fails with `libxv_simulator_kernel.so: cannot
  open shared object file`, logs `Simulation process exited with code 134` as
  a *warning*, and then **hangs indefinitely** — it looks like a slow
  simulation, not a missing library.

  ```bash
  export LD_LIBRARY_PATH="$XILINX_VIVADO/lib/lnx64.o:$LD_LIBRARY_PATH"
  ```

- **The generated sim project hardcodes `/usr/include/jsoncpp/`.** Fine for a
  root install; add it to `CPATH` for a user prefix.

---

## 5. The static shell

Every *hardware* vbin links against a prebuilt **static shell** — the
platform base (PCIe/CPM, QDMA including the slave bridge, HBM, NoC, clocking,
and the DFX aperture the user design plugs into). It contains no user logic.

The shell is **not** in the repository. It ships inside the packaged
`slashkit` distribution, or is built locally:

```bash
cd linker
python3 -m slashkit install --out-dir ./slashkit/resources --build-dir ./install.prj
```

Expect several hours of Vivado. Three prerequisites, all easy to conflate:

1. **SMBus IP** (`xilinx.com:ip:smbus:1.1`) — *not* in the repository and
   *not* bundled with Vivado. Download from the AMD member portal
   (<https://www.xilinx.com/member/v80.html>) and place the IP directory at
   `linker/slashkit/resources/base/iprepo/smbus_v1_1/`.
2. **An SMBus license** — a separate thing from the IP. The IP will not
   elaborate without it. Node-locked; install at `~/.Xilinx/Xilinx.lic` and
   point `XILINXD_LICENSE_FILE` at it.
3. **The HLS IPs in the iprepo must be built first.** `hbm_bandwidth` and
   `traffic_producer` ship as HLS *source*. Until compiled, the shell build
   fails with `[BD::TCL 103-2012] The following IPs are not found in the IP
   Catalog` — which reads like a tool-version problem but is not:

   ```bash
   cd linker/slashkit/resources/base/iprepo && make
   ```

### Programming the shell to flash

Use `v80-smi`, which goes through the AMC over GCQ behind PF0 (`ami.ko` must be
loaded). Write **both** boot partitions, same PDI, ~10 min each:

```bash
PDI=<resources>/static_shell_compute/amd_v80_gen5x8_25.1.pdi

v80-smi write-static-shell --flash --shell-type compute -d 0000:01:00 --pdi $PDI
v80-smi write-static-shell --flash --shell-type service -d 0000:01:00 --pdi $PDI
```

Four things that are easy to get wrong:

1. **`--pdi` is effectively mandatory.** Without it, `v80-smi` resolves an
   *installed* shell path, which may be the root-owned
   `/usr/lib/python3.12/dist-packages/slashkit/resources/static_shell_compute/`
   copy rather than the tree you just built — silently reflashing an older
   shell. Point it at the PDI you mean.
2. **Both partitions, or the next reboot may lose the card.** `compute` is boot
   partition 1, `service` is partition 0, and **POST reads partition 0**.
   Writing only `compute` works immediately — the board comes back reporting
   `Shell: compute` and tests pass — and then fails to enumerate after the next
   reboot, with root port `0000:00:01.1` absent and no JTAG that boot.
3. **Use the FPT image, not `_nofpt`.** `--flash` needs the flash image; the two
   differ by a 32 KB FPT header (magic `0x92F7A516`). `_nofpt` is JTAG-only.
4. **`--shell-type all` is rejected with `--pdi`**, despite what `--help`
   implies, so the two writes must be separate commands.

**No reboot or power cycle is needed.** Each write ends by resetting the link
(`Toggling secondary bus reset` / `Removing PCIe functions`, then
`Rescanning PCIe`) and the board returns on its own. Confirm with `v80-smi list`
— all PFs `OK` and a shell reported rather than `unknown`.

Once both partitions are written, the runtime programs the PL on device open
like XRT's `load_xclbin()`; there is no never-program switch.

> **Flashing writes the card's boot memory.** A bad write can leave the board
> unbootable, recoverable only over JTAG. Confirm the image and the BDF before
> running this.

---

## 6. Bring-up troubleshooting

**PF0 only; PF1/PF2 missing.** PF1 (`slash_qdma`) and PF2 (`slash_ctl`) come
from the static shell in flash. On a board that has never had it programmed,
PF0-only is expected — complete §5 first. `slash.ko` binds *only* PF1/PF2, so
until they appear the driver has nothing to attach to.

**BARs unassigned / `enable=0` / config space reads `0xff`.** The Versal can
finish booting from OSPI *after* the host has enumerated PCIe, leaving the
kernel's view frozen from a failed enumeration. Symptom in `dmesg`:

```
pci ...: Max Payload Size set to 512 (was 16384, max 512)
```

`16384` is not a legal MPS — it means config space read back as garbage. The
durable fix is a **Tandem PCIe** image in OSPI (stage 1 brings up the endpoint
inside the 100 ms PCIe deadline; SLASH is already built for it —
`-DTANDEM_BOOT_SUPPORTED=1`). As a mitigation, force re-enumeration via
`/dev/slash_hotplug`, in the documented order:

```
REMOVE  ->  TOGGLE_SBR  ->  RESCAN
```

The Secondary Bus Reset is **essential** — a bare remove + rescan without it
loses the device entirely.

Host-side kernel parameters worth adding: `pci=realloc pci=pcie_bus_safe`.

**Module fails to load: `Key was rejected by service`.** Secure Boot rejects
unsigned modules. Either disable Secure Boot, or enrol a MOK and sign:

```bash
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der   # then reboot, enrol
sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 \
  /var/lib/shim-signed/mok/MOK.priv /var/lib/shim-signed/mok/MOK.der slash.ko
```

Re-sign after every rebuild.

**`ami` / `ami_tool` not found.** They come from AVED, not SLASH. Build from
the submodule:

```bash
D=submodules/AVED/sw/AMI/driver
make -C $D KCPPFLAGS="-I$D -I$D/fal -I$D/fal/gcq -I$D/gcq-driver/src -include linux/vmalloc.h"
make -C submodules/AVED/sw/AMI/api
make -C submodules/AVED/sw/AMI/app
```

The `KCPPFLAGS` override works around two AVED bugs on current kernels: the
driver Makefile uses `$(PWD)` for include paths (which resolves to the kernel
tree under kbuild recursion, so `gcq.h`/`fw_if.h` are never found), and
`linux/vmalloc.h` is no longer transitively included, making `vzalloc`/`vfree`
implicit declarations.

**Driver builds fail on `del_timer` / `from_timer`.** The vendored QDMA
snapshot predates their removal in kernel 6.15. Add a `kcompat` probe mapping
them onto `timer_delete()` / `timer_container_of()` and force-include the
shim; SLASH's `driver/kcompat/probe.sh` mechanism is designed for exactly
this.

---

## 7. Vivado 2025.2

SLASH pins **2025.1** across every branch. On 2025.2, in order of what you
hit:

| Issue | Nature | Resolution |
|---|---|---|
| `BD::TCL 103-2041` version guard | A `write_bd_tcl` string compare, not an engine restriction | Relax `set scripts_vivado_version` in the shell BD scripts |
| Missing HLS IPs | Not a version issue at all | Build them (§5) |
| NoC `invalid bareword "E"` | Bug in AMD's `axi_noc_v1_1/xit/update_contents.xit`: the unit table stops at `T`, so a `1E` aperture is left as a literal | Add `P 2**50 E 2**60` to the `string map` |
| `dcmac:3.0` not in catalog | Genuine IP version bump — 2025.2 ships `dcmac_v3_1` | Use the compute-only shell (below) |

The DCMAC break is the only one without a local fix. It lives in the **service
layer** (600G Ethernet), which a compute workload does not need. The
`feature/compute_only_platform_v2` branch splits the shell into `compute` and
`service` variants; the compute variant instantiates no DCMAC:

```bash
python3 -m slashkit install --shell-type compute \
  --out-dir ./slashkit/resources --build-dir ./install.prj
```

Verified working on 2025.2: HLS synthesis, IP-XACT packaging,
`slashkit link -p sim`, and `xsim` execution.

---

## 8. Related

- [proposals/aved_afu_proposal.md](proposals/aved_afu_proposal.md) — the AVED
  backend design and `TARGET` semantics
- [proposals/slash_v80_bringup_report.md](proposals/slash_v80_bringup_report.md)
  — hands-on bring-up log this guide distils
- [fpga_setup.md](fpga_setup.md) — the XRT/Alveo and Altera paths
