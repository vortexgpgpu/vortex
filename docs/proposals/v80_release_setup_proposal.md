# Proposal: a release-quality V80 setup

**Status:** proposal — not implemented
**Scope:** SLASH installation, `aved` driver setup, `hw/syn/xilinx/aved/tools/`, docs

---

## 1. The problem, stated plainly

Bringing up a V80 today takes a pile of hand-written scripts, root on every
boot, and knowledge that exists only in one person's head. Bringing up a U55C
with XRT takes installing a package.

The gap is not technical. **Upstream SLASH already ships everything needed** —
Debian and RPM packaging, DKMS, systemd units, udev rules, a `vrtd` system
user. We are not using any of it. We run SLASH out of a source tree with
`insmod` and a manually started daemon, and have accumulated scripts to paper
over the consequences.

### 1.1 What a user must do today

```bash
sudo bash ~/dev/v80/slash_only_load.sh      # every boot: insmod + restart vrtd
bash ~/dev/v80/jtag_load_vortex.sh          # every boot: reload the AFU
make -C sw/runtime/aved TARGET=hw VRT_HOME=/opt/xilinx/slash
```

…having first built SLASH from source with a non-default prefix, exported
`VRT_HOME`, `CPATH` and `LIBRARY_PATH` by hand, and applied an include-shim
directory (`~/dev/v80/inc-shim`) because VRT's public headers pull in
`zmq`/`CLI`/`inih`/`jsoncpp`/`libxml2` transitively.

### 1.2 The sprawl

* **17 scripts** under `hw/syn/xilinx/aved/tools/` (in-tree).
* **68 items** in `~/dev/v80/` — scripts, one-off probe binaries, logs,
  `.vbin` files, capture dumps — untracked, machine-local, and unreproducible.

Several of those scripts exist purely to work around the unpackaged install:
`slash_only_load.sh`, `step1_load.sh`, `bringup.sh`, `after_boot.sh`,
`slash_persistent_setup.sh`.

### 1.3 The bar

XRT on a U55C:

```bash
sudo apt install ./xrt_*.deb
# reboot; done. No root afterwards.
```

That is the target, and it is achievable because SLASH is built the same way.

---

## 2. What upstream already provides

From `packaging/debian/control` and `packaging/rpm/slash.spec`:

| Package | Contents |
|---|---|
| `slash` | metapackage: everything below |
| `slash-dkms` | kernel module **via DKMS** — survives kernel upgrades |
| `vrtd` | the daemon, with systemd unit + socket |
| `libslash`, `libvrt`, `libvrtd` | runtime libraries |
| `libslash-dev`, `libvrt-dev`, `libvrtd-dev` | headers |
| `slashkit` | the linker |
| `v80-smi` | the CLI |
| `slash-sim-emu` | simulation/emulation runtime |

Plus `vrt/vrtd/udev/99-vrtd.rules`, `sysusers.d/vrtd.conf`, and a `postinst`
that reloads udev and creates the system user.

### 2.1 The ownership model we have been fighting

Upstream's udev rule:

```
KERNEL=="slash_ctl*",      MODE="0600", OWNER="vrtd", GROUP="vrtd"
KERNEL=="slash_qdma_ctl*", MODE="0600", OWNER="vrtd", GROUP="vrtd"
```

The device nodes belong to the **daemon**, not to users. A user never opens
`/dev/slash_*` — they talk to `vrtd` over its socket, and `vrtd` brokers access.
That is why `vrtd.socket` is enabled and why no user needs device permissions.

This matters because it invalidates an approach taken during bring-up: a
`slash_persistent_setup.sh` script was written that adds a udev rule granting
`/dev/slash_*` to a `vrtadmin` group, so users could open the devices directly.
**That is the wrong model** — it widens the permission surface to work around a
daemon that simply was not running as a service. It should be deleted, not
kept. Authorization belongs in `vrtd`, which our fork already extends with
role-based checks (`vrt/vrtd/src/auth.c`).

---

## 3. Proposed end state

### 3.1 Installation

```bash
sudo apt install ./slash_*.deb ./vrtd_*.deb ./slash-dkms_*.deb ...
# or, once published:
sudo apt install slash
```

DKMS builds the module against the running kernel and rebuilds it on upgrade.
`vrtd.service` and `vrtd.socket` are enabled by the package. udev assigns the
device nodes to the daemon. **No further root, ever.**

### 3.2 Per-boot user experience

```bash
v80-smi list                          # works, no sudo
make -C sw/runtime/aved TARGET=hw     # works, no sudo
```

The AFU still needs loading, but that is a *design* action, not setup — the
equivalent of XRT's `xclbin` load, and it belongs to the runtime
(`vrt::Device(bdf, vbin)`), not to a shell script.

### 3.3 Header hygiene

Publish `libvrt-dev` such that `#include <vrt/device.hpp>` compiles against the
installed package with no `CPATH` gymnastics. The transitive
`zmq`/`CLI`/`inih`/`jsoncpp`/`libxml2` includes should either be declared
`Depends:` on the `-dev` package or removed from the public headers. This
deletes `~/dev/v80/inc-shim` and the `CPATH` exports in `run_hw_test.sh`.

### 3.4 What survives in-tree

Keep, as genuine engineering tools:

| Tool | Why it stays |
|---|---|
| `jtag_load_vortex.sh` | loading a design over JTAG is a real developer workflow |
| `jtag_load_shell.sh` | recovery when the card is off the bus |
| `run_hw_test.sh` | the test harness (simplified once `CPATH` hacks go) |
| `hw_sweep.sh` | regression sweep |
| `instrument/` | the forensic harness that found the driver bugs |

Delete, once packaging lands:

| Tool | Replaced by |
|---|---|
| `slash_only_load.sh` | the package (module autoloads, `vrtd` is a service) |
| `slash_persistent_setup.sh` | the package — and its permission model is wrong |
| `step1_load.sh`, `bringup.sh` | the package |
| `hw_ladder.sh`, `hw_ladder_noprogram.sh`, `stage_ladder.sh` | superseded by `hw_sweep.sh` + `instrument/run_ladder_instrumented.sh` |

That is **17 tools down to 6**, and the six that remain are tools rather than
workarounds.

### 3.5 Everything in `~/dev/v80/` goes

68 untracked machine-local items. Anything worth keeping (the probe programs,
the address-map notes) moves in-tree or into `docs/`; the rest is deleted. No
workflow should depend on a path in one person's home directory — today
`run_hw_test.sh` hardcodes `/home/blaise/dev/v80/inc-shim` and every tool
references `/home/blaise/dev/v80/`, which alone makes the setup unreproducible
on any other machine.

---

## 4. Work items

Ordered by value per unit of effort.

1. **Build the SLASH packages from our fork.** `packaging/debian/` already
   exists; verify `dpkg-buildpackage` produces installable artifacts from the
   fork, including our host-buffer and kcompat changes.
2. **Verify the DKMS path** builds the module on this kernel (7.0.0-30) with
   `SLASH_HAVE_TIMER_MODERN=y` wired in, so kernels 6.15+ work out of the box.
   Our timer shim must be part of the DKMS source tree, not a manual make flag.
3. **Fix the `-dev` header dependencies** (§3.3) so `VRT_HOME`, `CPATH` and the
   include shim all disappear.
4. **Publish the packages** to `vortex-toolchain-prebuilt` and wire
   `ci/toolchain_install.sh --slash` to install them rather than unpacking a
   tarball into `$TOOLDIR`.
5. **Delete the superseded scripts** (§3.4) and empty `~/dev/v80/` (§3.5).
6. **Rewrite the setup documentation** as a single linear procedure with no
   branches for "if this fails, try…".

Items 1–3 are the substance. Items 4–6 are consequences.

---

## 5. Risks

* **The fork must track upstream packaging.** If AMD changes `debian/control`,
  our fork has to merge it. This is an argument for keeping fork changes
  minimal and upstreaming the host-buffer feature.
* **DKMS needs the module to build unattended.** Anything requiring a manual
  make flag (like `SLASH_HAVE_TIMER_MODERN`) must be made automatic — detect
  the kernel version in the DKMS build rather than asking the user.
* **`vrtd` as a daemon changes failure modes.** Today a wedged board is fixed
  by killing a foreground process; as a service it needs
  `systemctl restart vrtd`. The bring-up docs must say so.
* **Board recovery still needs JTAG.** Packaging does not remove the need for
  `jtag_load_shell.sh` when a card leaves the PCIe bus. That path stays, and
  stays documented.

---

## 6. What this does not fix

Honesty about scope: this proposal makes *setup* release-quality. It does not
address the open hardware issues — the CP's missing software reset, the
`demo`/`stencil3d` wrong results, or the shell's fixed 200 MHz kernel clock.
Those are tracked in
[`../reports/v80_timing_closure.md`](../reports/v80_timing_closure.md) and
[`../designs/aved_driver_architecture.md`](../designs/aved_driver_architecture.md).

A clean install that reaches a board with known bugs is still progress: it
means the next person hits the real problems on their first afternoon instead
of their third week.
