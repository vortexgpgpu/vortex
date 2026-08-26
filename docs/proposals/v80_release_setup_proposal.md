# V80 setup: from scripts to packages

**Status:** implemented — SLASH installs from its own Debian packages; the
per-boot scripts and the include shim are gone. Three packaging defects had to
be fixed to get there. **Not yet exercised across a reboot.**
**Scope:** SLASH packaging (fork), `sw/runtime/aved/Makefile`,
`hw/syn/xilinx/aved/tools/`, docs

---

## 1. What was wrong

Bringing up a V80 took a pile of hand-written scripts and root on every boot.
Bringing up a U55C with XRT took installing a package.

The gap was never technical. **Upstream SLASH already ships everything
needed** — Debian and RPM packaging, DKMS, systemd units, udev rules, a `vrtd`
system user. We were not using any of it: SLASH ran out of a source tree with
`insmod` and a manually started daemon, and scripts accumulated to paper over
the consequences.

### 1.1 What a user had to do

```bash
sudo bash ~/dev/v80/slash_only_load.sh      # every boot: insmod + restart vrtd
bash ~/dev/v80/jtag_load_vortex.sh          # every boot: reload the AFU
make -C sw/runtime/aved TARGET=hw VRT_HOME=/opt/xilinx/slash
```

…having first built SLASH from source into a non-default prefix, exported
`VRT_HOME`, `CPATH` and `LIBRARY_PATH` by hand, and created an include-shim
directory (`~/dev/v80/inc-shim`) of symlinks because VRT's public headers pull
in `jsoncpp`/`libxml2` transitively.

---

## 2. Correcting this document's own premises

The first version of this proposal claimed the packaging was largely unused
*and* incomplete, and listed missing udev rules and systemd units among the
work. **That part was wrong.** `scripts/package-deb.sh:106-108` already copies

```
vrt/vrtd/systemd/vrtd.service → debian/vrtd.service
vrt/vrtd/systemd/vrtd.socket  → debian/vrtd.socket
vrt/vrtd/udev/99-vrtd.rules   → debian/vrtd.udev
```

which are the canonical debhelper names that `dh_installsystemd` and
`dh_installudev` consume automatically. A change to "fix" this was written,
found to double-install the same files, and reverted.

The mistake was concluding "no package installs them" by grepping
`packaging/debian/*.install` without reading the script that assembles
`debian/`. The real defects were narrower, and none of them were visible from
the file list.

---

## 3. The three defects that actually blocked packaging

Each produced a failure well away from its cause.

### 3.1 Missing `dh-dkms` build dependency

`debian/rules` runs `dh $@ --with dkms`, but `Build-Depends` listed only
`dkms` — the runtime tool, not the debhelper sequence, which lives in a
separate package. `dpkg-checkbuilddeps` passed, then the build died in
`debian/rules clean`:

```
dh: error: unable to load addon dkms: Can't locate Debian/Debhelper/Sequence/dkms.pm
```

Fixed by declaring `dh-dkms`, so the build is reproducible from a clean
chroot rather than only on a machine that happens to have it.

### 3.2 `vrtd.postinst` had no `#DEBHELPER#` token

`dh_installsystemd` and `dh_installudev` inject their enable/start and
rule-reload snippets at that token. With no token they had nowhere to put them
and **dropped them silently**. The units shipped in the package and were never
enabled — a "packaged" install that still needed
`systemctl enable --now vrtd.socket` by hand on every machine.

This is the one that mattered. It is exactly the failure the packaging exists
to prevent, and it is invisible unless you read the generated `postinst`.

### 3.3 A `./` prefix broke the DKMS module build

```makefile
LIBQDMA_LOCAL_DIR := ./libqdma        # DKMS layout
LIBQDMA_FALLBACK  := ../submodules/qdma_drv/.../libqdma   # in-tree layout
```

Every libqdma object entered `$(MODULE)-objs` as `./libqdma/foo.o`. kbuild
normalises the paths it compiles but matches the `-objs` list **literally**, so
the entries never matched and those files built as plain built-in objects
rather than module objects:

```
DKMS:    CC      libqdma/qdma_mbox.o      ← no [M]
in-tree: CC [M]  libqdma/qdma_mbox.o
```

Without `-DMODULE`, `static_call()` expands to the built-in form that
references the static-call *key* instead of the module-visible trampoline, so
modpost failed:

```
ERROR: modpost: "__SCK__might_resched" [slash.ko] undefined!
ERROR: modpost: "__SCK__WARN_trap" [slash.ko] undefined!
```

The kernel exports 233 `__SCK__` symbols but not those two, because modules
are not meant to reach them. The in-tree fallback path has no `./`, which is
why the driver always built by hand and failed only under DKMS — so nobody hit
it until the packaging was tried.

---

## 4. Secure Boot

DKMS signs the module with a locally generated MOK. Under Secure Boot
(`sig_enforce=Y`) that key must be enrolled:

```
modprobe: ERROR: could not insert 'slash': Key was rejected by service
# dmesg: Loading of module with unavailable key is rejected
```

`sudo mokutil --import /var/lib/shim-signed/mok/MOK.der`, then complete the
enrolment in MokManager at the next boot. Once, ever.

**An open question.** The old `slash_only_load.sh` did `insmod` on a module
with *no signature at all*, which under `sig_enforce=Y` should also have been
rejected — yet those loads reportedly succeeded. Either Secure Boot was
enabled after them, or they ran under a different boot state. This is
unresolved and recorded rather than guessed at. It does not change the
conclusion: enrolling the MOK is correct regardless, and it is strictly better
than depending on unsigned module loading that Secure Boot exists to prevent.

---

## 5. The header shim

`libvrt-dev` depends on `libjsoncpp-dev` and `libxml2-dev` but never says
where their headers are, so with packaged headers alone:

```
/usr/include/vrt/device.hpp:30:10: fatal error: json/json.h: No such file or directory
```

Debian puts those under `/usr/include/jsoncpp/` and `/usr/include/libxml2/`.
That single fact is why `~/dev/v80/inc-shim` existed.

`sw/runtime/aved/Makefile` now asks `pkg-config --cflags jsoncpp libxml-2.0`
instead of hardcoding a path. Verified: `make TARGET=hw` builds
`libvortex-aved.so` with `VRT_HOME`, `CPATH` and `LD_LIBRARY_PATH` all unset,
and `ldd` resolves `libvrt`, `libvrtdpp`, `libslash` and `libvrtd` entirely
from `/lib/x86_64-linux-gnu` — no reference to `/opt/xilinx` remains.

The proper upstream fix is a `vrt.pc`, or not leaking those includes into
public headers at all. `libvrt-dev` ships a CMake config
(`vrtConfig.cmake`) but no pkg-config file.

---

## 6. End state

### 6.1 Installation

```bash
SLASH_PKG_SKIP_ROOT_DESIGN_BUILD=1 bash scripts/package-deb.sh --noninteractive
cd deb && sudo apt install --no-install-recommends \
  $(ls -1 *.deb | grep -v '^ami_' | sed 's|^|./|')
```

DKMS builds the module against the running kernel and rebuilds it on upgrade;
`vrtd.service`/`vrtd.socket` are enabled by the package; udev assigns the
device nodes to the daemon. `ami` is built but not installed — nothing in the
`slash` dependency chain needs it.

### 6.2 Per boot

```bash
v80-smi list                          # no sudo
make -C sw/runtime/aved TARGET=hw     # no sudo, no VRT_HOME
```

The module autoloads from its PCI aliases (`10ee:50c1`, `10ee:50c2` among
them) and `vrtd.socket` is socket-activated. Loading the AFU is still a
*design* action, not setup — the equivalent of XRT's `xclbin` load.

### 6.3 Tools kept

| Tool | Why it stays |
|---|---|
| `jtag_load_vortex.sh` | loading a design over JTAG is a real developer workflow |
| `jtag_load_shell.sh` | recovery when the card is off the bus |
| `step2_flash.sh` | flashing the static shell — not covered by packaging |
| `run_hw_test.sh` | the test harness |
| `hw_sweep.sh` | regression sweep |
| `instrument/` | the forensic harness that found the driver bugs |

### 6.4 Tools deleted

`slash_only_load.sh`, `step1_load.sh`, `bringup.sh` — superseded by the
package. `hw_ladder.sh`, `hw_ladder_noprogram.sh`, `stage_ladder.sh` —
superseded by `hw_sweep.sh` plus `instrument/run_ladder_instrumented.sh`.

**12 tools down to 6**, and the six that remain are tools rather than
workarounds.

### 6.5 The ownership model

Upstream's udev rule gives the device nodes to the **daemon**, not to users:

```
KERNEL=="slash_ctl*",      MODE="0600", OWNER="vrtd", GROUP="vrtd"
KERNEL=="slash_qdma_ctl*", MODE="0600", OWNER="vrtd", GROUP="vrtd"
```

A user never opens `/dev/slash_*`; they talk to `vrtd` over its socket and it
brokers access. A `slash_persistent_setup.sh` written during bring-up added a
rule granting a `vrtadmin` group direct access to those nodes. **That was the
wrong model** — it widened the permission surface to work around a daemon that
simply was not running as a service. It was deleted rather than kept.

---

## 7. What remains

* **Not yet exercised across a reboot.** The autoload and socket-activation
  claims above follow from the installed modalias table and enabled units, not
  from having watched a boot. The MOK enrolment completes at the same reboot.
* **`/opt/xilinx/slash` is still on disk** (7.7 MB) and no longer on the
  linker path. It can go once a hardware run has passed against the packaged
  libraries.
* **`~/dev/v80/`** still holds ~68 untracked machine-local items. Anything
  worth keeping moves in-tree or into `docs/`; the rest is deletable. No
  workflow should depend on a path in one person's home directory.
* **Upstream the fixes.** All three defects in §3, plus the `vrt.pc` gap in
  §5, belong upstream rather than in a fork.
* **`ci/toolchain_install.sh --slash`** still unpacks a userspace-only
  tarball. It should install the packages instead.

---

## 8. What this does not fix

This makes *setup* release-quality. It does not address the open hardware
issues — the `demo`/`stencil3d` wrong results, or the shell's fixed 200 MHz
kernel clock. Those are tracked in
[`../reports/v80_timing_closure.md`](../reports/v80_timing_closure.md) and
[`../designs/aved_driver_architecture.md`](../designs/aved_driver_architecture.md).

A clean install that reaches a board with known bugs is still progress: the
next person hits the real problems on their first afternoon instead of their
third week.
