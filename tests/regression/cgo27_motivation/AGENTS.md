# Rules for this directory

## What this harness is for

**It exists to show that no single placement wins — that which mode is fastest depends on
the GEMM's shape and on whether an epilogue is attached.** The deliverable is a set of
regimes, each with a different winner, not a ranking with the DTCU on top.

That has been written down late, after several rounds of analysis drifted into "the engine
wins, here is by how much" and had to be pulled back. Two axes carry the diversity:

| axis | what changes | why it flips the winner |
| --- | --- | --- |
| **shape** | K relative to M and N | K sets arithmetic intensity. Small fixed K (attention: `M = N = seqlen`, `K = head_dim`) is memory-bound and the engine wins because nothing stalls it. K growing with M and N (a cubic ladder) becomes compute-bound and the in-core TCU wins because its array is 64× wider. |
| **epilogue** | `-a 1` vs an elementwise app | The in-core modes fuse the activation into the accumulator on the way out, for ~0.7 %. The DTCU has no epilogue hardware, so the same app costs it a **second full launch over D**. |

**When you report a number, report which regime it belongs to.** A ratio quoted without its
shape family is the failure mode this file exists to prevent — "mode 7 is 2× faster" was
true on one ladder and false on the next rung of another.

Corollaries:

- A mode that loses everywhere is not necessarily a failed mode; it may be a **control**.
  Modes 2 and 4 exist to isolate what DXA and the workgroup geometry are worth, and 3 is 5
  without A-residency. Do not delete a control because it is slow, and do not put controls
  in the headline table — the current main table shows 0/1/5/7/8 only.
- Mode 0 (SIMT) is quoted at the smallest shape only. It has no tensor unit, no staging and
  no tuning knob, so re-running it at scale measures nothing.
- **A cubic ladder is not a workload.** `M : N : K = 4 : 2 : 1` held constant matches no
  real layer; it is a machine-scaling probe and must be labelled as one. Attention-shaped
  (`K` fixed at a head dimension) and FFN-shaped (`K = N = hidden`) are the families worth
  claiming representativeness for.

## Every experiment result goes into BOTH documents, in the same change

There are three places a number can live and they must never disagree:

| file | what it holds |
| --- | --- |
| [`docs/dtcu_figures.html`](docs/dtcu_figures.html) | the figures and the results table, published as an Artifact |
| [`docs/260824_DTCU_update_RFC.md`](docs/260824_DTCU_update_RFC.md) | the design argument the numbers support |
| [`README.md`](README.md) | the full result table and the how-to-reproduce |

**When a measurement changes, update all three before reporting it.** Not "the RFC now,
the figure later" — a figure showing superseded numbers is worse than no figure, because
it is the artifact that gets shared and read on its own. The HTML in this directory is the
source of the published Artifact; edit it here and republish from here.

Republish with the existing URL so the link keeps working. If the target 404s the
artifact was deleted and the URL cannot be revived — publish fresh and update the id
here, because a stale id in this file silently mints a second artifact instead of
updating the first:

```
Artifact(file_path=".../docs/dtcu_figures.html",
         url="https://claude.ai/code/artifact/3cd01bb8-628c-4a9a-bac2-45c29260b90b")
```

When a number is superseded, say so where the old one was, rather than silently swapping
it — "before the per-socket split this was 1,154,569" is the finding, and deleting it
throws the finding away.

## Layout

```
main.cpp  Makefile  sweep_exp*.py      the driver, the build, the sweeps
common.h  epilogue.h                   host/device ABI -- both sides include these
docs/          RFCs + dtcu_figures.html (the published Artifact's source)
host/          x86 driver: host_modes.h, run_modes.h, host_run.h, host_args.h, host_types.h
kernel_modes/  one GPU program per mode (kernel_m<N>.cpp) + the helpers 2+ modes share
```

## Do not modify without being asked

- `common.h` — **the restriction is on `kernel_arg_t`'s size, not on the file.** Every
  kernel reads that struct, so changing its size reshuffles codegen in paths you did not
  touch: appending four fields (64 → 80 B) moved mode 2 by +15.8 % and mode 5 by −32.9 %.
  Anything derivable from a build constant (`VX_CFG_SOCKET_SIZE`, `VX_CFG_NUM_CORES`) or
  from an address the kernel already has must be derived, not passed. If a field is
  genuinely unavoidable, ask first, then re-measure **every** mode.

  **`#define`s are a different matter and belong here.** They add no bytes to the struct —
  verified: adding `MOTI_WG_KSTEPS`, `MOTI_WG_NCOLS` and `MOTI_AUX_ELEM_OFFSET` left all
  eight modes at exactly their previous cycle counts. This file is the right home for any
  constant the **host and the kernel must agree on**, because a second `#ifndef` default in
  one of them is silent: the kernel tiles one way, the host sizes the grid, the Local
  Memory and the DXA descriptors the other way, and D comes out wrong with no error.
  `MOTI_WG_KSTEPS` and `MOTI_WG_NCOLS` each carried a default in *two* files until
  2026-08-08 and agreed only because nobody had edited one of them.

  **Auxiliary epilogue operands do not need a field either.** Apps 4/5/7/8 want a residual
  matrix, a scale vector or a bias; the host appends them to the C buffer and still passes
  that buffer's base as `C_addr`, and the kernel derives the address from `C_addr`, `M` and
  `N` via `MOTI_AUX_ELEM_OFFSET`. App 6 (row-wise softmax) needs no operand at all, and
  apps 7/8's int8 inputs are a build variant of `ITYPE` rather than a runtime app id.
- `kernel.cpp` — the include list only.
- `../../../sim/common/dram_sim.cpp`.

`kernel_modes/wmma_common.h` and `epilogue*`: propose changes, do not make them.

The kernels that used to live in `k_core.h` / `k_tcu.h` were moved verbatim into
`kernel_modes/kernel_m<N>.cpp`, one per file, on request; those two headers are gone. No
logic inside any entry was changed by the move — the cycle counts are identical to the
digit before and after.

## Building

**Read `../../../README.md` first and follow it.** Its build and run flow is the
documented one; everything below is a delta for this directory, not a replacement. Ignore
it and you invent a private procedure that happens to work until it does not — which has
already happened here several times:

| what the README says | what going around it cost |
| --- | --- |
| `ci/blackbox.sh --driver=simx --app=<test> --args=...` assembles `CONFIGS` from `--cores=`, `--warps=`, `--l2cache`, … | Passing `CONFIGS` by hand as a *make argument* made it immutable, dropped the Makefile's own `CONFIGS +=`, and produced a simulator with no DXA and no DTCU — every engine mode came back `skipped=1` |
| `../configure` runs **once** when the build folder is created | Re-running it "to refresh a header" overwrote the build-tree copy of `sw/runtime/simx/Makefile` and deleted an include path that was only ever there, breaking the build |
| the toolchain comes from `ci/toolchain_install.sh` at `$TOOLDIR/llvm-vortex` | That copy does not run on this host (GLIBC), so every `make` needed `LLVM_PATH=` passed by hand; forget it once and the device binary is built by a different compiler |

Fix the toolchain path in `build/config.mk` (`LLVM_PATH ?= …/llvm-vortex-hostfix`) rather
than passing it per-invocation, so `blackbox.sh` and a bare `make` agree.


- **The vendored `llvm-vortex` clang does not run on this host** — its binaries want a
  newer GLIBC than focal ships, so a device build dies before it starts.
  `tools/llvm-vortex-hostfix` is a copy of `bin/` with RPATH and PT_INTERP repointed at
  `tools/glibc-2.35`; it still RPATHs the real `lib/`, so it supplements the toolchain
  rather than replacing it.

  **Already pinned in `build/config.mk`, so no `LLVM_PATH=` on the command line.** That is
  deliberate: `ci/blackbox.sh` takes no such argument, so the per-invocation habit breaks
  the moment anything goes through the documented entry point.

  ⚠️ `configure` regenerates `config.mk` and drops the pin. A device build suddenly
  failing on `GLIBC_2.38` means someone re-ran configure — put the line back rather than
  going back to passing it by hand.
- **The build tree holds a configure-time COPY of this directory's `Makefile`, and that
  copy is the one `make` reads.** Editing the source-tree `Makefile` alone changes
  nothing. Retiring modes 5/6 hit this immediately: `MOTI_MODES` was updated in
  `tests/regression/cgo27_motivation/Makefile`, the source `kernel_m5.cpp` was deleted,
  and the build still failed with

  ```
  make: *** No rule to make target 'kernel_m5.elf', needed by 'kernel_m5.vxbin'.
  ```

  because the stale copy still listed it. `rm .depend` does not help — the mode list is
  in the Makefile, not the dependency file.

  **Copy the file across; do NOT re-run `configure` to "refresh" it.** Configure would
  regenerate `config.mk` and drop the `LLVM_PATH` pin two bullets up, trading this failure
  for a `GLIBC_2.38` one:

  ```sh
  cp tests/regression/cgo27_motivation/Makefile \
     build/tests/regression/cgo27_motivation/Makefile
  ```

  Anything else that reaches `make` through the build tree — not just `MOTI_MODES` — needs
  the same copy.
- **`kernel.elf` depends on `kernel.cpp`, not on the `k_*.h` files it includes.** Editing
  a device header does not rebuild the device binary, and the run then measures the old
  kernel while reporting the new source. `rm vx_start.o kernel.elf kernel.vxbin` to force
  it. Two measurements were reported against a stale kernel before this was noticed.
- **One device program per mode — do not put a new kernel in a shared `.vxbin`.**
  Each mode builds `kernel_m<N>.vxbin` from `kernel_modes/kernel_m<N>.cpp`, which holds
  that kernel's body directly. A header there only exists when more than one mode needs
  it (`wmma_common.h`, `k_smem_stage.h`, `k_dtcu_desc.h`, `k_epilogue.h`). The host loads
  `kernel_m<mode>.vxbin`.

  This is not tidiness. In the old all-in-one binary a mode's cycle count depended on
  which OTHER modes existed: inserting modes 3/4 moved mode 2 from 15,468 to 24,106 with
  a **byte-identical** `moti_tcu_dxa` — same 423 instructions, same `0x698` size, same
  1,368 executed instructions, same 1,120 instruction fetches, only the start address
  differing (icache set 41 → 62) and the average fetch latency going 54.0 → 101.8 cycles.
  Per-mode programs put every kernel at `0x180000034` regardless of what else is in the
  tree, and shrink each program from 14,700 B (90 % of the 16 KB icache) to 536–2,940 B.
  Verified: growing mode 3 by a dummy kernel now moves no other mode's address by a byte.

  Adding a mode: a `kernel_modes/kernel_m<N>.cpp` holding the kernel, a `run_mode_N()` in
  `host/run_modes.h`, an id in `host/host_modes.h`. Then re-run `-m all` and check the other modes
  did not move — if they did, something is still shared that should not be.
- **Stopping a background sweep does not stop its simulators.** `TaskStop` kills the
  script; the `cgo27_motivation` processes it launched keep running, keep holding cores,
  and their results go nowhere. Twelve of them from two cancelled sweeps were still
  running four hours later, slowing the sweep that replaced them. After stopping one,
  check and clean up:

  ```sh
  ps -eo pid,etimes,args | grep 'cgo27_motivation -m' | grep -v grep
  # anything older than the current sweep is an orphan
  ```

  Count processes with `ps` and read the arguments, not `pgrep -c`: each run shows up
  twice (the `timeout` wrapper and the binary), so a bare count reports half or double
  and it is easy to conclude a sweep died when it is running fine.

- **Builds are serial; RUNS are not.** The lock below is about `make` sharing a
  directory. Running the built binary is not a build — it only reads `cgo27_motivation`
  and `kernel_m*.vxbin` — so a sweep should build each configuration once and then launch
  every point at once. This host has 128 cores and the largest shapes dominate the wall
  clock, so overlapping them is most of the time saved: a 16-point sweep that took ~60 min
  serially converges to the single slowest point.

  Snapshot per configuration, because `CONFIGS` reaches the simulator build too and every
  configuration writes the same filenames:

  ```sh
  # serial, under the lock
  ( exec 9>"$BUILD/.moti-build.lock"; flock 9
    CONFIGS="$cfg" make cgo27_motivation kernel_m12.vxbin ... LLVM_PATH=$LLVM
    CONFIGS="$cfg" make -C "$RT/simx" DESTDIR="$RT"
    mkdir -p "$WORK/$tag"
    cp -f cgo27_motivation kernel_m*.vxbin "$RT"/libsimx.so "$RT"/libvortex*.so "$WORK/$tag/" )

  # parallel, no lock needed
  ( cd "$WORK/$tag" && LD_LIBRARY_PATH="$WORK/$tag" VORTEX_DRIVER=simx \
      ./cgo27_motivation -m $m -M $M -N $N -K $K ) &
  ```

  Copying `libsimx.so` into the snapshot is what makes "each run used the binaries built
  for its own configuration" true by construction rather than by timing.

  **Both `make` lines are required.** A bare `make` in the test directory builds the test
  and the kernels but NOT the simulator, so copying `libsimx.so` then snapshots whatever
  was last built there — which is the same stale-runtime failure as running
  `./cgo27_motivation` directly, just carried into the snapshot where it is harder to see.
  Its signature is unmistakable once you know it: **mode 1 comes back `FAILED!` and the
  engine modes come back `skipped=1`**, and a workgroup mode simply never terminates. That
  reads exactly like a kernel bug and cost a wrong "I broke it" conclusion here. Before
  trusting a snapshot, run one untouched mode in it and check the number against README —
  mode 1 at 128×64×32 is 23,513.

- **One `make` at a time in the build directory, and take the lock.** Two concurrent
  builds there once corrupted a sweep point — the same configuration read 1,097,497 in
  one run and 377,077 in another. It is easy to do by accident: a background sweep left
  running while you rebuild is exactly this, and it has already happened in this project
  by forgetting to stop one before editing.
  **Before starting a background measurement, `TaskStop` any earlier one**, and have the
  script hold `build/.../.moti-build.lock` with `flock -n` so forgetting fails loudly
  instead of silently:

  ```sh
  exec 9>"$BUILD/.moti-build.lock"
  flock -n 9 || { echo "another build holds the lock" >&2; exit 1; }
  ```

  If a sweep might have overlapped anything, throw its results away and re-run — a
  number you cannot vouch for is worse than no number.
- **Extra `-D`s: prefer `ci/blackbox.sh`'s flags; otherwise the `CONFIGS` ENVIRONMENT
  variable — never a make argument.** `make CONFIGS=...` makes the variable immutable, so
  the Makefile's own `CONFIGS += -DVX_CFG_EXT_DTCU_ENABLE` (and the rest of the machine
  config) are silently dropped and every engine mode comes back `skipped=1`. For a knob
  blackbox has no flag for (`-DDTCU_ACC_BANKS=8`, `-DMOTI_WG_KSTEPS=2`), use
  `CONFIGS="…" make run-simx OPTS=…`.

## Measuring

- **`ci/blackbox.sh` or `make run-simx` — never `./cgo27_motivation` directly.** Only
  those rebuild the simulator with this test's `CONFIGS`; the bare binary picks up
  whatever `libsimx.so` was last built, and then DXA modes come back `skipped=1` and mode
  1 segfaults. The one exception is a snapshot directory built for a parallel sweep, where
  the point is that the binaries beside it are the ones its configuration produced.
- **Skip mode 0 (SIMT) above 128×64×32 and quote the recorded number.** It is the
  no-tensor-unit baseline and its cost is the point, but it is also the slowest thing in
  the suite by an order of magnitude — 9,581,708 cycles at 512×256×128 against mode 1's
  377,131 — and it holds a slot in every parallel batch for as long as everything else
  put together. It does not change: it has no engine, no staging and no tuning knob, so
  re-running it measures nothing new. Recorded values, app 1:

  | shape | cycles | MAC/cyc | vs mode 1 |
  |---|--:|--:|--:|
  | 128 × 64 × 32 | 190,995 | 1.37 | 13.1× slower |
  | 256 × 128 × 64 | 1,145,460 | 1.83 | 11.8× slower |
  | 512 × 256 × 128 | 9,581,708 | 1.75 | **25.4× slower** |

  Re-measure it only when something it actually depends on changes — `kernel_arg_t`, the
  machine config, or `kernel_m0.cpp` itself.

- **Never report a cycle count from a run that did not print `PASSED!`.** Every mode is
  verified element-by-element against an independently computed CPU reference (full M×N,
  ULP ≤ 6), and D is zeroed before each run so a mode that writes nothing always fails.
  A wrong result is usually *faster*, so an unexplained speedup is a correctness suspect
  first and a finding second.
- **A launch that misses a core is silent wrong output, not a hang.** The engine kernels
  derive their slice from `vx_core_id()`, so `grid_dim` must be `NUM_CORES`. Mode 8 kept a
  1×1×1 launch after it was changed to a per-core split and produced 6,144 of 8,192
  elements wrong with no error, no timeout, and a plausible cycle count. Only the verify
  pass caught it.
- **A zeroed descriptor is a *valid* descriptor.** `fmt_d = 0` is `fp32`, so
  `init_tile_state_` accepts it, `M = N = K = 0` retires instantly, and the engine sets
  `done = 1`. A submitter polling that descriptor therefore succeeds while nothing was
  computed. Whenever an engine mode "passes" suspiciously fast, check the D output before
  the cycle count.
- Quote a mode's number only if it is stable. Mode 5 is bimodal — see the ⚠ in README.md.
