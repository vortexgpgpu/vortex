# Rules for this directory

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

- `common.h` — and in particular **do not add fields to `kernel_arg_t`**. Every kernel
  reads that struct, so changing its *size* reshuffles codegen in paths you did not
  touch: appending four fields (64 → 80 B) moved mode 2 by +15.8 % and mode 5 by −32.9 %.
  Anything derivable from a build constant (`VX_CFG_SOCKET_SIZE`, `VX_CFG_NUM_CORES`) or
  from an address the kernel already has must be derived, not passed. If a field is
  genuinely unavoidable, ask first, then re-measure **every** mode.
- `kernel.cpp` — the include list only.
- `../../../sim/common/dram_sim.cpp`.

`kernel_modes/wmma_common.h` and `epilogue*`: propose changes, do not make them.

The kernels that used to live in `k_core.h` / `k_tcu.h` were moved verbatim into
`kernel_modes/kernel_m<N>.cpp`, one per file, on request; those two headers are gone. No
logic inside any entry was changed by the move — the cycle counts are identical to the
digit before and after.

## Building

- **The vendored `llvm-vortex` clang does not run on this host.** Its binaries want a
  newer GLIBC than focal provides, so a device build dies before it starts. Use the
  patched copy: `LLVM_PATH=/nethome/sjeong306/vortex_scheduler/tools/llvm-vortex-hostfix`
  on every `make`. That tree is a copy of `bin/` with RPATH and PT_INTERP repointed at
  `tools/glibc-2.35`; the original is untouched.
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
- **Extra `-D`s go in the `CONFIGS` ENVIRONMENT variable, never as a make argument.**
  `make CONFIGS=...` makes the variable immutable, so the Makefile's own
  `CONFIGS += -DVX_CFG_EXT_DTCU_ENABLE` (and the rest of the machine config) are silently
  dropped and every engine mode comes back `skipped=1`. Use
  `CONFIGS="-DDTCU_ACC_BANKS=8" make run-simx OPTS=...`.

## Measuring

- **`make run-simx`, never `./cgo27_motivation`.** Only the `run-simx` target rebuilds the
  simulator with this test's `CONFIGS`. Running the binary directly picks up whatever
  `libsimx.so` was last built — DXA modes come back `skipped=1` and mode 1 segfaults.
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
