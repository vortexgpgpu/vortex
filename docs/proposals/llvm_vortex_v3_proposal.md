**Date:** 2026-05-10
**Status:** Draft
**Author:** Blaise Tine
**Related:**
[pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md),
[hip_support_proposal.md](hip_support_proposal.md),
[chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md),
[building_toolchain.md](../building_toolchain.md).

### Update history

- **2026-05-10** — Initial draft. Triggered by the chipStar recon
  ([chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md))
  surfacing that our current `llvm_vortex` is RISCV-only with no
  `ld.lld` and a RISC-V default triple, which forced a build-time
  rebuild during the session (X86 added, lld added, host-default
  triple, `GCC_INSTALL_PREFIX`, `clang.cfg` for libstdc++ pinning).
  This proposal formalizes that rebuild **plus** the LLVM 18 → 20
  bump into a single planned migration so the next major effort is
  ~12 months out, not piecemeal.

# llvm_vortex v3 — Migration Proposal (LLVM 20 + X86 + lld)

## 1. Summary

Migrate the Vortex LLVM fork from its current state (LLVM 18.1,
RISCV-only, no `ld.lld`, RISC-V default triple) to a clean LLVM 20
base with both **RISCV and X86 backends**, **`lld` shipped**,
**host-default triple**, and the Vortex-specific RISC-V target
extensions ported forward. The new branch is **`vortex_3.x`** in
`vortexgpgpu/llvm`, paralleling `pocl_vortex/vortex_3.x` and the
broader v3 Vortex stack.

Goals — in priority order:

1. **LLVM 20.x base** with the Vortex-specific RISCV target patches
   ported forward. Matches POCL `release_7_0`'s newly enabled
   SPIR-V 1.5 path and unblocks chipStar's `chipStar-llvm-20`
   compatibility window.
2. **Multi-target** (`LLVM_TARGETS_TO_BUILD="RISCV;X86"`) so the
   same clang serves both Vortex device compilation *and* x86_64
   host compilation. Required for any bespoke HIP path
   (`hip_support_proposal.md` Path A) and for chipStar's host
   bits.
3. **`lld` shipped** (`LLVM_ENABLE_PROJECTS="clang;lld"`). Avoids
   the "ld.lld doesn't exist" build failures encountered with our
   previous LLVM 18 install.
4. **Build-time defaults baked in** — host-default triple (no
   `LLVM_DEFAULT_TARGET_TRIPLE`), `GCC_INSTALL_PREFIX=/usr`,
   `CLANG_DEFAULT_LINKER=lld`, plus a `bin/clang.cfg` pinning
   `--gcc-install-dir` to the system gcc-11's libstdc++ headers
   (we have gcc-12 binaries on most boxes but only gcc-11 dev
   headers).
5. **One-time investment** sized to last ~12 months without
   another forced migration. Quarterly rebases onto LLVM 20.x
   point releases are routine. A LLVM 21+ bump becomes a
   separate, deliberate proposal at that point.

Out of scope: chipStar's *own* LLVM fork (`CHIP-SPV/llvm-project`'s
`chipStar-llvm-20` branch). Their patches are mostly clang HIP-mode
SPIR-V emission tweaks; we evaluate during the chipStar recon
resume whether to absorb any of them into `vortex_3.x`. Default
position: track upstream LLVM, not chipStar's fork.

---

## 2. Background — current state

| Property | Current `llvm_vortex` |
|---|---|
| Branch | `vortex_2.x` (on `github.com/vortexgpgpu/llvm`) |
| HEAD | `d78d4a25e` (2025-08-08, "fixes to the cross-function branch divergence analysis") |
| LLVM version | 18.1.7 |
| Targets | RISCV only |
| Projects | clang only (no `lld`) |
| Default triple | `riscv32-unknown-elf` (forces explicit `--target=` for host compiles) |
| Default sysroot | `$TOOLDIR/riscv32-gnu-toolchain/riscv32-unknown-elf` |
| Vortex-specific RISCV patches | `llvm/lib/Target/RISCV/VortexBranchDivergence.{cpp,h}`, `llvm/lib/Target/RISCV/VortexIntrinsicFunc.cpp`, plus `+vortex` and `+zicond` target features |
| Build configuration | `BUILD_SHARED_LIBS=ON` (libLLVM-18.so.*); `GCC_INSTALL_PREFIX` not set; no `clang.cfg` |
| `ld.lld` | **not built / not installed** — `lld` project not enabled in current build |
| chipStar interop | Earlier session demonstrated chipStar finds and accepts the Vortex device on this LLVM 18 build; full chipStar test paused for separate reasons ([chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)) |

### Pain points the current state has produced

- The chipStar recon hit `Executable "ld.lld" doesn't exist!` because
  our LLVM never built `lld`.
- chipStar's `enable_language(CXX)` failed because our default
  triple = riscv32 makes a casual `clang foo.cpp` a cross-compile.
- Adding X86 backend support required a multi-hour mid-recon
  rebuild.
- Header autodetection picked gcc-12 (newest binary) but the
  matching dev package isn't installed; needed a `clang.cfg`
  override for `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11`.

This proposal absorbs all of those one-offs into a deliberate
migration.

---

## 3. Goals

### 3.1 LLVM 20.x base

Track `upstream/release/20.x` (or a specific 20.x tag — see §10).

- LLVM 20 is POCL `release_7_0`'s newest supported window
  ([pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md)).
- Unlocks SPIR-V 1.5 path in POCL when needed.
- Inside chipStar's stated supported window (LLVM 18–20).

### 3.2 Multi-target (RISCV + X86), lld shipped

```
-DLLVM_TARGETS_TO_BUILD="RISCV;X86"
-DLLVM_ENABLE_PROJECTS="clang;lld"
```

- One clang serves device (RISCV+Vortex) and host (x86_64) compilation.
- `ld.lld` ships with the toolchain → no system linker dependency.
- Required by bespoke HIP via Clang HIP mode ([hip_support_proposal.md](hip_support_proposal.md) Path A).
- Required by chipStar when we resume — chipStar's `hipcc` uses
  this clang for both host and device passes.

### 3.3 Vortex ISA extension patches ported forward

Two small Vortex-specific source files in
`llvm/lib/Target/RISCV/`:

| File | Purpose | Lines (current) | Port effort estimate |
|---|---|---:|---|
| `VortexBranchDivergence.cpp` | Vortex cross-function branch-divergence analysis pass | ~few hundred | Low–Medium — LLVM 18 → 20 may have minor API drift in `MachineFunctionPass` / `LoopAnalysis` / `RegisterPassParser` |
| `VortexBranchDivergence.h` | Header for the above | small | Low |
| `VortexIntrinsicFunc.cpp` | Vortex-specific built-in intrinsic lowering | ~few hundred | Low–Medium |

Plus target-feature flags (`+vortex`, `+zicond`) in
`RISCVFeatures.td` / `RISCVProcessors.td` — small entries that
need to be re-applied onto upstream's tablegen tables.

Approach: **cherry-pick** the Vortex commits from `vortex_2.x`
onto a new branch from `release/20.x`. Resolve API-drift conflicts
where they arise.

### 3.4 Build-time defaults

Bake the workarounds we discovered into the install:

| Config | Value | Rationale |
|---|---|---|
| `LLVM_DEFAULT_TARGET_TRIPLE` | (unset; let `llvm_release_200` default to host) | A RISC-V default makes every `clang foo.c` a cross-compile and breaks chipStar/host build tools. Device code passes explicit `--target=riscv{32,64}-unknown-elf`. |
| `DEFAULT_SYSROOT` | (unset) | Same reason. |
| `GCC_INSTALL_PREFIX` | `/usr` | Clang picks the system gcc-toolchain at runtime; without this it can pick a wrong-version gcc binary. |
| `CLANG_DEFAULT_LINKER` | `lld` | `-fuse-ld=lld` is the default; `ld.lld` ships alongside. |
| Installed `bin/clang.cfg` (and `clang++.cfg`) | `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11` | Lock libstdc++ to gcc-11 (which has dev headers installed on our target boxes); avoid silent fall-through to gcc-12 binary with no headers. Project-local; tweak per machine. |
| `BUILD_SHARED_LIBS` | `ON` | Smaller binaries; SPIRV-LLVM-Translator and downstream consumers link the shared `libLLVM`. |

### 3.5 One-time investment & maintenance cadence

The whole migration is meant to last ~12 months without another
forced major bump. Cadence after this lands:

- **Quarterly** — rebase `vortex_3.x` onto the next LLVM 20.x
  point release (`20.1`, `20.2`, ...). Routine work, no API
  surprises within 20.x.
- **Annual / on-demand** — evaluate LLVM 21+ bump. Triggered by
  POCL upstream dropping support for LLVM 20 *or* a chipStar
  bugfix that's LLVM 21-only.
- **`vortex_2.x` deprecation** — once `vortex_3.x` is green
  through Phase 3 (validation), tag `vortex_2.x` final commit and
  freeze. New work goes to `vortex_3.x` only.

### 3.6 Other recommendations

- **Compiler-rt and musl libc rebuild** ([building_toolchain.md](../building_toolchain.md) §4 and §5) — the existing `libcrt32`/`libcrt64`/`libc32`/`libc64` installs were built with LLVM 18's clang. They'll continue working with LLVM 20 (RISC-V ABI is stable), but it's worth rebuilding to verify and to pick up any clang codegen improvements. Lower priority than the LLVM migration itself.
- **SPIRV-LLVM-Translator** — rebuild against LLVM 20 on the `llvm_release_200` branch. We already do this as part of the toolchain build per [building_toolchain.md](../building_toolchain.md) §3.
- **CI** — add a `.github/workflows/llvm_vortex_build.yml` that smoke-builds `vortex_3.x` against a known-good base. Catches LLVM upstream merges that break the Vortex passes.

---

## 4. Target architecture

```
github.com/vortexgpgpu/llvm
   ├── vortex_2.x         (frozen at d78d4a25e, LLVM 18.1, RISCV-only)
   └── vortex_3.x         (new — LLVM 20.x + Vortex patches)
        │
        ▼  cmake configure
        │       LLVM_TARGETS_TO_BUILD="RISCV;X86"
        │       LLVM_ENABLE_PROJECTS="clang;lld"
        │       BUILD_SHARED_LIBS=ON
        │       GCC_INSTALL_PREFIX=/usr
        │       CLANG_DEFAULT_LINKER=lld
        │
        ▼  install to $TOOLDIR/llvm-vortex/
            bin/clang, bin/clang++           (host x86_64 + Vortex RISCV)
            bin/ld.lld                       (linker)
            bin/llvm-objdump, llvm-objcopy   (binary tools)
            bin/clang.cfg                    (--gcc-install-dir pin)
            lib/libLLVM-20.so.*
            lib/cmake/llvm/                  (CMake export for SPIRV-LLVM-Translator etc.)
            include/                         (LLVM/Clang headers)
```

Downstream consumers (no change to their build recipes beyond
pointing `LLVM_DIR` / `WITH_LLVM_CONFIG` at the new install):

- `pocl_vortex/vortex_3.x` (post-Phase-2 KMU, post-POCL-7.0 rebase per [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md))
- `SPIRV-LLVM-Translator` (rebuilt on `llvm_release_200`)
- Eventually, `hip_vortex` (the `HIPVortex.cpp` toolchain per [hip_support_proposal.md](hip_support_proposal.md) Path A)
- Eventually, chipStar (built against `llvm_vortex` 20 instead of `chipStar-llvm-20` — pending compatibility validation)

---

## 5. Inventory — what stays, what's new, what to drop

### 5.1 Stays (port forward to LLVM 20)

| Path in `vortex_2.x` | Disposition |
|---|---|
| `llvm/lib/Target/RISCV/VortexBranchDivergence.{cpp,h}` | Port. Check `MachineFunctionPass`/`LoopAnalysis`/`PassRegistry` API changes between 18 → 20. |
| `llvm/lib/Target/RISCV/VortexIntrinsicFunc.cpp` | Port. Check builtin/intrinsic registration API. |
| `+vortex` and `+zicond` target features in `RISCVFeatures.td` | Re-apply onto upstream's tablegen. |
| Any Vortex-specific intrinsic declarations in `IntrinsicsRISCV.td` or `IntrinsicsRISCVVortex.td` | Audit + port. |
| Any clang-side Vortex changes (driver flags, target attributes) | Audit (likely few/none). |

### 5.2 New on `vortex_3.x` (compared to `vortex_2.x`)

| Item | Why |
|---|---|
| `lld` project enabled in build | §3.2 |
| `X86` target backend | §3.2 |
| `GCC_INSTALL_PREFIX=/usr` | §3.4 |
| `CLANG_DEFAULT_LINKER=lld` | §3.4 |
| `bin/clang.cfg` installed | §3.4 |
| LLVM 20.x base | §3.1 |

### 5.3 Dropped on `vortex_3.x`

| Item | Why |
|---|---|
| `LLVM_DEFAULT_TARGET_TRIPLE=riscv32-unknown-elf` | §3.4 — breaks host tool builds |
| `DEFAULT_SYSROOT=$TOOLDIR/riscv32-gnu-toolchain/riscv32-unknown-elf` | §3.4 — same |
| Any vortex_2.x-only LLVM passes we no longer use (audit first) | TBD during Phase 0 inventory |

---

## 6. Phases

### Phase 0 — Inventory + branch creation  (1–2 days)

- [ ] Enumerate every Vortex-specific commit on `vortex_2.x` after
  its fork point. Build a portable cherry-pick set.
- [ ] Create `vortex_3.x` from `upstream/release/20.x` (specific
  20.x tag pinned in §10).
- [ ] Cherry-pick the Vortex commits onto the new base.
- [ ] Resolve API-drift conflicts in `VortexBranchDivergence.cpp` /
  `VortexIntrinsicFunc.cpp` against LLVM 20's
  `MachineFunctionPass` / `Pass` / `RegisterPass` APIs.

**Validation gate**: `cmake -B build -DLLVM_TARGETS_TO_BUILD="RISCV;X86" -DLLVM_ENABLE_PROJECTS="clang;lld" ... && cmake --build build` completes green, including the Vortex tablegen for `+vortex`/`+zicond`.

### Phase 1 — Smoke tests on the new clang  (1 day)

- [ ] `clang --print-targets` lists both `riscv32` (with `+vortex`) and `x86-64`.
- [ ] `clang -target riscv32-unknown-elf -march=rv32imaf -mabi=ilp32f -mcmodel=medany -c hello.c` produces a valid Vortex object.
- [ ] `clang++ hello.cpp -o hello` (host) produces a working x86_64 executable.
- [ ] `clang -target riscv32-unknown-elf -Xclang -target-feature -Xclang +vortex` emits Vortex-specific intrinsics correctly (one targeted IR test).
- [ ] `ld.lld --version` reports LLVM 20.

**Validation gate**: all four work without environment-variable contortions.

### Phase 2 — Rebuild downstream toolchain stack  (1–2 days)

- [ ] Rebuild `compiler-rt` for RISCV against LLVM 20's clang (32 + 64-bit baremetal) → `libcrt32`/`libcrt64`.
- [ ] Rebuild musl libc against LLVM 20's clang (32 + 64-bit) → `libc32`/`libc64`.
- [ ] Rebuild `SPIRV-LLVM-Translator` on `llvm_release_200` branch against LLVM 20 → `bin/llvm-spirv`.

### Phase 3 — Integration validation  (1–2 days)

- [ ] Build `pocl_vortex/vortex_3.x` (post-POCL-7.0-rebase per [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md)) against new LLVM.
- [ ] `tests/opencl/vecadd` passes on Vortex device on SimX (legacy launch path).
- [ ] SPIR-V vecadd through `clCreateProgramWithIL` passes (POCL 7.0 SPIR-V path).
- [ ] KMU-dispatched OpenCL kernel passes (Phase 2 of pocl_vortex proposal).
- [ ] (deferred) chipStar HIP test resumes per [chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md).

**Validation gate**: same set of tests that pass on current LLVM 18 also pass on LLVM 20.

### Phase 4 — Hygiene  (1 day)

- [ ] Update `docs/building_toolchain.md` §3 to reflect: LLVM 20.x base, `+X86` and `+lld`, the new install-time defaults.
- [ ] Add `.github/workflows/llvm_vortex_build.yml` — periodic smoke build of `vortex_3.x` against current `release/20.x` HEAD.
- [ ] Rewrite `vortexgpgpu/llvm` README to describe `vortex_3.x` as the active branch; mark `vortex_2.x` as frozen with the deprecation date.
- [ ] Document the quarterly rebase recipe in the README.

### Phase 5 — Frozen `vortex_2.x` cutover

- [ ] Tag `vortex_2.x` final at its current HEAD (`d78d4a25e` or wherever it lands at cutover time).
- [ ] Switch `pocl_vortex`, `hip_vortex`, and any future bespoke HIP work to default to `vortex_3.x`.
- [ ] Update `docs/building_toolchain.md` to point at `vortex_3.x` as the canonical install.

---

## 7. Test plan

| Stage | Test | What it validates |
|---|---|---|
| Phase 0 | LLVM build green | Cherry-picks applied; API drift resolved |
| Phase 0 | tablegen for `+vortex`/`+zicond` produces same target-feature output as `vortex_2.x` | Target features ported correctly |
| Phase 1 | `clang --target=riscv32-unknown-elf` produces same `.o` symbol layout as `vortex_2.x` for a reference kernel | No codegen regression |
| Phase 1 | `clang -target x86_64-linux-gnu hello.cpp` runs | Host backend works |
| Phase 2 | `compiler-rt` test (RISCV builtin) | LLVM 20 doesn't break the lib build |
| Phase 2 | `musl-libc` test | Same |
| Phase 2 | `llvm-spirv` round-trip a kernel | SPIRV-LLVM-Translator on 20 works |
| Phase 3 | All `tests/opencl/*` pass on `pocl_vortex` against new LLVM | End-to-end |
| Phase 3 | KMU launch test (legacy + SPIR-V) passes | Phase 2 of pocl_vortex_v3 still green |
| Phase 4 | CI workflow green on next push | Regression catch on future upstream LLVM bumps |

---

## 8. Risks

1. **LLVM 18 → 20 API drift in Vortex passes.** `MachineFunctionPass`,
   `LoopInfo`, `PassRegistry`, intrinsic lowering — all have had
   minor revisions. `VortexBranchDivergence.cpp` is the highest-
   risk file (interacts with multiple LLVM analysis APIs).
   **Mitigation**: do the port early in Phase 0; if API drift is
   severe, consider holding at LLVM 19 as an intermediate.

2. **`compiler-rt`/`musl-libc` LLVM-20 incompatibility.** Less
   likely (these libs are stable across LLVM versions), but
   possible if clang's RISC-V codegen produced something the libs
   weren't built for.
   **Mitigation**: Phase 2 rebuilds them against new clang.

3. **chipStar's `chipStar-llvm-20` patches.** We're tracking
   upstream LLVM, not chipStar's fork. If chipStar carries
   non-trivial fixes (SPIR-V emission bugs, HIP-mode fixes) we'd
   miss them.
   **Mitigation**: during chipStar recon resume, evaluate each
   chipStar LLVM patch; cherry-pick into `vortex_3.x` if needed.
   Default: track upstream.

4. **Vortex passes block upstreaming.** Long-term, having
   Vortex-specific source in `llvm/lib/Target/RISCV/` makes
   rebases expensive. **Mitigation**: keep the Vortex patch set
   tightly scoped; consider an out-of-tree pass plugin
   architecture as a separate effort (post-12-month).

5. **The 12-month bet is wrong.** LLVM moves fast; some downstream
   project might force an earlier bump (e.g., chipStar dropping
   LLVM 20 support, POCL upgrading their minimum). **Mitigation**:
   none — accept the risk; the 12-month bet is a budget, not a
   guarantee.

---

## 9. Timeline

| Phase | Days | Cumulative |
|---|---:|---:|
| 0: branch + inventory + cherry-pick + green build | 1–2 | 2 |
| 1: smoke tests on clang | 1 | 3 |
| 2: rebuild compiler-rt / musl-libc / SPIRV-LLVM-Translator | 1–2 | 5 |
| 3: integration validation against `pocl_vortex` | 1–2 | 7 |
| 4: hygiene (docs, CI) | 1 | 8 |
| 5: `vortex_2.x` cutover | 0 (administrative) | 8 |

**Total: ~1.5 weeks (8 working days), with a soft cap at 2 weeks
before reassessment.** Phase 0 + Phase 3 are the riskiest;
budget can flex within those.

---

## 10. Decisions to make before Phase 0

1. **Exact LLVM 20 base.** `release/20.x` branch tip, or a tagged
   `llvmorg-20.x.y` release? **Default: a tagged release** (most
   recent point release at start of Phase 0). Tags are
   reproducible; branch tips drift. The tag is pinned in this
   proposal once chosen.
2. **`vortex_3.x` vs `release/20.x-vortex` branch naming.**
   **Default: `vortex_3.x`** — mirrors `pocl_vortex/vortex_3.x`
   and the broader v3 stack convention.
3. **chipStar LLVM patches.** Cherry-pick none now, evaluate when
   chipStar recon resumes? **Default: yes — track upstream.**
4. **Whether to bump `compiler-rt`/`musl-libc` versions.**
   `compiler-rt` should match LLVM 20.x. musl libc is independent
   — current `v1.2.5` is fine. **Default: bump compiler-rt only.**
5. **`vortex_2.x` deprecation date.** When to freeze? **Default:
   when `pocl_vortex/vortex_3.x` is green on the new LLVM
   (Phase 3 exit).**
