# AGENTS Guide for Vortex GPGPU Development

This is the canonical entry point for **both human contributors and AI coding agents** working on Vortex. It captures the *rules* — invariants, conventions, and footguns — that every change should respect. Build/test/debug *recipes* live in the topic docs linked below.

---

## 1. Documentation Map

### Setup & build
- [docs/install_vortex.md](docs/install_vortex.md) — initial install + build setup
- [docs/building_toolchain.md](docs/building_toolchain.md) — building Verilator, RISC-V GNU, LLVM, compiler-rt, musl, POCL from source
- [docs/environment_setup.md](docs/environment_setup.md) — environment variables and toolchain layout
- [docs/fpga_setup.md](docs/fpga_setup.md) — FPGA target setup

### Codebase orientation
- [docs/codebase.md](docs/codebase.md) — repo file-tree map
- [docs/microarchitecture.md](docs/microarchitecture.md) — pipeline & cache architecture
- [docs/cache_subsystem.md](docs/cache_subsystem.md) — cache subsystem details
- [docs/hardware_library.md](docs/hardware_library.md) — `hw/rtl/libs/` reference

### Coding conventions
- [docs/coding_guidelines_cpp.md](docs/coding_guidelines_cpp.md) — C++ style
- [docs/coding_guidelines_verilog.md](docs/coding_guidelines_verilog.md) — Verilog/SystemVerilog style

### Simulation & test
- [docs/simulation.md](docs/simulation.md) — driver modes (simx, rtlsim, opae, xrt) and blackbox usage
- [docs/testing.md](docs/testing.md) — test and regression flow
- [docs/debugging.md](docs/debugging.md) — debug traces (`--debug`), VCD, scope, trace_csv
- [docs/debug_mode.md](docs/debug_mode.md) — debug-mode hardware support
- [docs/perfetto_analysis.md](docs/perfetto_analysis.md) — Perfetto trace and analysis
- [docs/synthesis_analysis.md](docs/synthesis_analysis.md) — synthesis/PPA analysis

### Process
- [CONTRIBUTING.md](CONTRIBUTING.md) — public fork/PR contribution flow
- [docs/bug_fixes.md](docs/bug_fixes.md) — bug-fix discipline (root-cause vs patch)
- [docs/continuous_integration.md](docs/continuous_integration.md) — CI pipeline
- [docs/proposals/](docs/proposals/) — design and migration proposals (drafts and in-progress)
- [docs/designs/](docs/designs/) — accepted designs (post-proposal, post-implementation)

---

## 2. Build & Toolchain Rules

See [docs/install_vortex.md](docs/install_vortex.md) for the full recipe. The non-negotiable rules:

- **Out-of-tree.** From the repo root:
  ```bash
  mkdir -p build && cd build
  ../configure --xlen=64 --tooldir=$HOME/tools   # or --xlen=32
  ./ci/toolchain_install.sh                # first time only
  make -s
  ```
- **Separate build dirs per major variant** (`build32/`, `build64/`, `build_fpga64/`, ...) to avoid config/tool contamination. Never reuse one build dir for incompatible configurations.
- **`configure` generates a runnable tree** by copying and instantiating `ci/`, `runtime/`, `sim/`, and `tests/` into `build/`. For execution and test automation, *always* prefer the generated scripts/Makefiles under `build/` over the source-tree `.in` files.
- **Re-`../configure` from `build/`** whenever you `git pull`, edit source Makefiles, edit `VX_config.toml` / any `*.toml`, or add/remove a build-participating directory. Symptom of forgetting this: stale binaries, missing targets, or "I edited this Makefile and nothing happened."
- **Always ensure the build is current before running any test or app** — re-run `../configure` from `build/` first. `configure` regenerates `<build>/sw/VX_config.h` and `<build>/hw/*.vh` from `VX_config.toml`, but only when the toml is newer (it guards on mtime). The simx/RTL **cores `#include` this generated header**, so a stale header makes a core compile against old config values and silently diverge from the toml — and from the runtime/RTL, which re-expand the config every build. This is a real footgun: a stale `VX_config.h` once made SimX run a write-back D-cache while the toml/RTL were write-through, producing SimX-only wrong results. **Never** work around such a divergence by injecting `-DVX_CFG_*` overrides into a Makefile — that masks the stale artifact and fights the config system. The toml is the single source of truth; fix it by re-`configure`-ing.
- **`ccache` can serve stale objects on `fmt`-version mismatches** (typical symptom: `fmt::v8` undefined-reference link errors in sim builds). Before deep-diving, retry with `CCACHE_DISABLE=1`.

---

## 3. Bug-Fix Rules

See [docs/bug_fixes.md](docs/bug_fixes.md) for the full rationale and examples. The rules:

- **Fix root causes, not symptoms.** Diagnose before patching.
- **Don't paper over upstream regressions** or mask bugs with fallback paths and suppressed warnings.
- **If a patch is genuinely unavoidable** (e.g. blocked by an external dep), label it as a patch explicitly *in the commit message* and pair it with a follow-up to do the proper fix.

---

## 4. Testing & Verification Rules

See [docs/testing.md](docs/testing.md) and [docs/debugging.md](docs/debugging.md) for recipes. The rules:

- **All test commands run from `build/`.** The generated `ci/` scripts assume it.
- **120-second timeout cap** on every test invocation. No exceptions.
- **`CONFIGS` must match on both sides.** `blackbox.sh` only rebuilds the driver. If your test was compiled with `-DVX_CFG_NUM_THREADS=4`, blackbox can't fix that — rebuild the app first:
  ```bash
  make -C tests/regression/<app> clean
  CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_EXT_TCU_ENABLE" make -C tests/regression/<app>
  CONFIGS="-DVX_CFG_EXT_TCU_ENABLE" ./ci/blackbox.sh --driver=simx --app=<app> --threads=8
  ```
- **`make tests` / `make -C tests/regression` build with *default* macros.** Use `CONFIGS` + explicit per-app rebuild for non-default configurations.
- **`--rebuild=1` forces a driver rebuild** even if the hardware configuration is unchanged. Use it when iterating on the driver itself; `--rebuild=0` suppresses rebuild regardless.
- **RTL coverage path is `xrt`, not `rtlsim`.** When discussing or planning RTL verification, `xrt` is the canonical path — `rtlsim` bypasses the AFU surface. `rtlsim` remains useful for fast iteration on processor RTL; `xrt` is what proves the full integration.
- **`ci/regression.sh` is the canonical source of tested configurations.** Use it to discover supported parameter combinations before inventing ad hoc ones.
- **When RTL debugging stalls, switch to the SimX-as-oracle pattern.** For numerical bugs, deep pipeline races, or any failure mode where rtlsim is "close but wrong": (1) build/extend the SimX C++ model so it mirrors the *new* RTL architecture and gets to PASS; (2) add matching trace dumps to both SimX and RTL (cycle, FU events, SRAM addresses+data, hazards) — same CSV format on both sides; (3) diff trace files — the first divergence is the bug. Don't keep guessing from output values; localize via trace diff. See [docs/debugging.md](docs/debugging.md#simx-as-oracle-for-rtl-debug).

### Smoke tests

```bash
./ci/blackbox.sh --driver=simx --app=demo
./ci/blackbox.sh --driver=simx --app=sgemm --args="-n10"
```

### Regression suites

```bash
make -C tests/regression run-simx
make -C tests/regression run-rtlsim
make -C tests/opencl     run-simx
make -C tests/opencl     run-rtlsim
```

### Architecture overrides

`blackbox.sh` exposes the common knobs directly: `--clusters=`, `--cores=`, `--warps=`, `--threads=`, `--l2cache`, `--l3cache`, `--debug=`, `--perf=`. For anything not exposed as a flag, use `CONFIGS="-D..."` (all parameters take the `VX_CFG_*` prefix — e.g. `-DVX_CFG_NUM_THREADS=8`, `-DVX_CFG_EXT_TCU_ENABLE`). Baseline parameters live in `VX_config.toml` and `VX_types.toml` at the repo root — edit those only when an override is needed for *all* builds, and re-`configure` afterward.

---

## 5. Design & Architecture Rules

- **Align with mainstream CUDA, HIP, OpenCL, and Vulkan API and driver stacks.** For any design question — driver surface, command-processor model, memory model, scheduling — pick the solution those stacks would use. This keeps Vortex's externals familiar to mainstream software and avoids one-off abstractions.

---

## 6. Coding Conventions

See [docs/coding_guidelines_cpp.md](docs/coding_guidelines_cpp.md) and [docs/coding_guidelines_verilog.md](docs/coding_guidelines_verilog.md) for full style. Cross-cutting comment rules:

- **`sw/{kernel,runtime}/` and `sim/`/`hw/` are bidirectionally isolated.** Source files under `sw/kernel/` and `sw/runtime/` MUST NOT reference anything in `hw/*` or `sim/*`. Equally, source files under `sim/*` and `hw/*` MUST NOT reference anything in `sw/kernel/` or `sw/runtime/`. If the layers genuinely need to share host-side helper code or an on-wire ABI definition, put it in `sw/common/` — vortex-internal, never installed, accessible from all four layers. Enforced by [ci/check_sw_sim_boundary.sh](ci/check_sw_sim_boundary.sh).
- **Default to no comment.** Add one only when the *why* is non-obvious — a hidden constraint, a subtle invariant, a workaround for a specific bug, a surprising behavior.
- **Comments explain *why*, not *what*.** Well-named identifiers carry the *what*.
- **Never reference the current task, PR, caller, issue number, or fix.** Those belong in the commit message and rot in code. Don't write `// used by foo()`, `// added for the X flow`, `// handles issue #123`.
- **`RTL_PKGS` is for `VX_*_pkg.sv` only.** Verilog interfaces are illegal there. If an interface isn't being discovered, fix it via include paths or file naming — not by stuffing it into `RTL_PKGS`.
- **Library RTL defaults to `TRACING_OFF`.** Modules under `hw/rtl/libs/` are excluded from VCD by default to keep waveform size manageable. Toggle per-file with `TRACING_ON`/`TRACING_OFF`, or globally with `CONFIGS="-DTRACING_ALL"`.
- **No multi-paragraph docstrings or multi-line block comments** for trivial code. One short line is the ceiling unless the logic genuinely needs more.
- **No `// removed code` or stale-rename breadcrumbs.** If something is unused, delete it.

---

## 7. Proposals

Design and migration proposals belong in [`docs/proposals/`](docs/proposals/) — never in build dirs or the repo root. A proposal captures the *why* of a non-trivial change before the code lands.

Accepted designs that outlive their proposal phase live in [`docs/designs/`](docs/designs/).

---

## 8. Living Document Policy

This file is versioned alongside the code and evolves with it. Both humans and agents are expected to update it:

- **Add a rule** when you discover a footgun the next contributor will hit. Include the *why*.
- **Update or remove a rule** that's gone stale. A stale rule is worse than no rule.
- **Don't duplicate** content from the topic docs in `docs/`. AGENTS.md links; the topic doc explains.
- **Branch-specific addenda are forbidden.** Branch context goes in a `docs/proposals/<feature>_proposal.md`, not in branch-local `AGENTS.md` variants. If the rule is general, promote it here; if it's a project-tracking detail, keep it in the proposal.
