# AGENTS Guide for Vortex GPGPU Development

This file contains context and guidelines for coding agents working in this repository.

## Vortex Documentation Map

- `docs/codebase.md` (repo file tree map)
- `docs/install_vortex.md` (initial install + build setup)
- `docs/testing.md` (test and regression flow)
- `docs/debugging.md` (debug traces and workflows)
- `docs/simulation.md` (driver modes and blackbox usage)
- `docs/coding_guidelines_cpp.md` (C++ style guidelines)
- `docs/coding_guidelines_verilog.md` (Verilog style and verilator warning/linting guidelines)
- `docs/contributing.md` (contribution guidelines and PR process)
- `docs/perfetto_analysis.md` (perfetto trace and analysis guide)

## Build Directory Setup

```bash
# From repo root:
mkdir -p build && cd build

# Configure word size (pick one XLEN):
# 32-bit
../configure --xlen=32 --tooldir=$HOME/tools
# 64-bit
../configure --xlen=64 --tooldir=$HOME/tools

# Install toolchain (first time / when missing):
./ci/toolchain_install.sh --all

# Source toolchain environment (every new shell):
source ./ci/toolchain_env.sh

# Build
make -s
```

Wait until build completes before running anything else in parallel terminals.

`configure` generates a runnable build tree by copying and instantiating `ci/`, `runtime/`, `sim/`, and `tests/` into `build/`. For execution and test automation, prefer the generated scripts and Makefiles under `build/` over the source-tree `.in` files.

### Common Gotchas:

- If you git pull from origin, modify Makefiles or add/remove directories that participate in build logic, remember to re-run configure from `build/`

    ```bash
    ../configure
    make -C hw clean && make -C hw
    ```

- Prefer separate build dirs for major variants (example: `build32/`, `build64/`) to avoid config/tool contamination.
- `./ci/blackbox.sh` rebuilds the selected runtime driver with `CONFIGS` / `--cores` / `--warps` / `--threads`, but it does **not** rebuild the target app with matching compile-time macros. If the app binary already exists and was built with different values (for example `NUM_THREADS=4`), blackbox can launch a mismatched binary and fail host-side checks.

    ```bash
    # Rebuild the test/app with the same compile-time overrides first
    make -C tests/regression/sgemm_tcu clean
    CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE" make -C tests/regression/sgemm_tcu

    # Then run blackbox with matching runtime overrides
    CONFIGS="-DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu --threads=8
    ```

- `make tests` and `make -C tests/regression` build test binaries using their default macros. If you intend to run a test with non-default thread count, data types, or feature flags use `CONFIGS` + `blackbox.sh`

## Testing & Debugging

All commands below MUST run directly from the `build/` directory (where generated `ci/` scripts and resolved env are expected).

### Smoke tests

```bash
./ci/blackbox.sh --driver=simx --app=demo
./ci/blackbox.sh --driver=simx --app=sgemm --args="-n10"
```

### Regression suites

```bash
make -C tests/regression run-simx
make -C tests/regression run-rtlsim

make -C tests/opencl run-simx
make -C tests/opencl run-rtlsim
```

### Targeted test during development

```bash
# build specific test
make -C tests/regression/<test-name>

# run specific test with debug log
./ci/blackbox.sh --driver=rtlsim --app=<test-name> --debug=1 --log=run.log
```

When using non-default compile-time macros, pass them directly via `CONFIGS` on the same command:

```bash
CONFIGS="-DNUM_THREADS=8 -DITYPE=fp16 -DOTYPE=fp32 -DEXT_TCU_ENABLE" \
./ci/blackbox.sh --driver=simx --app=<test-name> --threads=8 --args="..."
```

### Roofline Perf Plot

Example to run sgemm_tcu test with perf collection and generate roofline plot (Peak vs Actual FLOPS, Compute vs Memory BW)
```bash
/usr/bin/python3 ../perf/roofline.py --app=sgemm_tcu --driver=simx --cores=1 --warps=4 --threads=8 --issue-width=2 --n=32 --perf=1 --by-cycle --output=sgemm_tcu_roofline.png
```

For multi-suite coverage, `ci/regression.sh` is the canonical source of tested configurations. Use it to discover supported parameter combinations before inventing ad hoc ones.

## Configuring architecture parameters

Use command-line configuration overrides first; avoid editing baseline config in `VX_config.toml` unless needed.

### Direct command-line overrides

`blackbox.sh` supports direct architecture overrides:

- `--clusters=<n>`
- `--cores=<n>`
- `--warps=<n>`
- `--threads=<n>`
- `--l2cache`
- `--l3cache`
- `--debug=<level>`
- `--perf=<class>`

Example:

```bash
./ci/blackbox.sh --driver=simx --app=sgemm --clusters=1 --cores=2 --warps=4 --threads=4 --l2cache
```

### Advanced pass `CONFIGS` macros

For parameters not exposed as explicit flags, use `CONFIGS` with `-D...` overrides:

Example: To enable TCU, select FEDP backend, and set I/O data types:
```bash
CONFIGS="-DNUM_THREADS=4 -DEXT_TCU_ENABLE -DTCU_TYPE_TFR -DITYPE=bf16 -DOTYPE=fp32" \
./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu --threads=4
```

### Editing default config files

Default hardware/type parameters are defined in:

- `hw/VX_config.toml`
- `hw/VX_types.toml`

Re-run configure and build after changing baseline configuration files.

## Graphics extensions (TEX / RASTER / OM)

Three skybox-era extensions are integrated and gated behind separate
flags. All three are off by default; enabling them is opt-in via
`CONFIGS` per build/run.

| Extension | Flag | Op (CUSTOM1 / `INST_EXT2`) | Cluster path |
|---|---|---|---|
| Texture sampling | `EXT_TEX_ENABLE` | `vx_tex(stage, u, v, lod)` (R4, funct3=1) | per-socket → `VX_tex_arb` → `VX_tex_core` → `tcache` → L2 |
| Output merger    | `EXT_OM_ENABLE`  | `vx_om(x, y, face, color, depth)` (R4, funct3=2) | per-socket → `VX_om_arb` → `VX_om_core` → `ocache` → L2 |
| Rasterizer pop   | `EXT_RASTER_ENABLE` | `vx_rast()` (R, funct3=3, returns packed pos_mask `{pos_y, pos_x, mask}`, 0 ⇒ drained) | `VX_raster_core` → per-socket fan-out via `VX_raster_arb` |

DCR address space (host-side configuration writes):
- `VX_DCR_TEX_*`     `0x020..0x03F`
- `VX_DCR_RASTER_*`  `0x040..0x045`
- `VX_DCR_OM_*`      `0x060..0x071`

CSR address space (per-warp readback for raster bcoords):
- `VX_CSR_RASTER_*`  `0x7C0..0x7CC`
- `VX_CSR_OM_*`      `0x7CD..0x7CE`

### Graphics tests (rtlsim)

End-to-end PNG-driven tests live under `tests/regression/`:

```bash
CONFIGS="-DEXT_TEX_ENABLE"    ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-i palette64.png -r palette64_ref_g0.png"
CONFIGS="-DEXT_RASTER_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=raster --args="-w 32 -h 32 -t triangle.cgltrace -r triangle_ref_32.png"
CONFIGS="-DEXT_OM_ENABLE"     ./ci/blackbox.sh --driver=rtlsim --app=om --args="-w 32 -h 32 -c 16777215 -r whitebox_32.png"
# Full graphics matrix:
./ci/regression.sh --graphics
```

### Standalone unit elaboration

Each unit also has a verilator unittest at `hw/unittest/{tex,om,raster}_unit/`
that elaborates the cluster-shared unit in isolation (no SFU agent, no
cluster wrapper) — useful for fast iteration on RTL changes without
rebuilding the full driver:

```bash
make -C hw/unittest/tex_core run
make -C hw/unittest/om_core run
make -C hw/unittest/raster_core run
```

### SimX caveat

The C++ functional simulator (`--driver=simx`) does **not** yet model the
graphics units; running a `vx_tex/om/rast` op under simx triggers an
abort. Use `--driver=rtlsim` for any kernel that issues graphics ops.
SimX support is gated on simx_v3 Phase 5 (caches-carry-data) — see
[docs/proposals/gfx_migration_proposal.md](docs/proposals/gfx_migration_proposal.md)
§4 Phase 3.

### Migrating a skybox graphics test

The full skybox PNG-driven regression suites (`tests/regression/{tex,om,raster,draw3d}`)
were written for the **vortex v1 kernel API** (`int main()` + `vx_spawn_threads`).
Porting one requires translating to feature_gfx's vortex2 API
(`__kernel void kernel_main(...)` with CUDA-like `blockIdx`/`threadIdx`).
The smoke tests above show the v2 pattern; the host-side DCR-write
sequence translates directly from the skybox `TEX_DCR_WRITE` style.
Helpers `sim/common/{gfxutil,graphics}.{cpp,h}` and `third_party/cocogfx/`
are vendored from skybox and ready to link against.

## General Notes

- Keep changes minimal, targeted, and reversible.
- Fix root causes; avoid workaround-only patches.
- Add only concise, straightforward comments at the start of code sections (unless complex logic/targeted optimizations warrant more).
- If this guide conflicts with commands emitted by generated scripts under `build/ci/`, trust the generated scripts and update this file in the same change.
- AGENTS.md is intended to be living documentation; contributors and agents are encouraged to keep updating it as best practices and workflows evolve.
