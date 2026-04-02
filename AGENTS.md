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

- `make tests` and `make -C tests/regression` build test binaries using their default macros. If you intend to run a test with non-default `NUM_THREADS`, data types, or feature flags, rebuild that specific test explicitly before invoking it.

## Testing & Debugging

All commands below should run from `build/` (where generated `ci/` scripts and resolved env are expected).

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

When using non-default compile-time macros, split the flow into an explicit rebuild step and a run step:

```bash
make -C tests/regression/<test-name> clean
CONFIGS="-DNUM_THREADS=8 -DITYPE=fp16 -DOTYPE=fp32 -DEXT_TCU_ENABLE" make -C tests/regression/<test-name>
CONFIGS="-DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=simx --app=<test-name> --threads=8 --args="..."
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
make -C tests/regression/sgemm_tcu clean && CONFIGS="-DNUM_THREADS=4 -DITYPE=bf16 -DOTYPE=fp32" make -C tests/regression/sgemm_tcu
CONFIGS="-DNUM_THREADS=4 -DEXT_TCU_ENABLE -DTCU_TYPE_TFR" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu
```

### Editing default config files

Default hardware/type parameters are defined in:

- `hw/VX_config.toml`
- `hw/VX_types.toml`

Re-run configure and build after changing baseline configuration files.

## General Notes

- Keep changes minimal, targeted, and reversible.
- Fix root causes; avoid workaround-only patches.
- Add only concise, straightforward comments at the start of code sections (unless complex logic/targeted optimizations warrant more).
- If this guide conflicts with commands emitted by generated scripts under `build/ci/`, trust the generated scripts and update this file in the same change.
- AGENTS.md is intended to be living documentation; contributors and agents are encouraged to keep updating it as best practices and workflows evolve.
