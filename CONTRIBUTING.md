# Contributing to Vortex

Thanks for your interest in Vortex! Contributions of all sizes are welcome —
bug reports, documentation fixes, new backends, RTL features, runtime work,
or test coverage.

## Getting in touch

The Vortex community lives at <https://vortex.cc.gatech.edu/community>.
Start there for questions, design discussions, and to find out what others
are working on before you sink time into a large change.

## Reporting bugs

Open an issue at <https://github.com/vortexgpgpu/vortex/issues> with:

- **What you ran** — the exact `./configure` flags, `make` target, and
  driver (`simx`, `rtlsim`, `xrt`, `opae`).
- **What you expected vs. what happened** — include the failing log,
  not just the final error line. Trim long traces but keep the surrounding
  context so we can reproduce.
- **Your environment** — OS, toolchain revision (`cat VERSION`), and
  whether you built the toolchain from `ci/toolchain_install.sh` or
  pulled the prebuilt drop.
- **A minimal reproducer**, if you have one — ideally a single
  `ci/blackbox.sh` invocation or a small kernel under `tests/`.

For suspected hangs, run with `DEBUG=3` and attach the trace; for
correctness issues, attach both the failing run and a known-good run
(e.g. with a smaller config or a different driver) so we can diff them.

## Submitting changes

External contributions go through GitHub pull requests against
`master`:

1. Fork the repo and branch from `master`.
2. Build out-of-tree and run the regression suite for the area you
   touched before submitting. See [docs/testing.md](docs/testing.md)
   for the full test catalog and
   [docs/continuous_integration.md](docs/continuous_integration.md)
   for how CI wires the same suite up under GitHub Actions:

   ```bash
   mkdir build && cd build
   ../configure --tooldir=$HOME/tools
   make -s
   ./ci/regression.sh --unittest
   ./ci/regression.sh --riscv
   ./ci/regression.sh --kernel
   ./ci/regression.sh --regression
   ```

   For HW changes also run `./ci/regression.sh --synthesis`;
   for runtime/driver work, run the test groups that exercise your
   backend (e.g. `--opencl`, `--vulkan`, `--cache`, `--cupbop`,
   `--hip`, `--gem5`, `--vm`, `--scope`).
3. Keep the PR focused. Unrelated cleanups belong in their own PR so
   reviewers can evaluate each change on its own merits.
4. Match the surrounding code style — Vortex doesn't enforce a strict
   formatter, but new code should look at home next to the file it
   sits in (C++17 in `sw/` and `sim/`, SystemVerilog conventions in
   `hw/rtl/`, Python in `ci/`).
5. Write commit messages that explain **why**, not just what. A diff
   already shows the what.

## Adding tests

Every functional change should land with a test, ideally one that
would have caught the bug or that exercises the new path:

- **RISC-V kernels** → `tests/regression/<name>/` (host + kernel +
  Makefile). Wire it into `tests/regression/Makefile` (`all`,
  `run-simx`, `run-rtlsim`, `clean`) and add a corresponding entry
  in [ci/regression.sh.in](ci/regression.sh.in) so CI picks it up.
- **Host-side / runtime unit tests** → `tests/unittest/`
  (`./ci/regression.sh --unittest` builds and runs them).
- **RTL unit tests (Verilator)** → `hw/unittest/<block>/` for
  block-level testbenches (cache, cp_*, tcu_fedp, raster_core, etc.).
  Add a subdir with `Makefile` + C++ harness following the
  surrounding convention; `--unittest` also runs `make -C hw/unittest`.
- **RTL behavior at the system level** → an `rtlsim` run of an
  existing kernel with a config that triggers the path, added to the
  relevant test group.

If a feature is intentionally incomplete and a test is expected to
fail, document the failure mode in
[ci/regression{32,64}_failures.md](ci/regression64_failures.md)
rather than disabling the test silently.

## Licensing

Vortex is released under the Apache License 2.0 (see [LICENSE](LICENSE)).
By submitting a contribution you agree that it may be released under
the same terms.
