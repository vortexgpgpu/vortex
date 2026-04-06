---
name: Vortex coding conventions and debugging guide
description: Must-read docs for every session — coding guidelines (Verilog/C++) and debugging procedures
type: reference
---

Always read these files at the start of work to follow Vortex conventions:

- `docs/coding_guidelines_verilog.md` — Verilog style rules
- `docs/coding_guidelines_cpp.md` — C++ style rules
- `docs/debugging.md` — How to debug SimX, RTL, FPGA

## Key Verilog conventions (docs/coding_guidelines_verilog.md)

- 4 spaces indent, no tabs
- Modules: `VX_PascalCase`, signals: `lower_snake_case`, macros/params: `UPPER_SNAKE_CASE`
- Generate blocks prefixed `g_`
- Warnings: resolve at source using VX_platform.vh macros (`UNUSED_VAR`, `UNUSED_PARAM`, `UNUSED_PIN`, etc.). verilator.vlt is only for third-party code.
- `ifdef`/`else`/`endif` indented one level left of guarded code (not column 0)
- `TRACE` macro arguments must be comma-separated
- `RUNTIME_ASSERT(cond, (fmt, args))` for runtime checks
- `STATIC_ASSERT(cond, (msg))` for parameter validation

## Key C++ conventions (docs/coding_guidelines_cpp.md)

- 2 spaces indent, no tabs
- K&R brace style (opening brace on same line)
- Constructor initializer lists: each on own line, aligned under colon
- `#ifdef` indented one level left of guarded code
- `__unused(var)` for unused parameters
- Debug macros (`DT`, `DPN`, etc.) arguments must be comma-separated

## Debugging (docs/debugging.md)

- `./ci/blackbox.sh --driver=simx --app=demo --debug=3` — SimX trace → run.log
- `./ci/blackbox.sh --driver=rtlsim --app=demo --debug=3` — RTL trace → run.log + trace.vcd
- `--rebuild=1` to force driver rebuild after RTL/sim changes
- `CONFIGS="-DTRACING_ALL"` for full waveform tracing
- CSV trace comparison for SimX vs RTLsim divergence:
  ```
  ./ci/trace_csv.py -trtlsim run_rtlsim.log -otrace_rtlsim.csv
  ./ci/trace_csv.py -tsimx run_simx.log -otrace_simx.csv
  diff trace_rtlsim.csv trace_simx.csv
  ```
- First column is UUID — use it to correlate same instruction across SimX and RTL
- `--saif` flag for switching activity (power analysis)
- FPGA scope: build with `SCOPE=1`, run with `--scope`, signals defined in `hw/scripts/scope.json`
