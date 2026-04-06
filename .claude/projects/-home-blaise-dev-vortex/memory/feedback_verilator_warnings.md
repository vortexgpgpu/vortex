---
name: Verilator warning suppression
description: Never use verilator.vlt.in to suppress warnings; use VX_platform.vh macros or fix the code directly
type: feedback
---

Never modify verilator.vlt.in (or generated verilator.vlt) to silence Verilator warnings. Instead, use the Vortex compiler macros defined in VX_platform.vh:

- `UNUSED_VAR(x)` / `UNUSED_VAR({a, b, c})` — unused signals
- `UNUSED_PARAM(x)` — unused parameters
- `UNUSED_SPARAM(x)` — unused string parameters
- `UNUSED_PIN(x)` — unused module output pins
- `IGNORE_UNOPTFLAT_BEGIN` / `IGNORE_UNOPTFLAT_END` — cyclic combinational logic Verilator can't resolve
- `IGNORE_UNUSED_BEGIN` / `IGNORE_UNUSED_END` — block-level unused suppression
- `IGNORE_WARNINGS_BEGIN` / `IGNORE_WARNINGS_END` — suppress all warnings (for external code wrappers)

The verilator.vlt file is only for external/third-party code (cvfpu, hardfloat).

**Why:** Project convention is to resolve warnings at the source, keeping the lint config minimal and only for code we don't own.

**How to apply:** When Verilator reports a warning, fix the root cause in the RTL or use the appropriate VX_platform.vh macro. Never add rules to verilator.vlt.in.
