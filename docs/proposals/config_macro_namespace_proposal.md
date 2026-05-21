**Date:** 2026-05-20
**Status:** Draft — not yet approved
**Author:** Blaise Tine
**Related:** [command_processor_proposal.md](command_processor_proposal.md)

# Vortex Macro Namespace & Header Layering — Proposal

## 1. Summary

Vortex's build-time symbols — hardware config, ISA constants, register
maps, debug knobs, tool predicates — are today an undisciplined mess.
Three problems are tangled together:

1. **Flat global namespace.** Every key in [VX_config.toml](../../VX_config.toml)
   (~245 of them) is emitted as a bare `#define` / `` `define `` —
   `NUM_THREADS`, `XLEN`, `DEBUG_LEVEL`, `ISA_EXT_DXA`, … — into the
   global C and Verilog macro namespace.

2. **Header-layering leak.** The *public* runtime header
   [sw/runtime/include/vortex2.h](../../sw/runtime/include/vortex2.h)
   currently does `#include <VX_config.h>` — so any host application
   that includes the Vortex runtime inherits all ~245 build-internal
   config macros into its translation units.

3. **Conflated kinds.** `VX_config.toml` is a dumping ground for at
   least six *different kinds* of symbol — hardware config, debug
   knobs, ISA-spec constants, external tool predicates, internal
   expression locals, language builtins — emitted into one flat space
   with nothing to tell them apart.

This proposal fixes all three with one coherent design:

- **A namespace scheme.** Every Vortex-emitted macro becomes
  `VX_<ROLE>_*`. The sub-prefix names the symbol's *role* and its
  *source of truth*, so a reader always knows what a symbol is and
  where it came from. The three subspaces are disjoint by construction.

- **A header layering.** The public runtime headers (`vortex2.h`,
  `vortex.h`) depend on **standard C headers only** — nothing from the
  Vortex build. Build-internal configuration never reaches the public
  include graph.

- **Mechanically.** In-TOML key renames + a self-contained `vortex2.h`
  + a mechanical `sed` codemod. The namespace rename needs **no
  generator change** (§3.6); the §9 follow-up then adds one new,
  additive emission format (`sv_pkg`) on top.

The approach mirrors [VX_types.toml](../../VX_types.toml), which already
spells its keys with prefixes directly (`VX_CSR_*`, `VX_DCR_*`): the
generator has no naming logic; the TOML author makes the namespace
decision by how the key is spelled.

A separate, deferred follow-up (§9) then gives the RTL a typed
`localparam` *view* of the value configs — orthogonal to the rename and
committed on its own.

---

## 2. Goals and non-goals

### 2.1 Goals

- Eliminate collisions between Vortex build macros and (a) the public
  runtime API, (b) host/OS/POSIX headers, (c) EDA tool macros.
- Make every emitted symbol **self-identifying**: the prefix names its
  role and the file it comes from.
- Make the public runtime headers **self-contained** — zero dependency
  on the Vortex build (`VX_config.h`, `VX_types.h`).
- Keep the researcher configurability story unchanged: flip one TOML
  knob, or pass one `-D`, to retarget the design.

### 2.2 Non-goals

- **No generator change for the rename.** The namespace migration
  (Phases 0–3) does not modify [ci/gen_config.py](../../ci/gen_config.py)
  — it already emits whatever key names it reads (§3.6). The §9
  follow-up *does* extend the generator, but only additively: one new
  output format (`sv_pkg`) emitting the typed `localparam` package; the
  existing `verilog`/`cpp`/`cflags` paths are untouched.
- **No mechanism change** *in this migration.* `#ifdef` / `` `ifdef ``
  stays; no `constexpr`, no `if constexpr`. Conditional-compilation
  flexibility (structural gating, conditional `#include`s, conditional
  port lists, asm/Verilog preprocessing reach) is worth keeping. A
  typed SystemVerilog `localparam` *view* of the value configs is a
  deferred, separately-committed follow-up — see §9.
- **No type-safety upgrade.** Macros remain untyped.
- **No new public API.** This proposal *removes* leakage; it does not
  add a single new public symbol.

---

## 3. Problem analysis

### 3.1 Current emission

`gen_config.py` walks the TOML and emits one bare `#define` per key:

```c
#define NUM_THREADS       4
#define XLEN              32
#define ICACHE_ENABLE
#define DEBUG_LEVEL       3
#define ISA_EXT_DXA       10
```

No prefix. Every TOML section feeds the same flat global namespace.

### 3.2 The six kinds of symbol

The root cause is that `VX_config.toml` (and the ISA constants inside
it) conflate fundamentally different things:

| Kind | What it is | Examples | Today |
|---|---|---|---|
| **Hardware config** | tunable design parameters *owned by* `VX_config.toml` | `NUM_THREADS`, `XLEN`, `DCACHE_SIZE`, `EXT_F_ENABLE` | bare |
| **Debug / verification knobs** | observe/exercise the design, not the design itself | `DEBUG_LEVEL`, `STALL_TIMEOUT`, `RVTEST_MT` | bare |
| **ISA / identity constants** | RISC-V / MISA bit positions and device-identity values — spec facts | `ISA_STD_F`, `ISA_EXT_DXA`, `VENDOR_ID` | bare |
| **External selectors** | set from *outside* Vortex — the TOML only reads them | `VIVADO`, `QUARTUS`, `SYNTHESIS`, `ASIC`, `SV_DPI` | bare |
| **Internal `expr:` locals** | helpers used only during TOML evaluation | `fpu_dsp_vivado`, `__cache_repl_fifo` | lowercase, not emitted |
| **Language builtins** | `[[builtin]]` declarations | `__FILE__`, `__LINE__` | not emitted |

Five of these six are emitted into one undifferentiated namespace.

### 3.3 Collision surfaces

- **`vortex2.h` public API.** Already owns `VX_*` for enums
  (`VX_SUCCESS`, `VX_QUEUE_PRIORITY_*`, `VX_CAPS_*`, `VX_ISA_*`, …).
  The config namespace grows independently — collision is a matter of
  time and luck.
- **Host runtime / OS headers.** `NUM_THREADS`, `NUM_BARRIERS`,
  `DEBUG_LEVEL` are short, generic names — collision with OpenMP,
  pthreads-adjacent, spdlog, or application code is inevitable. The
  simx build alone links ramulator, softfloat, spdlog, yaml-cpp.
- **EDA tool macros.** Integrators pass `-DVIVADO`, `-DSYNTHESIS`,
  etc. The TOML deliberately *consumes* these (§3.5) — they are not
  Vortex config.

### 3.4 The header-layering leak

[vortex2.h:32](../../sw/runtime/include/vortex2.h#L32) currently does:

```c
#include <VX_config.h>
```

It was added to let `vortex2.h`'s `VX_ISA_*` macros resolve — they are
written as references to bare config macros:

```c
#define VX_ISA_STD_A      (1ull << ISA_STD_A)            // ISA_STD_A from VX_config.toml
#define VX_ISA_EXT_TCU    (1ull << (32+ISA_EXT_TCU))     // ISA_EXT_TCU from VX_config.toml
```

This is wrong on every axis:

- **The runtime API is hardware-agnostic by definition.** One
  `libvortex.so` / one `vortex2.h` must work against *any* device
  config; the host queries the device at runtime
  (`vx_device_query`). `VX_config.h` is the compile-time config of
  *one* build. Baking it into the API collapses the abstraction.
- **Host apps do not have `VX_config.h`.** A runtime consumer ships
  with `vortex2.h` + `libvortex.so`. `VX_config.h` is an artifact of
  building the Vortex *device*. `#include <VX_config.h>` in a public
  header means a standalone consumer does not compile.
- **It is the pollution of §3.3, guaranteed.** Every host TU that
  includes `vortex2.h` inherits all ~245 config macros. The
  `VX_CFG_` rename alone does **not** fix this — `VX_CFG_NUM_THREADS`
  still must not land in a host app's TU. *The `#include` itself is
  the defect.*

`grep -rn "VX_config" sw/runtime/include/` returns a hit today. Any
design that claims otherwise (an earlier draft of this proposal did)
is aspirational, not real.

### 3.5 Why a section-based rule fails

The naive rule "prefix every key, by TOML section" is wrong because
several sections are **internally mixed**:

- `[fpu]`, `[vm]` — contain lowercase internal `expr:` locals
  (`fpu_dsp_vivado`, `__cache_repl_*`) alongside real config keys.
- `[isa_signatures]` — contains ISA-spec *constants* (`ISA_STD_*`)
  **and** config-*derived* values (`MISA_STD`/`MISA_EXT`).
- `[debug]` — `STALL_TIMEOUT` is a *derived* value; `DEBUG_LEVEL` is
  a plain knob.

The categorization must key off **what each symbol is**, not which
section it sits in (§4.3).

### 3.6 Generator constraint: strictly per-file

[gen_config.py](../../ci/gen_config.py) takes a single `--config`
TOML ([gen_config.py:1370](../../ci/gen_config.py#L1370)) and builds
its resolver symbol table from *that one file*
([gen_config.py:1413](../../ci/gen_config.py#L1413)).
[configure](../../configure#L210-L219) loops `*.toml` and invokes the
generator **once per file** — `VX_config.toml` and `VX_types.toml` are
separate invocations, separate symbol tables.

**Consequence:** an `"expr:"` in one TOML cannot reference a `$symbol`
defined in the other. Any value whose `expr:` spans both files must be
made self-contained within one file. This is a hard constraint the
design respects — without changing the generator (§2.2).

---

## 4. Proposed design

### 4.1 The namespace scheme

Every Vortex-emitted macro is `VX_*`. A sub-prefix names the role and
the source of truth. The subspaces are provably disjoint:

| Prefix | Role | Source of truth | Emitted into | Visibility |
|---|---|---|---|---|
| `VX_*` (no sub-prefix) | public runtime API — handles, calls, API enums | [vortex2.h](../../sw/runtime/include/vortex2.h) (hand-written) | — | **public** |
| `VX_ISA_*` (flags) | ISA capability flags & bit layout | `vortex2.h` (hand-written, self-contained) | — | **public** |
| `VX_ISA_*` (identity) | device-identity constants (mvendorid/marchid/mimpid) | [VX_types.toml](../../VX_types.toml) | `VX_types.h` / `VX_types.vh` | **public** (peer header) |
| `VX_CSR_*`, `VX_DCR_*` | device CSR / DCR register map | [VX_types.toml](../../VX_types.toml) | `VX_types.h` | **public** (peer header) |
| `VX_CFG_*` | hardware design configuration | [VX_config.toml](../../VX_config.toml) | `VX_config.h` | **build-internal** |
| `VX_DBG_*` | debug / verification knobs | `VX_config.toml` | `VX_config.h` | **build-internal** |
| *(bare)* | external tool selectors, set from outside Vortex | `[toolchain]` keys | `VX_config.h` | input predicate |
| *(lowercase)* | internal `expr:` locals | `VX_config.toml` | *not emitted* | private |

`[[builtin]]` declarations (`__FILE__`, `__LINE__`) are never emitted.

Why sub-prefixes and not a bare `VX_`: `VX_` alone is the public API's
space; lumping HW build config under it re-creates the collision one
level up. `VX_CFG_` / `VX_DBG_` / `VX_CSR_` / `VX_DCR_` / `VX_ISA_`
split it cleanly.

### 4.2 Header layering

```
  PUBLIC  — shippable, namespace-clean, build-independent
  ┌────────────────────────────────────────────────────────────┐
  │  vortex2.h   runtime API + VX_ISA_* flags                   │
  │              includes: <stdint.h> <stddef.h> <stdio.h> only │
  │  vortex.h    legacy API — includes: <vortex2.h> only        │
  │  VX_types.h  device register map (VX_CSR_*/VX_DCR_*)        │
  │              — peer header, standalone, pure constants      │
  └────────────────────────────────────────────────────────────┘
  BUILD-INTERNAL  — never in the public include graph
  ┌────────────────────────────────────────────────────────────┐
  │  VX_config.h   VX_CFG_* + VX_DBG_* + bare [toolchain]       │
  │                consumed by sim / RTL / kernel builds only   │
  └────────────────────────────────────────────────────────────┘
```

**The invariant:** an `#include` may never point from PUBLIC into
BUILD-INTERNAL.

`VX_types.h` is a *peer* public header, not nested inside `vortex2.h`:
it is namespace-clean and config-invariant, so it is a legitimate
public header — but most runtime users (create queue, launch by kernel
handle, read buffer) never touch a raw CSR/DCR. A low-level consumer
that programs DCRs directly includes `VX_types.h` **explicitly**.
`vortex2.h` stays minimal and does not pull it in.

Anything that genuinely varies by hardware config is **never** a
compile-time macro in a public header — it is a **runtime query**
(`vx_device_query(VX_CAPS_*)`). `VX_ISA_EXT_TCU` is a fixed bit
*position*; whether a device *has* TCU is the runtime result tested
against it.

### 4.3 Categorization rule (per-key)

For each key in `VX_config.toml`, in order:

1. **lowercase** (fails `gen_config.py`'s `_has_public_scope`) →
   internal `expr:` local. Not emitted. **Untouched.**
2. in **`[toolchain]`** → external predicate. **Stays bare.**
3. in **`[debug]` / `[testing]`** → **`VX_DBG_*`**.
4. otherwise (uppercase HW-config key, incl. `XLEN`, `[[param]]` and
   `[[enum]]` bases + their companion predicates) → **`VX_CFG_*`**.

`[isa_signatures]` is not covered by this rule — it is dissolved
(§4.4). The device-identity constants are relocated to `VX_types.toml`
(§4.6). `[[builtin]]` is not emitted.

This is mechanical: the cases are decidable from the key's case and
section alone — no per-key annotation needed. `XLEN` is an ordinary
HW-config parameter and takes the general step-4 path → `VX_CFG_XLEN`
(§4.5).

### 4.4 Dissolving `[isa_signatures]`

`[isa_signatures]` holds two different kinds of symbol; they go to two
different homes.

**ISA bit-position constants** — `ISA_STD_*`, `ISA_EXT_*`. These are
RISC-V / MISA-CSR layout facts (not tunable knobs) and they are
consumed by the **public API**: `vortex2.h` builds its public
`VX_ISA_STD_*` / `VX_ISA_EXT_*` flag masks from them, and ~6 tests
reference them. They are therefore public-API material:

- **Moved into `vortex2.h`**, self-contained. `VX_ISA_*` macros are
  defined with the bit position **inlined as a literal**:
  ```c
  #define VX_ISA_EXT_TCU   (1ull << (32 + 9))    // was (1ull << (32+ISA_EXT_TCU))
  ```
- **Deleted from `VX_config.toml`.** (Not relocated to
  `VX_types.toml`: the consumer is the public API, so the public
  header should own them outright.)
- The ~6 tests using bare `ISA_EXT_DXA` switch to the public
  `VX_ISA_EXT_DXA`.

**Config-derived MISA values** — `MISA_STD`, `MISA_EXT`. These are the
computed `misa` CSR contents — a function of which extensions are
enabled (`$EXT_*_ENABLE`). They are config-derived, so by §3.6 they
**must stay in `VX_config.toml`** (their inputs live there):

- Renamed `VX_CFG_MISA_STD` / `VX_CFG_MISA_EXT` — they are config
  outputs, and the prefix correctly says "from `VX_config.toml`",
  keeping the prefix↔file invariant intact.
- Their `"expr:"` is rewritten to use **literal** bit positions
  instead of `$ISA_STD_*` (which have left the file). `MISA_STD`
  already mixes literals in today (`<< 12`, `<< 1`, `<< 6`) — this
  just finishes that.

Net: `[isa_signatures]` ceases to exist. The frozen RISC-V bit numbers
appear in two places (`vortex2.h` and the `MISA_*` expr) — acceptable
duplication of a spec constant, in exchange for a genuinely
self-contained public header.

### 4.5 `XLEN` — read-only HW-config parameter

`XLEN` (32 vs 64) **is** a hardware design parameter — the datapath
width — so it is `VX_CFG_*` like any other HW config: `VX_CFG_XLEN`,
with companion predicates `VX_CFG_XLEN_32` / `VX_CFG_XLEN_64`. It takes
the general §4.3 step-4 path; there is **no special case**.

What is unusual about `XLEN` is only *where its value comes from*: it
is selected from outside via `configure --xlen`, not assigned in the
TOML. That affects the mechanism, not the namespace:

- The `[isa] XLEN = 32` assignment is **deleted** — `XLEN` is declared
  only by `[[enum]] XLEN = [32, 64]`, making it read-only in the TOML.
  The TOML therefore does not emit the `XLEN` macro; the build supplies
  `VX_CFG_XLEN` directly.
- Only the **build-system layer** stays bare: the make variable
  `$(XLEN)`, the `configure --xlen` flag, the `@XLEN@` substitution.
  That layer never enters a C/Verilog TU as a macro, so it needs no
  prefix (same rationale as `TOOLDIR`, `DEBUG`, `CONFIGS`).
- The bridge: every place the build passed `-DXLEN_$(XLEN)` /
  `-DXLEN=$(XLEN)` now passes `-DVX_CFG_XLEN_$(XLEN)` and
  `-DVX_CFG_XLEN=$(XLEN)`.
- TOML `"expr:"` cross-references update with the rest: `$XLEN` →
  `$VX_CFG_XLEN`, `$XLEN_64` → `$VX_CFG_XLEN_64`.

### 4.6 Device-identity constants → `VX_types.toml`

`VENDOR_ID`, `ARCHITECTURE_ID`, `IMPLEMENTATION_ID` are the read-only
contents of the `mvendorid` / `marchid` / `mimpid` CSRs. They are
**fixed device-identity constants**, not tunable design parameters —
changing them does not retarget the design — so they do not belong in
`VX_config.toml` and must not become `VX_CFG_*`.

They are **moved to [VX_types.toml](../../VX_types.toml)** and spelled
`VX_ISA_VENDOR_ID` / `VX_ISA_ARCH_ID` / `VX_ISA_IMPL_ID`:

- `VX_types.toml` already owns the public, config-invariant constant
  surface (`VX_CSR_*`, `VX_DCR_*`) and emits to both `VX_types.h` (C)
  and `VX_types.vh` (Verilog) — exactly the reach these need: the RTL
  CSR file and any host/sim identity check both resolve them.
- The `VX_ISA_*` prefix marks them as ISA-spec / identity facts, peers
  of the hand-written `VX_ISA_*` capability flags in `vortex2.h`.
- They are **deleted from `VX_config.toml`** — it then carries no
  device-identity key at all.

---

## 5. Migration plan

Staged commits — substantial and testable, no skeletons, no WIP.

### Phase 0 — Decouple the public headers (prerequisite, one commit)

1. In `vortex2.h`, redefine `VX_ISA_STD_*` / `VX_ISA_EXT_*` with
   inlined literal bit positions — no reference to any external macro.
2. Remove `#include <VX_config.h>` from `vortex2.h`.
3. Switch the ~6 tests using bare `ISA_EXT_DXA` to `VX_ISA_EXT_DXA`.
4. Verify `grep -rn "VX_config" sw/runtime/include/` is **empty**.

This must land first: the rest of the proposal assumes the public
headers are already decoupled.

### Phase 1 — TOML rename (one commit)

1. In `VX_config.toml`, apply the §4.3 rule: HW-config keys (incl.
   `XLEN` → `VX_CFG_XLEN`) → `VX_CFG_*`; `[debug]`/`[testing]` →
   `VX_DBG_*`; `[toolchain]` stays bare; lowercase locals untouched.
2. Dissolve `[isa_signatures]` per §4.4: delete `ISA_STD_*`/
   `ISA_EXT_*`; keep `MISA_STD`/`MISA_EXT`, inline their bit
   positions, rename `VX_CFG_MISA_*`.
3. Relocate the device-identity constants per §4.6: delete
   `VENDOR_ID`/`ARCHITECTURE_ID`/`IMPLEMENTATION_ID` from
   `VX_config.toml`; add `VX_ISA_VENDOR_ID`/`VX_ISA_ARCH_ID`/
   `VX_ISA_IMPL_ID` to `VX_types.toml`.
4. Update every `"expr:"` cross-reference (including `$XLEN` →
   `$VX_CFG_XLEN`, `$XLEN_64` → `$VX_CFG_XLEN_64`, and enum-companion
   predicates `$VX_CFG_FPU_TYPE_*`, …).
5. Regenerate `VX_config.h`/`VX_config.vh` (and `VX_types.h`/`.vh`);
   diff against a saved pre-change baseline — any unresolved `$NAME`
   surfaces here.

No `gen_config.py` change.

### Phase 2 — Codemod across the source tree (one commit per subsystem)

Generate the rename list from the TOML so it stays exhaustive; apply
one `sed` per subsystem; verify each builds before the next.

```bash
python3 ci/list_config_keys.py --emit-rename-map > /tmp/rename.sed   # new ~30-line helper
# rename.sed: s/\bKEY\b/VX_CFG_KEY/g  (and VX_DBG_ for the 3 debug keys)
find hw -name '*.sv' -o -name '*.vh' -o -name '*.svh' -o -name '*.v' | xargs sed -i -E -f /tmp/rename.sed
```

Subsystem order (each its own commit, for clean bisect):

1. `hw/` — `*.sv`, `*.vh`, `*.svh`, `*.v`
2. `sim/simx/`, `sim/rtlsim/` — `*.cpp`, `*.h`, `*.hpp`
3. `sw/runtime/`, `sw/kernel/` — `*.cpp`, `*.c`, `*.h`, `*.hpp`
4. `tests/` + `ci/` — kernel/host sources, `Makefile`, `*.sh`,
   `*.sh.in`, `README.md`

Word-boundary anchors (`\b`) prevent partial-token corruption and
leave non-config identifiers (test-local `ITYPE`, `PROFILE_ENABLE`,
CLI flags like `--threads`) untouched — they are simply absent from
the rename list.

### Phase 3 — CI guard + docs (one commit)

1. New `ci/check_public_headers.sh`: fail if any header under
   `sw/runtime/include/` reaches `VX_config.h` in its include graph.
   (Now genuinely passable, thanks to Phase 0.)
2. One-line comment in `vortex2.h` documenting the layering rule.
3. Update [README](../../README.md) and developer docs that mention
   bare `NUM_THREADS`/`XLEN`-style symbols.

---

## 6. Risk and rollback

- **Risk:** a stale bare-macro reference slips the codemod and
  silently expands to nothing. **Mitigation:** `-Wundef` for C/C++;
  RTL elaboration catches undefined backtick-defines; diff the
  regenerated headers against baseline.
- **Risk:** an incomplete `"expr:"` rewrite breaks codegen.
  **Mitigation:** regenerate immediately after the TOML edit and diff
  — unresolved `$NAME` surfaces at once.
- **Risk:** downstream forks carry patches referencing bare
  `NUM_THREADS`/`XLEN`. **Mitigation:** document in release notes;
  the rename map is exhaustive and reusable by forks.
- **Rollback:** Phase 0 reverts independently. Phases 1–3 revert
  cleanly on top — the codemod is mechanical, the CI guard additive.

---

## 7. Cost

- Generator change: **none**.
- Phase 0: hand-edit of `vortex2.h` (~30 `VX_ISA_*` lines) + 6 tests.
- Phase 1: mechanical rename of ~245 keys + their `"expr:"` references,
  one file.
- Phase 2: one ~30-line helper + mechanical `sed` over four subsystems.
- Test matrix: existing CI is sufficient — the change is name-only,
  semantics byte-identical (verify with a regenerated-header diff).

Estimated wall-clock: ~1 day Phase 0+1, ~1 day Phase 2, ~2 h Phase 3.

---

## 8. Alternatives considered

- **Namespaced `constexpr` + SV `package`.** Cleaner type story, but
  loses conditional-compilation flexibility. Rejected per project
  preference (§2.2).
- **Bare `VX_` prefix (no sub-prefix).** Conflates the public API
  namespace with HW build config. Rejected (§4.1).
- **Per-section `_prefix` meta-key in the generator.** Adds generator
  behavior to maintain; the TOML stops reading as the literal source
  of symbol names. Rejected.
- **Keep `vortex2.h #include <VX_config.h>`, just rename the macros.**
  Does not fix §3.4 at all — `VX_CFG_NUM_THREADS` still must not enter
  a host TU. Rejected.
- **Move `[isa_signatures]` wholesale to `VX_types.toml`.** Blocked by
  §3.6 (the per-file generator) for `MISA_*`, and wrong for the
  constants — their consumer is the public API, so `vortex2.h` should
  own them (§4.4).

---

## 9. Follow-up: RTL config localparams (separate commit)

After the rename, Vortex RTL references hardware config purely as
preprocessor macros — `` `VX_CFG_XLEN ``, `` `VX_CFG_NUM_THREADS ``,
`` `VX_CFG_FLEN ``, … — scattered through module bodies. Backtick
macros are untyped, unscoped, and invisible to elaboration-time
checking. SystemVerilog has a better tool for *value* parameters:
typed `localparam`s in a `package`.

**Proposal.** For each RTL package `VX_*_pkg.sv`, resolve the
value-carrying `` `VX_CFG_* `` macros once into typed `localparam`s:

```systemverilog
package VX_gpu_pkg;
  localparam int XLEN        = `VX_CFG_XLEN;
  localparam int FLEN        = `VX_CFG_FLEN;
  localparam int NUM_THREADS = `VX_CFG_NUM_THREADS;
  // ...
endpackage
```

Modules then `import VX_gpu_pkg::*;` and use the typed, scoped
`localparam` instead of the bare macro.

**Boundary — what converts and what does not:**

- **Value configs only** (widths, counts, latencies) — the symbols
  that read naturally as `localparam int`. These convert.
- **Boolean enable-flags consumed by `` `ifdef ``** — `VX_CFG_*_ENABLE`
  used for structural gating (conditional `generate`, port lists,
  `` `include ``) — **stay macros**: `` `ifdef `` cannot test a
  `localparam`, and that conditional-compilation reach is exactly what
  §2.2 keeps.
- Scope is Vortex's **own design RTL** (non-`rtl/libs` modules); the
  vendored library primitives under `rtl/libs` are out of scope.
- No value is duplicated: the `localparam` is *initialized from* the
  macro. `VX_config.vh` stays the single source of truth; the package
  is a typed *view* of it.

**Why a separate commit, after the rename.** It is design-touching
(every package, plus every module's parameter references) and
orthogonal to the namespace work; it is independently testable via RTL
elaboration + `rtlsim`. The rename must land first so the packages
resolve `` `VX_CFG_* `` rather than bare names.

**Benefits.** Type and width checking at elaboration; proper scoping;
`localparam`s appear in waveforms and tooling; module bodies stop being
macro soup.
