# Kernel Entry & Multi-Entry `.vxbin` — Design

**Scope:** how a Vortex kernel binary is entered — the unified CTA
prologue, the `.weak kernel_main` model, and the multi-entry `.vxbin`
symbol-table mechanism that lets one binary hold several named kernels
(the device-side foundation for PoCL / multi-kernel programs). Covers
[`sw/kernel/src/vx_start.S`](../../sw/kernel/src/vx_start.S),
[`sw/kernel/scripts/vxbin.py`](../../sw/kernel/scripts/vxbin.py),
the link scripts, and
[`sw/runtime/common/module.cpp`](../../sw/runtime/common/module.cpp).

---

## 1. Unified entry model

There is a single shared CTA prologue `__vx_cta_entry`
([`vx_start.S:69-136`](../../sw/kernel/src/vx_start.S#L69)) that sets up
SATP/gp/sp/tp+TLS, runs `__libc_init_array`, loads kargs into `a0` from
`MSCRATCH`, and dispatches with `jalr ra, s11` (`s11`, callee-saved,
carries the kernel/trampoline address). `_start` is itself a 2-instruction
stub ([`:30-66`](../../sw/kernel/src/vx_start.S#L30)):
`lla s11, kernel_main; j __vx_cta_entry`, with `kernel_main` declared
`.weak` so a multi-entry program needs no dead default stub (the
PC-relative `lla` resolves a weak-undef to 0 cleanly). Per-kernel entry
stubs live in a `.vx_entry` section, `KEEP`'d against `--gc-sections` by
both link scripts ([`link32.ld:59-61`](../../sw/kernel/scripts/link32.ld#L59),
`link64.ld:59-61`). The dispatch `jalr` is emitted under `.option norvc`
so it stays 4 bytes for CTA rewind.

---

## 2. Multi-entry `.vxbin` format

[`vxbin.py`](../../sw/kernel/scripts/vxbin.py) scans the linked ELF for
`__vx_kentry_*` symbols and, if any exist, appends a `VXSYMTAB` footer
([`:69-104`](../../sw/kernel/scripts/vxbin.py#L69)): a string blob + packed
`(name → PC)` entries + count + magic. A binary with no such symbols is
**byte-identical** to the legacy single-entry format.

The on-disk `.vxbin` layout is `[min_vma][max_vma][image][VXSYMTAB?]`. The
runtime loader [`module.cpp:75-119`](../../sw/runtime/common/module.cpp#L75)
sniffs the trailing magic: with a footer it parses the entries and
`vx_module_get_kernel(name)` resolves a name to a PC; without a footer it
synthesizes a single `"main"` → `min_vma` entry. This keeps legacy
single-kernel binaries working unchanged.

---

## 3. Flow

1. The compiler emits one `__vx_kentry_<name>` stub per kernel into
   `.vx_entry`; `vxbin.py` records them in the `VXSYMTAB` footer.
2. At load, `vx_module_load_file/bytes` parses the footer into a
   name→PC table.
3. `vx_module_get_kernel("<name>")` returns a `vx_kernel_h`; launch
   programs that PC into the KMU.
4. Every kernel shares `__vx_cta_entry`; `s11` carries its entry address,
   kargs arrive in `a0`.

[`tests/regression/multikernel/`](../../tests/regression/multikernel/)
(`kentry.S` with `add_k`/`mul_k`/`acc_k`) is the POCL-free in-tree test of
the multi-entry path.

---

## 4. Proposed but not yet implemented

1. **Single-entry regression re-validation** (proposal §8): the ~50
   single-entry `tests/regression` kernels should be re-run to confirm the
   legacy path's move from a direct `jal kernel_main` to `jalr ra, s11`
   (PoCL's runs already prove `jalr`/CTA-rewind compatibility; this is a
   test-coverage TODO, not a code gap).
2. **PoCL trampoline interop** (`__vx_tramp_<name>`) lives in the external
   mesa/PoCL tree; `__vx_cta_entry` is agnostic to it by design.

---

## 5. Source proposal

This design consolidates and supersedes `kmu_kernel_entry_proposal.md`
(now removed from `docs/proposals/`). The runtime module/kernel handle API
is in [`vortex_runtime_api.md`](vortex_runtime_api.md); the KMU CTA
dispatch is in [`cta_clustering_and_dispatch.md`](cta_clustering_and_dispatch.md).
