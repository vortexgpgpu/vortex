**Date:** 2026-05-20
**Status:** Draft — describes the refactored `vx_start.S` (KMU branch)
**Author:** Blaise Tine
**Related:**
[command_processor_proposal.md](command_processor_proposal.md),
[pocl_on_vortex2_proposal.md](pocl_on_vortex2_proposal.md),
[pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md).

### Update history

- **2026-05-20** — Initial write-up. Documents the unified KMU kernel-entry
  model in `sw/kernel/src/vx_start.S` after two changes landed together:
  the **kernel-id-switch removal** (each kernel becomes its own `.vxbin`
  entry point, resolved by name) and the **`vx_start.S` refactor** that
  collapsed the duplicated prologue into a single `__vx_cta_entry` and
  made `_start` just another entry stub.

---

# Vortex KMU Kernel Entry — `vx_start.S` Design

## 1. Summary

`sw/kernel/src/vx_start.S` is the kernel-side startup shim. Under the v3
KMU (Kernel Management Unit) model it answers one question: **when the
KMU launches a warp at a kernel's entry PC, what runs before the kernel
body?**

The refactored design has a single answer. There is exactly one entry
routine — **`__vx_cta_entry`** — that brings the warp up to a C-callable
state and dispatches to a kernel whose address it finds in register
`s11`. Everything that the KMU can jump to is a **two-instruction stub**
of the same shape:

```asm
    lla  s11, <kernel>
    j    __vx_cta_entry
```

`_start` itself is one such stub (for the legacy `kernel_main`
convention); each named kernel of a multi-entry `.vxbin` has its own.
This replaces the previous design where `_start` and `__vx_cta_entry`
each carried a separate copy of the prologue and `_start` hard-wired a
`jal kernel_main`.

## 2. Background — the KMU launch model

The KMU iterates over a kernel's `(block, thread)` grid in hardware. For
every coordinate it launches a warp at a single **entry PC** — the value
the runtime programs into `VX_DCR_KMU_STARTUP_ADDR` — with the
kernel-arguments pointer placed in `VX_CSR_MSCRATCH`. The grid / block
dimensions and per-coordinate IDs are surfaced through CSRs
(`VX_CSR_CTA_*`); the kernel never loops over coordinates itself.

So the entry PC is reached **once per `(block, thread)`**. Whatever sits
there must, every time, re-establish a sane execution environment —
stack pointer, global pointer, thread pointer / TLS — before kernel C
code can run. That is the *prologue*, and it is the bulk of
`vx_start.S`.

## 3. The two `.vxbin` shapes

A `.vxbin` is `[min_vma:u64][max_vma:u64][image…]` optionally followed by
a `VXSYMTAB` footer (see [vx_module.cpp](../../sw/runtime/common/vx_module.cpp)).
Which shape a program has decides how the KMU finds kernels:

| Shape | Footer | Kernel resolution | Producer |
|---|---|---|---|
| **Footer-less** | none | runtime synthesizes one entry, `"main"` → `min_vma` | the single-kernel regression tests |
| **Multi-entry** | `VXSYMTAB`, N symbols | `vx_module_get_kernel(mod,"<name>")` → that symbol's PC | POCL's `compile_vortex_program`; the `multikernel` test |

`vxbin.py` decides at build time: it scans the ELF for `__vx_kentry_*`
symbols and, if any exist, appends a footer mapping each `<name>` to its
stub address. No such symbols → no footer → the footer-less shape, byte
-identical to the pre-existing format.

`vx_module_get_kernel` then returns a PC the runtime feeds straight into
`VX_DCR_KMU_STARTUP_ADDR`. **Every kernel the KMU can launch is therefore
some entry PC inside `vx_start.S`'s world** — either `_start` or a
per-kernel stub.

## 4. The entry model

```
KMU launches a warp at <entry PC>;  VX_CSR_MSCRATCH = kargs pointer
        │
        ├─ footer-less .vxbin   entry PC = min_vma = _start
        │     _start:              lla s11, kernel_main ; j __vx_cta_entry
        │
        └─ multi-entry .vxbin   entry PC = __vx_kentry_<name>  (VXSYMTAB footer)
              __vx_kentry_<name>:  lla s11, <kernel>    ; j __vx_cta_entry
        │
        ▼
   __vx_cta_entry:
        SATP  / gp / sp / tp+TLS / __libc_init_array      ← per-CTA prologue
        a0 = VX_CSR_MSCRATCH
        jalr ra, s11                                      ← call the kernel
        tmc  x0                                           ← retire the warp
```

### 4.1 `__vx_cta_entry` — the one shared routine

`__vx_cta_entry` is the only place the prologue exists. In order:

1. **SATP** (`VM_ENABLE` only) — point the per-core MMU at the page
   table the runtime pre-populated at `PAGE_TABLE_BASE_ADDR`. Safe before
   the stack is set up because the STACK and PT regions are MMU-bypass.
2. **`gp`** (`NEED_GP`) — `la gp, __global_pointer`.
3. **`sp`** — `STACK_BASE_ADDR` minus `hartid << STACK_LOG2_SIZE`, i.e. a
   private stack slice per hardware thread.
4. **`tp` + TLS** (`NEED_TLS`) — thread pointer past `_end`, offset by
   `hartid * __tbss_size`; then `call __init_tls`.
5. **global ctors** (`NEED_INITFINI`) — `call __libc_init_array`.
6. **dispatch** — `a0 = VX_CSR_MSCRATCH` (the kargs pointer), then
   `jalr ra, s11`.
7. **retire** — `tmc x0` shuts the warp down when the kernel returns.

`NEED_GP` / `NEED_TLS` / `NEED_INITFINI` are compile-time gates: the
prebuilt `libvortex2.a` enables all three; the regression build detects
them per program from the linked ELF's sections via
[kernel_startup.sh](../../sw/kernel/scripts/kernel_startup.sh).

### 4.2 The `s11` / `a0` dispatch contract

The entry stub leaves the kernel address in **`s11`**, then jumps to
`__vx_cta_entry`. `s11` is a *callee-saved* register, so it survives the
prologue's `call __init_tls` and `call __libc_init_array` without the
stub having to spill it. `__vx_cta_entry` then calls through it with the
kargs pointer in `a0` — the standard first-argument register — so a
kernel may simply be `void kern(kernel_arg_t* arg)`.

`jalr ra, s11` is wrapped in `.option norvc`: the scheduler's CTA-rewind
logic requires the dispatch instruction to be exactly 4 bytes, and a
compressed `c.jalr` would be 2.

### 4.3 `_start` and the per-kernel stubs

`_start` is the ELF `ENTRY` symbol and, because `.init` is linked first,
sits at `min_vma` — exactly the PC the runtime synthesizes for the
footer-less `"main"` entry. Its body *is* an entry stub:

```asm
_start:
    lla  s11, kernel_main
    j    __vx_cta_entry
```

So the legacy "one kernel per program, named `kernel_main`" convention is
just "the entry stub for `kernel_main`" — no longer a special code path.

A multi-entry program's per-kernel stubs are identical in shape. They are
placed in a dedicated **`.vx_entry`** section which the linker scripts
[`KEEP`](../../sw/kernel/scripts/link32.ld) — without that, `--gc-sections`
would drop them, since their only referent is the `VXSYMTAB` footer,
which `vxbin.py` appends *after* the linker has run.

### 4.4 Weak `kernel_main`

`kernel_main` is declared `.weak`. A footer-less program defines it (the
regression kernels literally name their kernel `kernel_main`); a
multi-entry program does not. For a multi-entry `.vxbin`, `_start` is
dead code — the KMU only ever enters the footer stubs — so its
`lla s11, kernel_main` harmlessly resolves `s11` to 0. `lla` is
PC-relative with ±2 GB reach, so a weak-undefined `kernel_main` links
cleanly; a `jal` (±1 MB) to an absolute 0 would not, which is why the
old hard-wired `jal kernel_main` forced every multi-entry program to
carry a dead `kernel_main` stub. The weak symbol removes that wart
everywhere — POCL and the `multikernel` test alike.

## 5. Supporting pieces

| Piece | Role |
|---|---|
| [`vx_start.S`](../../sw/kernel/src/vx_start.S) | `_start`, `__vx_cta_entry`, the prologue |
| [`link32.ld`](../../sw/kernel/scripts/link32.ld) / `link64.ld` | `KEEP(*(.vx_entry .vx_entry.*))` so the stubs survive `--gc-sections` |
| [`vxbin.py`](../../sw/kernel/scripts/vxbin.py) | emits the `VXSYMTAB` footer from `__vx_kentry_*` symbols |
| [`vx_module.cpp`](../../sw/runtime/common/vx_module.cpp) | parses the footer; `vx_module_get_kernel` resolves a name → PC |
| `compile_vortex_program` (POCL) | generates one `__vx_kentry_<name>` stub + a `vortex.kernel` trampoline per kernel |

POCL's per-kernel **trampoline** (`__vx_tramp_<name>`) is an extra hop
between the stub and the kernel: it runs POCL's once-per-launch argument
prologue (`__vx_kernel_setup`) and then calls the kernel. So in a POCL
`.vxbin`, `s11` points at the trampoline rather than the kernel itself —
`__vx_cta_entry` neither knows nor cares. A hand-written test
(`tests/regression/multikernel`) skips the trampoline and points `s11`
straight at a kernel that takes the kargs pointer directly.

## 6. The non-KMU path

`vx_start.S` still carries a second, `#else` branch for builds **without**
`KMU_ENABLE` — the legacy `wspawn` software-thread model: `_start` spawns
warps, runs `init_regs` on each, then `call main` / `tail exit`. It is a
separate execution model (used by `tests/kernel`), untouched by this
work, and out of scope here. `libvortex.a` is the non-KMU archive;
`libvortex2.a` is the KMU one.

## 7. What changed, and why

**Before — the kernel-id model.** `compile_vortex_program` emitted a
single `.vxbin` entry plus a `__vx_get_kernel_callback(kernel_id)` switch
table; the kargs blob carried a `kernel_id`, and one `kernel_main`
dispatched on it. `vx_start.S` had `_start` (its own prologue copy +
`jal kernel_main`) and `__vx_cta_entry` (a second prologue copy, via a
macro).

**After.**

- Each kernel is its own KMU entry point, resolved **by name** through
  the `VXSYMTAB` footer — no `kernel_id`, no switch table, no dispatch
  blob field.
- `vx_start.S` has **one** prologue (`__vx_cta_entry`); `_start` is a
  2-instruction stub identical in shape to the per-kernel stubs.
- `kernel_main` is weak, so no program needs a dead stub.

The result is a single, uniform entry model: every kernel the KMU can
launch — legacy or multi-entry, POCL or hand-written — is a stub that
sets `s11` and falls into the same shared prologue.

## 8. Verification

- `tests/regression/multikernel` — a POCL-free multi-entry `.vxbin` (two
  kernels in one program); isolates footer emission, name resolution,
  `.vx_entry` survival, and `__vx_cta_entry`. Passes on simx 32- and
  64-bit.
- `tests/opencl` (14 tests) and `tests/hip` (2) — exercise the multi
  -entry path through POCL on simx, 32- and 64-bit.
- The ~50 single-entry `tests/regression` kernels exercise `_start` →
  `__vx_cta_entry`. Re-running that suite after the refactor is the one
  outstanding verification: the legacy path's kernel call moved from a
  direct `jal kernel_main` to `jalr ra, s11` inside `__vx_cta_entry`
  (the 30 POCL runs already prove `jalr` is compatible with the
  CTA-rewind logic, but the regression kernels should confirm it).
