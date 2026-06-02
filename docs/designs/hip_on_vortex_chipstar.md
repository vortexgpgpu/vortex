# HIP on Vortex (via chipStar) — Design

**Scope:** how a HIP program compiles and runs on Vortex today. The
working path is **chipStar → SPIR-V → POCL → Vortex**, with both 64-bit
(rv64) and 32-bit (rv32) supported. This document covers the in-tree glue
(CI install scripts + the HIP test suite) and the external toolchain it
orchestrates.

> **Note on a separate, unbuilt direction.** A bespoke HIP toolchain
> (a `HIPVortex` Clang driver, a native `libhip_vortex` runtime on
> `vortex2.h`, and an out-of-tree `vortex_mlir` dialect) was proposed but
> is **not implemented**; its proposal `hip_support_proposal.md` is
> **retained** in `docs/proposals/`. Its key motivation that the chipStar
> path cannot satisfy — exposing Vortex-specific intrinsics (WMMA/WGMMA/
> TMA) through HIP headers — is preserved in §5.

---

## 1. The compilation and execution path

```
  main.cpp (HIP, __global__ kernels, hipMalloc/hipMemcpy/<<<>>>)
    │
    ▼  chipStar hipcc  (--offload-pointer-width=$XLEN)        [external: $TOOLDIR/chipstar]
       clang++ (llvm_vortex) --offload=spirv{32,64}  →  device.spv  (Physical32/64)
       host ELF embeds the SPIR-V fatbin, links libCHIP.so
    │
    ▼  run: libCHIP.so (CHIP_BE=opencl) → POCL libOpenCL → POCL Vortex device   [external]
       POCL JITs SPIR-V → riscv$XLEN → .vxbin   (clang + vxbin.py)
    │
    ▼  libvortex.so executes on  simx / rtlsim / opae / xrt
```

The Vortex `sw/` runtime tree itself is untouched by HIP — everything
load-bearing (hipcc, `libCHIP.so`, device libs, the SPIR-V→Vortex
lowering, the runtime) is external, installed by CI into `$TOOLDIR`.

---

## 2. In-tree components

| Path | Role |
|---|---|
| [`ci/chipstar_install.sh.in`](../../ci/chipstar_install.sh.in) | Producer: clones `vortexgpgpu/chipStar @ vortex_3.x`, builds with `-DCHIP_TARGET_POINTER_WIDTHS="32;64"` against `$TOOLDIR/llvm-vortex` + `$TOOLDIR/pocl`, installs hipcc, `libCHIP.so`, and `hipspv-spirv{32,64}.bc`. |
| [`ci/toolchain_install.sh.in`](../../ci/toolchain_install.sh.in) | `chipstar()` + `pocl()` fetch prebuilt tarballs; both in the default `--all` set. |
| [`ci/toolchain_prebuilt.sh.in`](../../ci/toolchain_prebuilt.sh.in) | `chipstar()` packages `$TOOLDIR/chipstar` into a tarball. |
| [`tests/hip/common.mk`](../../tests/hip/common.mk) | The real build/run engine: chipStar hipcc → SPIR-V, POCL JITs to Vortex, runs on simx/rtlsim/opae/xrt. Passes `--offload-pointer-width=$(XLEN)` and `POCL_VORTEX_XLEN=$(XLEN)`. |
| [`tests/hip/{vecadd,sgemm}/`](../../tests/hip/) | Two real HIP tests (`__global__` kernels, `hipMalloc`/`hipMemcpy`/`<<<>>>`/`hipDeviceSynchronize`). |
| [`ci/regression.sh.in`](../../ci/regression.sh.in) | `hip()` runs `make -C tests/hip run-{simx,rtlsim,opae,xrt}`; `--hip` selector; in `--all`. |

There is **no in-tree HIP runtime shim** — the references to chipStar/hipcc
in `sw/runtime/{device.cpp,vortex2.h,vortex-kernel.pc.in}` are comments
naming downstream consumers, not code.

---

## 3. 32-bit (rv32) support

The headline capability: rv32 HIP works end-to-end. `common.mk` passes
`--offload-pointer-width=$(XLEN)` and sets `POCL_VORTEX_XLEN=$(XLEN)`; the
chipStar install builds both `hipspv-spirv32.bc` and `hipspv-spirv64.bc`.
rv32 emits `Physical32` SPIR-V, which POCL's rv32 Vortex device
(`address_bits=32`) accepts. This required, in the external repos:

- **llvm_vortex** (`vortex_3.x`): Clang HIPSPV accepts `spirv32`
  (`Driver.cpp`, `HIPSPV.cpp`).
- **chipStar** (`vortex_3.x`): a multi-width device library
  (`CHIP_TARGET_POINTER_WIDTHS` CMake cache var) + hipcc/ROCm-Device-Libs
  patches (carried as `HIPCC-patches/` + `ROCm-Device-Libs-patches/`).

rv32 `vecadd` and `sgemm` PASS on SimX; the broader chipStar conformance
smoke is "mixed" (~36% passing, catalogued in the fork's
`known-failures-vortex32.txt`).

---

## 4. Architecture notes

- **chipStar is the OpenCL backend** (`CHIP_BE=opencl`): HIP host calls map
  to OpenCL, and device code is SPIR-V JIT-compiled by POCL to a Vortex
  `.vxbin`. POCL is shared with the OpenCL test path (see the retained
  PoCL proposals).
- **External vs in-tree split** is deliberate: the Vortex repo owns only
  the test sources, the build/run `common.mk`, and the CI install/
  regression glue. The toolchain is versioned in `vortexgpgpu/chipStar`,
  `llvm_vortex`, and `vortexgpgpu/pocl`.

---

## 5. Proposed but not yet implemented

1. **Hardware-extension exposure via HIP** (`hip_support_proposal` Phase 4
   — the strongest reason that proposal is retained): `nvcuda::wmma`-style
   HIP headers exposing Vortex WMMA/WGMMA/TMA/async-barrier intrinsics.
   The chipStar/SPIR-V path structurally cannot reach Vortex-specific
   intrinsics.
2. **MLIR research middleware** (`hip_support_proposal`): an out-of-tree
   `vortex_mlir` dialect, `vortex-opt`, and GPUToVortex/VortexToLLVM
   lowerings — zero code exists.
3. **Native `libhip_vortex` runtime** (`hip_support_proposal` Phase 1):
   direct `hipMalloc → vx_mem_alloc` on the Vortex runtime, removing the
   POCL JIT layer — only a stub exists externally.
4. **chipStar conformance long-tail** (rv32): subgroups, FP64 atomics,
   image support, and `sizeof(size_t)==8` device assumptions — catalogued,
   not fixed. POD-arg width drift (host `size_t`=8 vs device=4 on rv32) is
   an accepted risk with no host-narrowing fix.

**Known discrepancies to fix** (not future work): stale "rv64-only"
comments and a stale `hip` exclusion in `ci_xlen32.sh` /
`tests/hip/common.mk` headers / `ci_xlen64.sh` — the rv32 gap was closed
(all phases of `chipstar_opencl_32bit_proposal` done) but these exclusion
and comment sites were never updated, so `ci_xlen32.sh` still excludes
`hip` from its test list despite the toolchain now supporting rv32. Also:
the proposal's `CHIP_ENABLE_TARGET_POINTER_WIDTHS` was renamed to
`CHIP_TARGET_POINTER_WIDTHS` during execution.

---

## 6. Source proposals

This design consolidates and supersedes `chipstar_on_vortex_proposal.md`
(rv64 validation — done) and `chipstar_opencl_32bit_proposal.md` (rv32
enablement — done), now removed from `docs/proposals/`.
`hip_support_proposal.md` (the bespoke toolchain / `libhip_vortex` / MLIR
direction) is **retained** in `docs/proposals/` as the unimplemented
forward roadmap (§5 items 1–3). The POCL layer this path depends on is
described by the retained PoCL proposals.
