# OpenCL on Vortex (via PoCL) — Design

**Scope:** how an OpenCL program compiles and runs on Vortex today. The
working path is **OpenCL C / SPIR-V → PoCL → Vortex**, on both 32-bit
(rv32) and 64-bit (rv64) devices. This document covers the in-tree glue
(the OpenCL test suite, the device runtime the kernels link against, and
the CI catalog) and the external PoCL fork that hosts the Vortex device
driver and kernel builtin library.

PoCL is the **single compute front end shared by three stacks**: the
native OpenCL tests here, [HIP via chipStar](hip_on_vortex_chipstar.md)
(`CHIP_BE=opencl`), and any host that speaks the OpenCL ICD. Everything
this document describes about the device driver and the JIT flow is
therefore also the substrate under the HIP path.

> **Relationship to the graphics stacks.** The Vulkan front end
> ([vortexpipe](vortexpipe_architecture.md)) does **not** go through PoCL —
> it is a Mesa/Gallium driver that emits `.vxbin` directly. PoCL is the
> *compute* front end only. Both ultimately drive the same Vortex SDK
> (`libvortex.so`, header `vortex2.h`) and the same device runtime.

---

## 1. The compilation and execution path

```
 host OpenCL app
   │  -lOpenCL  (system ocl-icd loader)
   ▼
 libpocl.so  ── PoCL ICD; Vortex device statically linked in
   │            (ENABLE_LOADABLE_DRIVERS=OFF)
   │  vx_* SDK calls
   ▼
 libvortex.so  ── driver stub; VORTEX_DRIVER selects the backend
   ▼
 libvortex-{simx,rtlsim,xrt,opae}.so  ──►  Vortex device
```

The Vortex `sw/` runtime tree is untouched by OpenCL as a host API —
everything load-bearing on the host side (the ICD, the OpenCL→LLVM
pipeline, the SPIR-V reader, the Vortex device driver, the JIT to
`.vxbin`) lives in the external PoCL fork. What the Vortex repo owns is the
**device-side** contract every kernel links against, plus the test and CI
glue.

The device is reached through the standard Vortex SDK, so an OpenCL run is
backend-agnostic: `VORTEX_DRIVER=simx` for the C++ model,
`rtlsim`/`xrt`/`opae` for RTL and FPGA. Only `libvortex.so` and below know
which.

---

## 2. In-tree vs. external ownership

| Path | Role |
|---|---|
| [`tests/opencl/`](../../tests/opencl/) | 48 real OpenCL applications (`vecadd`, `sgemm`, `blackscholes`, `bfs`, `nearn`, the `copybuf`/`dotproduct` smoke pair, …). Each is a host `.cpp` + one or more `.cl` kernels. |
| [`tests/opencl/common.mk`](../../tests/opencl/common.mk) | The build/run engine: host program links the system ocl-icd loader (`-lOpenCL`); kernels are JITed by PoCL at runtime. Threads `POCL_VORTEX_XLEN`, `POCL_VORTEX_CFLAGS`, `POCL_VORTEX_LDFLAGS`, and `POCL_VORTEX_BINTOOL` into PoCL so the JIT targets the same device config as the rest of the build. Pins `OCL_ICD_VENDORS` to the Vortex `.icd` so a coexisting vendor loader (e.g. CUDA's `libOpenCL`) is not picked up. |
| [`ci/testcases/opencl.yaml`](../../ci/testcases/opencl.yaml) | The `opencl` catalog category — one case per driver over `xlen: [32, 64]`; the A-extension apps (`atomicreduce`, …) run on `simx`/`rtlsim` under `-DVX_CFG_EXT_A_ENABLE`. Selected with `pytest ci -m opencl`. |
| [`sw/kernel/`](../../sw/kernel/) → `libvortex2.a` | The **device runtime** every kernel ELF links against: `vx_vprintf` (the printf backend), the KMU entry/startup (`vx_start.S`), and the intrinsic surface. |
| [`sw/kernel/include/`](../../sw/kernel/include/) | Device headers the PoCL kernel builtin library compiles against (`VX_types.h`, the intrinsics). |
| [`ci/toolchain_install.sh.in`](../../ci/toolchain_install.sh.in) / [`ci/toolchain_prebuilt.sh.in`](../../ci/toolchain_prebuilt.sh.in) | `pocl()` fetches / packages the prebuilt `$TOOLDIR/pocl` tarball; in the default `--all` set. |

Everything else — `lib/CL/devices/vortex/` (the device driver),
`lib/kernel/vortex/` (the builtin overrides), and the OpenCL→LLVM
pipeline — lives in **`vortexgpgpu/pocl` branch `vortex_3.x`**, built
against `$TOOLDIR/llvm-vortex` and installed into `$TOOLDIR/pocl`. The
references to PoCL in [`tests/opencl/common.mk`](../../tests/opencl/common.mk)
are the only in-tree coupling.

---

## 3. The PoCL Vortex device driver

The driver is `lib/CL/devices/vortex/` in the PoCL fork. Two files carry it:
`pocl-vortex.c` (the `pocl_device_ops` table, device init, kernel launch,
memory, and images) and `vortex_utils.cc` (the LLVM-level kernel transform
and the JIT to `.vxbin`).

### 3.1 Device model

- **One PoCL device per Vortex device**, opened with `vx_device_open`. The
  ISA is queried once (`VX_CAPS_ISA_FLAGS`) and validated against what the
  driver actually compiles and emits, so every later decision (software vs.
  fixed-function image sampling, whether atomics are legal) consults the
  real device rather than assuming a unit is present.
- **Reported caps are derived from the device, not the build-time
  extension list.** `CL_DEVICE_EXTENSIONS` / `CL_DEVICE_OPENCL_C_FEATURES`
  are built from `isa_flags` at init and kept for the process lifetime (the
  `cl_device_id` and its info queries outlive any single context). §5
  details the capability model.
- **Address space.** `address_bits` is 32 or 64 per `POCL_VORTEX_XLEN`; the
  LLVM target triple and ABI follow (`riscv32`/`ilp32f`,
  `riscv64`/`lp64d`).

### 3.2 Program build — OpenCL C / SPIR-V to `.vxbin`

A `clBuildProgram` (or `clCreateProgramWithIL` for SPIR-V) runs the PoCL
central pipeline, then the Vortex `post_build_program` hook:

1. **Front end.** OpenCL C → Clang → LLVM IR, or SPIR-V →
   `spirv_parser` → LLVM IR. SPIR-V is opt-in (`ENABLE_SPIRV`) and is what
   the chipStar/HIP path feeds in.
2. **Builtin link.** The module is linked against the per-XLEN kernel
   bitcode library `kernel-riscv{32,64}.bc` (§4).
3. **Work-group passes.** PoCL's work-group transforms run
   (`run_passes_on_program`).
4. **`processKernels` (`vortex_utils.cc`).** Each kernel entry is rebuilt
   into an **argument-buffer wrapper**: a new function that unpacks
   arguments from a single device buffer and calls the body. The wrapper
   is tagged with the `vortex-kernel` function attribute — the backend
   keys both the **kernel calling convention** (no callee-saved spills;
   the KMU trampoline never reads them back) and the **divergence pass's
   device-module detection** off it. A kernel that is *also* called as a
   plain function by another kernel (legal in OpenCL C) is first cloned to
   an internal `.callee` copy and the in-module call sites redirected, so
   moving the body into the wrapper does not leave a dangling call. The
   module is `verifyModule`-checked around this transform: broken IR is
   reported as a build failure instead of faulting in codegen.
5. **Codegen + ELF link.** The module is compiled to an object and linked
   into a device ELF with `libvortex2.a` (device runtime; provides
   `vx_vprintf`, the KMU entry) and newlib `libc.a` (provides the libm
   entry points Clang lowers to calls — `cbrt`, `erf`, `hypot`, `tgamma`,
   … — which are *not* self-contained in the builtin library). `--gc-sections`
   drops everything unreferenced.
6. **`vxbin.py` objcopy** packages the ELF into a `.vxbin`, which is loaded
   as a `vx_module`.

**Code-region slots.** Each live program's `.vxbin` is linked at a distinct
16MB code base so multiple programs in one context never overlap. The
driver hands out slots from a fixed pool (`VORTEX_NUM_MODULE_SLOTS`) and
**returns a slot when its program is freed** — a long-lived process
(a conformance run building thousands of programs) would otherwise exhaust
the code address space.

### 3.3 Kernel launch — `pocl_vortex_run`

At `clEnqueueNDRangeKernel` the driver packs one **argument buffer** and
launches through the KMU:

- **Pointer args** are written as device addresses. For a sub-buffer the
  base is the parent's device address plus the sub-buffer origin, so the
  kernel sees a correctly-offset pointer with no separate relocation.
- **`__local` args and automatic locals** are assigned offsets into the
  device scratchpad, each aligned to `VX_LOCAL_ALIGN` (128 B — the widest
  OpenCL type, `long16`/`double16`, and also the local-memory bank stride,
  so the padding is free in bank-conflict terms).
- **POD args** are copied inline at their ABI offset.

The KMU entry glue (`kernel_main.c`) publishes `g_work_dim` and
`g_global_offset` for the builtin `get_*` queries, then enters the
per-kernel wrapper. Work-items map onto Vortex **warps × threads**; a
work-group runs as one CTA, so `CL_DEVICE_MAX_WORK_GROUP_SIZE` is
`num_warps × num_threads`.

### 3.4 Memory and images

- **Buffers.** `alloc_mem_obj` wraps a `vx_buffer`; read/write/copy/fill and
  their rect variants map to the `vx_enqueue_*` async ops, each adding the
  sub-buffer origin (byte offset into the shared parent `vx_buffer`).
- **Sub-buffers** allocate no device memory of their own — the wrapper
  retains the parent `vx_buffer` and records the region origin.
- **Images** are served in software: `read_image`/`write_image` and the
  `get_image_*` queries are compiled into the kernel library (§4), and the
  six image transfer ops (`{read,write,copy}_image_rect`, `map`/`unmap`,
  `fill`) are wired in the ops table. Each image object carries a small
  device buffer holding its lifetime-constant `dev_image_t` descriptor,
  uploaded once at allocation, so binding an image adds no per-launch
  host↔device traffic. `CopyBufferToImage`/`CopyImageToBuffer` route to the
  device-to-device rect copy. When the device exposes a fixed-function TEX
  unit (`VX_ISA_EXT_TEX`), FF-eligible 2D reads can instead go through the
  hardware sampler; otherwise every sample is software.

---

## 4. The kernel builtin library

The device-side OpenCL C library is `lib/kernel/` in the PoCL fork,
compiled per-XLEN to `kernel-riscv{32,64}.bc` and linked into every
program (§3.2 step 2). Most of it is PoCL's generic, target-independent
builtin set (all of math, integer, relational, geometric, common,
conversion, async-copy, and synchronization). `lib/kernel/vortex/` carries
only the overrides Vortex genuinely needs:

| Override | Why |
|---|---|
| `read_image.cl` / `write_image.cl` | Scalar software sampling over the `dev_image_t` descriptor across every shape (1D / 1D-array / 1D-buffer / 2D / 2D-array / 3D, nearest + linear, all addressing modes). The generic PoCL versions rely on OpenCL vector arithmetic the no-vector Vortex backend (`+xvortex`) cannot select. |
| `workitems.c` | `get_global_id` etc. read the KMU CSRs, which decode only dims 0–2; the spec-mandated result for an out-of-range `dimindx` (sizes/counts → 1, ids/offsets → 0) is enforced here. |
| `atomics.c` | The 32-bit AMO builtins lower to hardware `amo*.w`. |
| `barrier.c` | Work-group barrier on the Vortex `vx_barrier`. |
| `printf.c` | Buffered printf; the host drains it (`device_side_printf = 0`). |
| `wait_group_events.cl` | A **real** work-group barrier. PoCL's generic stub is empty — sound only under the work-item-loop model where a group runs sequentially on one thread. A Vortex work-group spans independent hardware warps, so the stub would let non-copying warps race ahead of an async copy. |
| `vxlibm.c` | libm entry points Clang lowers to calls that newlib does not provide accurately enough (e.g. `nextafterf`) implemented directly via IEEE-754 integer encoding. |

The `.bc` is intentionally **not** self-contained: it leaves the C99 libm
symbols (`cbrt`, `erf`, `hypot`, `tgamma`, `remainder`, …), the KMU globals
(`g_work_dim`, `g_global_offset`), and `vx_vprintf` undefined, to be
resolved at the final kernel ELF link from newlib `libc.a` and
`libvortex2.a` respectively (§3.2 step 5). Any genuinely missing builtin
therefore fails **loudly at link**, in the one program that needs it, rather
than mis-executing silently.

---

## 5. Capability and conformance model

The device advertises exactly what both the hardware and the kernel
library can honor — nothing aspirational:

| Capability | Status |
|---|---|
| Profile / version | Full-profile OpenCL 1.2. |
| `int64` | **Core** (full-profile CL 1.2); soft-lowered on rv32. Reported via `has_64bit_long` / `__opencl_c_int64`, never as the non-standard `cl_khr_int64` token (the CTS rejects unapproved `cl_khr_*` tokens). |
| Atomics | int32 base/extended, gated on the RISC-V `A` extension. A device without `A` drops every `*atomics*` extension token *and* rejects an atomic-using program at build time. |
| Images | Supported, in software (optionally FF-accelerated for 2D reads). |
| printf | Host-buffered. |
| `fp64` (`cl_khr_fp64`) | **Not advertised** — the kernel library has no double-precision builtins, so claiming it would be a promise the toolchain cannot keep. Optional in CL 1.2; the CTS `double` paths auto-skip. |
| `fp16` (`cl_khr_fp16`) | Not advertised (no half builtins). Optional; CTS `half` auto-skips. |
| `max_mem_alloc_size` | `max(global/4, 128MB)`, clamped to global — the CL 1.2 floor. Reporting the full global size is conformance-hostile because code/stack regions are reserved. |
| `max_constant_buffer_size` | 64KB (CL 1.2 floor); constants live in global memory. |

**Conformance harness.** The OpenCL-CTS runs per-category, strictly serial,
against a SimX device (a dedicated `build_ocl_cts` tree; the functional
SimX kernel with MT workers is used to speed architectural runs — cycles
are non-physical but bit-exact). Fixes follow the true-GPU rule: a missing
*device* capability is added to Vortex (the kernel library or the driver),
never emulated in PoCL software.

### 5.1 Known limitations

- **3D image sampling (`read_imagef(image3d_t, …)` trilinear path).** The
  switch-heavy CFG of the software 3D sampler currently trips the Vortex
  SIMT branch-divergence structurizer, which emits `vx.split`/`vx.join`
  pairs that violate SSA dominance; with PoCL's verifier disabled this
  reaches codegen and faults. This is a **compiler** limitation, not a
  driver gap; its blast radius is 3D-image sampling only.
- **fp64 / fp16 conformance** is out of scope until double/half builtins
  exist — both are optional in CL 1.2 and correctly unadvertised.

---

## 6. Cross-references

- [HIP on Vortex (chipStar)](hip_on_vortex_chipstar.md) — the HIP host API
  layered on this same PoCL device driver (`CHIP_BE=opencl`).
- [vortexpipe architecture](vortexpipe_architecture.md) — the Vulkan/compute
  front end that bypasses PoCL and emits `.vxbin` directly.
- [Kernel Entry and Dispatch](kernel_entry_and_dispatch.md) — the KMU
  launch and argument-buffer ABI the driver targets.
- [Vortex Runtime API](vortex_runtime_api.md) — the `vortex2.h` SDK surface
  (`vx_*`) the driver is written against.
- [Atomic Memory Operations](atomic_memory_operations.md) — the `A`-extension
  hardware behind the OpenCL atomic builtins.
- [Texture Sampler](texture_sampler_architecture.md) — the fixed-function
  TEX unit the image path optionally routes 2D reads through.
