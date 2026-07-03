# vortexpipe — software, compiler, and rendering-pipeline architecture

This document describes the `vortexpipe` Gallium driver that lives in
[`mesa_vortex`](https://github.com/vortexgpgpu/mesa) at
`src/gallium/drivers/vortexpipe/`. It covers three things:

1. The **software architecture** — how vortexpipe plugs into Mesa /
   Gallium / lavapipe and how a draw or dispatch reaches the Vortex
   device.
2. The **compiler architecture** — how NIR shaders become Vortex
   `.vxbin` kernels, and how the translator detects and emits the
   Vortex graphics ISA.
3. The **rendering pipeline** — what happens end-to-end for one
   `vkCmdDraw` from a Vulkan app, including the VS/raster/FS/OM
   stages and the host↔device traffic between them.

Filename references use the upstream layout in `mesa_vortex`. Vortex
graphics ISA mnemonics (`vx_tex4`, `vx_om4`, the `SETW`/`GETW`/`GETWS`
window ops, `vx_barrier`) and CSR numbers come from
`sw/kernel/include/vx_graphics.h` + `sw/kernel/include/vx_gfx_window.h`
in this repo. The dispatch model is FWD-5 push (§2.3.1, §3.4).

---

## 1. Software architecture

### 1.1 What vortexpipe *is*

vortexpipe is not a from-scratch driver. It is a **thin decorator on
top of `llvmpipe`** that:

- Owns the llvmpipe `pipe_screen` and `pipe_context` lifecycle, so
  vortexpipe-side state can be threaded through them.
- Overrides only the entry points it specializes — context creation,
  compute hooks (`*_compute_state`, `launch_grid`, descriptor
  binding), and the graphics pipeline-state + draw hooks (vertex /
  fragment / depth-stencil / blend / vertex-elements / textures +
  samplers + framebuffer + `draw_vbo`).
- Forwards everything it doesn't override to llvmpipe.

Each overridden entry point follows the same pattern: vortexpipe
patches the vtable slot in place, capturing the previous pointer in a
side struct (`struct vp_screen` / `struct vp_context`) that is keyed
off the llvmpipe screen / context pointer in a process-wide hash table
(`vp_reg_put` / `vp_reg_get` / `vp_reg_del` —
[`vp_context.c:59-100`](../../src/gallium/drivers/vortexpipe/vp_context.c#L59)).
This avoids the alternative of a ~140-thunk full decorator while
keeping the override surface small enough to audit at a glance.

Patching the vtable is legitimate here because vortexpipe **created**
the llvmpipe screen — it owns the base.

### 1.2 Layering

```
              Vulkan app
                  │
            ┌─────▼──────┐
            │  lavapipe  │   (Vulkan → Gallium translation)
            └─────┬──────┘
                  │ pipe_screen / pipe_context
            ┌─────▼──────┐
            │ vortexpipe │   ← this driver: vtable interceptions
            └─────┬──────┘
                  │ forwarded vtable calls
            ┌─────▼──────┐
            │  llvmpipe  │   (TGSI/NIR-on-CPU baseline + util_blitter)
            └────────────┘
```

The Vortex device sits *beside* this stack, reached through the
Vortex SDK (`libvortex.so`, header `vortex2.h`). vortexpipe's
specialized entry points are the only places that ever touch
`vx_*` calls.

### 1.3 Per-screen state — `struct vp_screen`

Defined in
[`vp_private.h:29-61`](../../src/gallium/drivers/vortexpipe/vp_private.h#L29).
Lives for the screen's lifetime. The two important groups:

- **Device handle + saved llvmpipe vtable slots.** `vx_device_open(0,
  &dev)` runs once when the screen is created
  ([`vp_screen.c:81`](../../src/gallium/drivers/vortexpipe/vp_screen.c#L81)).
  The screen's `context_create`, `destroy`, and `get_name` slots are
  replaced; the originals are saved as `lp_context_create`,
  `lp_screen_destroy`, `lp_screen_get_name` so they can still be
  invoked on the forward path.
- **Device caps cached up front.** `hw_num_threads`, `hw_num_warps`,
  `hw_max_block_size`, `hw_isa_flags`, and three booleans `has_tex /
  has_raster / has_om` derived from `VX_ISA_EXT_*` bits
  ([`vp_screen.c:87-101`](../../src/gallium/drivers/vortexpipe/vp_screen.c#L87)).
  Caching these here lets every `launch_grid` and `draw_vbo` decide
  *fast* (no per-call `vx_device_query`) whether the workload fits
  one CTA and whether the hardware actually exposes the graphics
  fixed-function units.

The screen ctor also **clamps llvmpipe's advertised compute caps** to
the Vortex device's `hw_max_block_size`
([`vp_screen.c:128-137`](../../src/gallium/drivers/vortexpipe/vp_screen.c#L128))
so well-behaved Vulkan apps that read
`maxComputeWorkGroupInvocations` pick a workgroup that fits one CTA
in the first place. Apps that ignore the cap are caught at launch
time by an explicit refusal
([`vp_context.c:194-205`](../../src/gallium/drivers/vortexpipe/vp_context.c#L194))
and fall back to llvmpipe.

### 1.4 Per-context state — `struct vp_context`

Defined in
[`vp_private.h:121-204`](../../src/gallium/drivers/vortexpipe/vp_private.h#L121).
Carries the *bound* Gallium state vortexpipe needs at launch / draw
time:

- `cur_cso`, `cur_vs`, `cur_fs` — currently bound compute / vertex /
  fragment programs, each a `struct vp_cso` that pairs the original
  llvmpipe CSO with a translated Vortex `.vxbin`
  ([`vp_private.h:68-76`](../../src/gallium/drivers/vortexpipe/vp_private.h#L68)).
- `cbuf[8]`, `cbuf_off[8]` — compute constant buffers, captured in
  `set_constant_buffer`. lavapipe binds the descriptor buffer for
  descriptor-set N at constant-buffer index `N+1`, so `cbuf[1]` is
  the set-0 descriptor buffer the kernel will reach.
- `cur_dsa`, `cur_blend`, `cur_velems`, `cur_tex`, `cur_sampler`,
  `vbufs[]`, `fb_color`, `fb_depth`, `fb_width`, `fb_height` —
  pre-encoded graphics state captured as the Vulkan-side app binds
  it. The captured form is the *Vortex* encoding (e.g.
  `VX_OM_DEPTH_FUNC_*` packed words, not Gallium enums), so the draw
  path can write them straight into Vortex DCRs.
- The saved llvmpipe vtable slots for every entry point vortexpipe
  intercepts (`lp_create_compute_state`, `lp_bind_*_state`,
  `lp_set_constant_buffer`, `lp_draw_vbo`, `lp_set_framebuffer_state`,
  …).

### 1.5 The fallback contract

vortexpipe is a "best-effort accelerator on top of llvmpipe": every
overridden entry point either succeeds on Vortex or **forwards to the
saved llvmpipe slot**. The fallback is deliberate — it lets the
driver expose full Gallium capability (and pass lavapipe's
self-tests) even when a particular pipeline state combination isn't
covered yet.

In CI and other safety-critical contexts the silent fallback would
mask regressions, so vortexpipe has a STRICT mode gated on
`$MESA_VORTEX_STRICT`
([`vp_context.c:242-253`](../../src/gallium/drivers/vortexpipe/vp_context.c#L242),
also in `vp_draw_vbo`). When set, a missing Vortex path becomes a
`mesa_loge` and the call becomes a no-op so the application's own
validation step catches the data not landing.

The fallback is gated **per call**, not per pipeline. Some draws can
run on Vortex while their neighbours don't — a vertex shader whose
inputs the VS translator handles will execute on the device, but if
its companion fragment shader uses something the FS translator
doesn't yet cover, the VS still runs on Vortex and the rasterization
follows on llvmpipe through a cached passthrough VS
([`vp_context.c:1057-1064`](../../src/gallium/drivers/vortexpipe/vp_context.c#L1057)).

#### 1.5.1 Gated fallback vs. silent collapse

Not every "unsupported" feature is a gated fallback. Two kinds of
selection coexist in the current code and only the first preserves
Vulkan conformance:

- **Gated** — code that **detects** an unsupported case and routes
  the work to llvmpipe (or fails in STRICT mode). Examples:
  ISA-cap missing (§1.3 `has_tex/has_raster/has_om`), non-`texop_tex`
  NIR op in the FS, NPOT texture dim, non-simple `draw_vbo` shape.
- **Silent collapse** — code that **accepts** the call and projects
  it down to the nearest gfx-v1 encoding without telling the caller.
  Examples: mipmap/anisotropic filters → POINT, clamp-to-border /
  mirror-clamp → CLAMP, non-RGBA8 texture format → reinterpreted as
  RGBA8 by the readback memcpy.

Silent collapse is a **known conformance hole** — a Vulkan-CTS run
against gfx-v1 will produce wrong pixels (not refused draws) for
those inputs. See §3.2 for the full catalog and the work needed to
turn each silent collapse into a gated fallback.

---

## 2. Compiler architecture

### 2.1 Pipeline at a glance

```
   NIR shader (from lavapipe, post-SPIR-V → NIR → opt + lowering)
        │
        ▼
   vp_nir_to_llvm   ← scalar walker; emits one LLVM-IR module
        │
        ▼
   LLVM IR text  (riscv32-unknown-elf  +xvortex +zicond,  rv32imaf / ilp32f
                  or riscv64-unknown-elf  rv64imafd / lp64d depending on
                  $MESA_VORTEX_XLEN)
        │
        ▼
   vp_compile_vxbin
        ├──→ system("clang … -lvortex2 …")  ← link with libvortex2.a (KMU
        │                                      device kernel runtime)
        ▼
   .vxbin (kernel image)
        │
        ▼
   vp_launch / vp_launch_vs / vp_raster_draw
        └──→ vx_module_load_file + vx_enqueue_launch
```

### 2.2 Translator stage — `vp_nir_to_llvm`

[`vp_nir_to_llvm.c`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c)
(1700-ish lines) walks the lavapipe-lowered NIR and emits a single
LLVM-IR module per shader. The design is intentionally **scalar
walker, not LLVM PassManager** — no SLP / vector reflowing, no NIR-to-
NIR lowering inside the translator. Three shader stages map onto two
output shapes:

| NIR stage | LLVM function shape | KMU entry |
|-----------|---------------------|-----------|
| compute   | `void kernel_main(ptr %arg)` — one thread per work-item | `kernel_main` |
| vertex    | `void kernel_main(ptr %arg)` — one thread per vertex     | `kernel_main` |
| fragment  | `void fs_main(ptr %in, ptr %out, ptr %texstate)` wrapped by an emitted straight-line run-once `kernel_main` (`emit_fs_wrapper`, [`vp_nir_to_llvm.c`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c)) — RASTER launches it (§2.3.1, §3.4) | the wrapper's `kernel_main` |

Internal state ([`struct vp_tr`,
`vp_nir_to_llvm.c:98-128`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L98))
carries an LLVM context, module, builder, and a per-SSA-def value
table (`val[idx][component]`) that holds each NIR component as a raw
`iN` bit pattern. Operations bit-cast to whichever interpretation
(float, int, pointer) they need. Vertex and fragment stages add a
small amount of stage-specific state (`vid`, `out_base`,
`attr_table`; `fs_in_base`, `fs_out_base`).

Key reusable primitives:

- `emit_csr_read(t, csr, name)` — inline-asm `csrr` reading a Vortex
  CSR (CTA thread / block IDs, tmask, etc.).
- `emit_vx_barrier(t)` — `custom-0 funct3=4` with the CTA id as
  barrier id and the CTA's warp count as the count, matching
  `vx_spawn2.h::__syncthreads()`.
- `emit_vx_frag_payload(t, word)` — `custom-1 funct3=4` **`GETWS`**:
  slot-indexed window read of the pre-seeded frag record
  (`{pos_mask, pid}`), keyed by `block_idx`. The FS wrapper decodes the
  covered quad and recomputes per-corner edge values from the primitive
  `edges` (no bcoord CSRs).
- `emit_vx_tex(t, u, v, lod)` — `custom-1 funct3=5` **`vx_tex4`**
  (single mode). Returns the filtered texel as a packed `A8R8G8B8` i32;
  or a `gfx_tex_sample_sw` call when TEX is routed to software.
- `emit_vx_om4(t, desc, base)` — `custom-1 funct3=2 R-type`, `rd=x0`
  fire-and-forget. Submits a covered 2×2 quad (`desc = pos_mask |
  face<<31`) to the OM unit, which does depth-stencil + blend and writes
  the colour/depth buffers via its own AXI master; or a
  `gfx_om_fragment_sw` call when OM is routed to software.
- window ops `SETW` / `GETW` (`funct3=6`) stage/read the shared graphics
  register window. There is **no** `emit_vx_rast`/`emit_vx_rast_begin` —
  RASTER has no shader op (§2.3.4).

### 2.3 How the compiler **detects and selects** Vortex graphics ISA

There is no per-instruction "should I use TEX?" decision in the
translator — the selection happens at three earlier and clearer
points.

#### 2.3.1 Selection by shader stage

The translator routes on `nir->info.stage`: compute and vertex stages
become a plain `kernel_main(ptr %arg)`; fragment becomes
`fs_main(ptr %in, ptr %out, ptr %texstate)` wrapped by an emitted
`kernel_main` (`emit_fs_wrapper`). Under the **FWD-5 push model** the
wrapper is **straight-line, run-once — not a poll loop**. The RASTER
fixed-function unit *launches* the FS as a bare 1-warp CTA per
covered-quad wave and pre-seeds the per-lane payload into the warp's
graphics register window at launch; the wrapper reads its record with
`vx_frag_load` (a slot-indexed **`GETWS`**, funct3=4, keyed by
`block_idx`), recomputes the per-corner edge (barycentric) values from
the primitive `edges` + quad origin, runs `fs_main` per covered
sub-pixel, and returns. There is **no shader-issued raster op** — the
retired `vx_rast`/`vx_rast_fetch` pull, `vx_rast_begin`, and the bcoord
CSRs are gone. The windowed `vx_om4` / `vx_tex4` are emitted on the FS
path because the stage is FS, not because a NIR opcode asked for them.

#### 2.3.2 Selection by NIR opcode

Compute / vertex stages map NIR opcodes onto Vortex intrinsics through
the `emit_intrinsic` and `emit_tex` switches:

- `nir_intrinsic_barrier` with execution scope ≠ NONE →
  `emit_vx_barrier`
  ([`vp_nir_to_llvm.c:840`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L840)).
  Pure memory barriers in this per-thread model are no-ops.
- `nir_intrinsic_load_workgroup_id` / `_local_invocation_id` /
  `_num_workgroups` → CSR reads of `VX_CSR_CTA_BLOCK_ID_X / +c`,
  `_THREAD_ID_X / +c`, `_GRID_DIM_X / +c`
  ([`vp_nir_to_llvm.c:660-670`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L660)).
- `nir_intrinsic_load_vertex_id{,_zero_base}` → `t->vid` (the CTA
  thread id captured in the VS prologue).
- `nir_intrinsic_load_input` (VS) → `emit_vs_attr_addr` → a load from
  the per-attribute `{base, stride}` table arg slot 1 points at.
- `nir_intrinsic_load_vertex_id{,_zero_base}` → the index-resolved
  `vid` (an indexed draw resolves `index_buf[raw_id]` in the VS
  prologue); the VS *output* slot uses the sequential global id `vraw`
  so records are written in draw order for in-order triangle assembly.
- `nir_tex_instr` with `op == nir_texop_tex` → `emit_vx_tex`, which
  emits the **windowed `vx_tex4`** (single mode, LOD 0) after converting
  the float UVs to S.23 fixed-point — or, when TEX is routed to software
  (§2.3.3), a call to `gfx_tex_sample_sw`. No other tex op is currently
  lowered; anything else falls back.

If a NIR opcode has no mapping, the translator sets `t->ok = false`
and the whole shader fails translation (`vp_nir_to_llvm` returns
`false`), at which point the consumer call site
(`vp_create_compute_state` / `_vs_state` / `_fs_state`) keeps the
llvmpipe CSO around without a `vxbin` and the per-call fallback at
`launch_grid` / `draw_vbo` kicks in.

#### 2.3.3 Selection by device capability — per-unit HW/SW routing

The runtime decides, **per FF unit at FS-compile time**, whether each
stage runs on its hardware unit or its on-device SIMT software fallback
(`libgfx_sw`), from the device caps and the pipeline state.
`vp_fs_routing` computes `sw_tex` / `sw_om` / `sw_raster` (SW-raster
implies SW-OM — it has no FF window to merge through) from
`has_raster` / `has_om` / `has_tex` (the cached `VX_ISA_EXT_*` bits of
`VX_CAPS_ISA_FLAGS`) plus whether the draw needs a feature the FF unit
lacks. A unit that is absent or unfit routes **that unit** to software,
**not the whole draw to llvmpipe** (full residency, charter pillar 4).
The FS is co-compiled with `gfx_sw_abi.cpp` (divergence-bbs guard)
whenever any unit is SW; `emit_vx_tex` / the OM path / the wrapper then
emit the `gfx_*_sw` calls in place of the FF ops.

`$VORTEXPIPE_SW_RASTER` forces the SW-raster path even on a capable
device (A/B'ing). Two known residual gaps (tracked in the master plan):
a coarse `gfx_hw = has_raster && has_om` check still drops some
unsupported state *wholly* to llvmpipe rather than to SW (`L4`), and the
VS-on-Vortex → host-readback → llvmpipe-raster path
(`vp_draw_passthrough`) is still reachable at runtime (`L1`) — both to be
retired so llvmpipe is an offline oracle only.

#### 2.3.4 Selection by encoding constants

Vortex's graphics ISA uses the **RISC-V custom-1 opcode** (43 decimal
= 0x2B). `vp_nir_to_llvm` emits the instructions through LLVM inline
asm with `.insn r 43, funct3, …` / `.insn r4 43, funct3, …`
templates. The `funct3` map (byte-identical to the kernel SDK
`sw/kernel/include/vx_graphics.h`, and verified against
`hw/rtl/core/VX_decode.sv` + `sim/simx/decode.cpp`):

| `funct3` | Mnemonic  | What it does                                                   |
|----------|-----------|---------------------------------------------------------------|
| 2        | `vx_om4`  | submit a 2×2 quad to OM (R-type, `rd=x0` fire-and-forget)      |
| 4        | `GETWS`   | slot-indexed window read — the FS frag-record read (`block_idx`) |
| 5        | `vx_tex4` | sample TEX; single / quad via `funct7.mode`                   |
| 6        | window    | `SETW` / `GETW` / `GETWF` / `CB_RET` (by `funct2`)             |
| 7        | RTU       | `TRACE2` / `WAIT2` (by `funct2`)                               |

**`funct3` = 1 and 3 are unallocated and abort in the decoder** — the
legacy forms `vx_tex`(1), 3-operand `vx_om`(2), `vx_rast`(3), and
`vx_rast_begin`(4) are all retired across sw + simx + rtl + mesa. RASTER
has **no shader op**: it auto-arms on its DCR config write and launches
the FS itself. `vx_barrier` is on custom-0 (opcode 11), since custom-1
is reserved for graphics + RTU.

### 2.4 Backend stage — `vp_compile_vxbin`

[`vp_compile.c`](../../src/gallium/drivers/vortexpipe/vp_compile.c)
turns the LLVM-IR text from the translator into a `.vxbin` kernel
image by **fork/exec'ing the existing Vortex device toolchain**. The
in-process LLVM-API alternative would be cleaner but is deferred — the
shell out keeps the front-end and the device-side toolchain
decoupled.

The flags
([`vp_compile.c:145-172`](../../src/gallium/drivers/vortexpipe/vp_compile.c#L145))
mirror the canonical Vortex kernel toolchain invocation used by
`tests/regression/common.mk` in this repo:

- `--target=riscv{32,64}-unknown-elf`, `--sysroot` + `--gcc-toolchain`
  pointing at the GNU toolchain matching the chosen XLEN.
- `-Xclang -target-feature -Xclang +xvortex` — the Vortex ISA
  extension. **This is what makes Clang's RISC-V backend emit Vortex
  intrinsics + the SIMT-aware branch-divergence pass.**
- `-Xclang -target-feature -Xclang +zicond` — conditional-zero
  instructions (Vortex uses these for divergent control flow folding).
- `-mllvm -disable-loop-idiom-all` — keep loop bodies as-is (no
  memset/memcpy idiom recognition).
- The Vortex branch-divergence pass is left at its default (enabled).
  It's the pass that lowers divergent SIMT control flow into masked
  execution; explicitly disabling it via
  `-mllvm -vortex-branch-divergence=0` would break the SIMT semantics
  the kernel relies on.

The link line pulls in `libvortex2.a` (the KMU device kernel runtime,
which provides `vx_start.S`, `vx_putchar`, `vx_spawn2.h`'s
`__syncthreads`, etc.) plus the baremetal libc / compiler-rt. The
result is an ELF; `sw/kernel/scripts/vxbin.py` packages that ELF into
a `.vxbin`.

The target XLEN comes from `$MESA_VORTEX_XLEN` (default `32`; `64`
selects rv64). The env var is mesa-namespaced so it doesn't collide
with anything the linked `libvortex.so` runtime reads.

### 2.5 Launch stage — `vp_launch`, `vp_launch_vs`

[`vp_launch.c`](../../src/gallium/drivers/vortexpipe/vp_launch.c)
holds the host-side dispatch. The two entry points share the bones:
materialize the `.vxbin` in a temp file, `vx_module_load_file +
vx_module_get_kernel("kernel_main")`, build the kernel arg block,
upload everything, `vx_enqueue_launch`, read results back,
`vx_queue_finish`.

**The arg block is the contract** between the translator and the
launcher. It's a fixed-size `i64[VP_ARG_SLOTS=8]` array of device
addresses passed inline to the kernel via `vx_launch_info_t.args_host`:

| Slot | Compute (vp_launch)                      | Vertex (vp_launch_vs)            |
|------|------------------------------------------|----------------------------------|
| 0    | unused                                   | output vertex-record buffer      |
| 1    | set-0 descriptor buffer                  | vertex attribute table           |
| 2-7  | reserved / used by later descriptor sets | unused                           |

The arg block is what `nir_intrinsic_load_const_buf_base_addr_lvp`
and `_load_ubo` read in
[`vp_nir_to_llvm.c:691-749`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L691);
slot index maps directly to constant-buffer index.

For compute kernels the launcher walks the descriptor table the
translator pre-scanned (`vp_scan_descriptors` →
`cso->descs[0..num_descs)`), and for each descriptor:

- `VP_DESC_BUFFER` (an SSBO): allocate a Vortex device buffer the
  size of the host resource, upload, and **rewrite the
  `lp_jit_buffer.ptr` field** in the staged descriptor buffer to the
  device address. `load_ssbo` in the kernel then dereferences a
  device-side `i64` pointer at the descriptor slot. This is the
  bridge between the lavapipe-emitted descriptor format and Vortex
  device memory.
- `VP_DESC_AS` (an acceleration structure for ray tracing): copy the
  TLAS BVH into device memory, recursively copy each instance node's
  BLAS, and rewrite the absolute `bvh_ptr` links to the device
  address. Box-node children are BVH-relative offsets and survive
  the copy. ([`vp_launch.c:94-150`](../../src/gallium/drivers/vortexpipe/vp_launch.c#L94))

For VS launches the launcher uploads the vertex buffer plus a small
`{base, stride}[VP_ATTR_TABLE_LOCS=8]` table indexed by `driver_
location`; the VS kernel's `load_input` path reads from
`table[loc].base + vid * table[loc].stride`
([`vp_launch.c:381-396`](../../src/gallium/drivers/vortexpipe/vp_launch.c#L381),
together with `emit_vs_attr_addr` in the translator).

---

## 3. Rendering pipeline — what one draw call actually does

The end-to-end story for a Vulkan `vkCmdDraw` once it has been
translated to a `pipe_context::draw_vbo` call into vortexpipe
([`vp_draw_vbo`,
`vp_context.c:914-1076`](../../src/gallium/drivers/vortexpipe/vp_context.c#L914)).

### 3.1 Stage 0 — eligibility check

vortexpipe takes the Vortex device path for a **simple direct or
indexed, non-instanced single draw** with a translated VS:

```c
bool indexed = info->index_size == 2 || info->index_size == 4;
bool simple = vp->dev && vs && vs->vxbin && vs->vs_layout.stride &&
              !indirect && num_draws == 1 &&
              (info->index_size == 0 || indexed) &&
              !info->primitive_restart && info->instance_count == 1 &&
              draws[0].count > 0;
```

Anything else (instanced, multi-draw, indirect, prim-restart, no or
untranslatable VS) takes the wholesale llvmpipe fallback — or fails
loudly in STRICT mode. An indexed draw uploads its index buffer widened
to u32 (folding in the base-vertex bias); the VS resolves `index_buf[i]`
on device.

### 3.2 The device-orchestrated draw — `vp_raster_draw`

On the hardware-raster path the **whole draw is one device-resident
transaction** ([`vp_raster.cpp`](../../src/gallium/drivers/vortexpipe/vp_raster.cpp)):
the VS is *folded in* as the front end's stage 0 (no host readback of
transformed vertices), and the **on-device sort-middle front end**
produces the RASTER buffers. There is **no host `graphics::Binning`** in
the runtime path — that reference renderer is retained only as the
coverage oracle. The draw is recorded as one `DrawCommands` batch and
submitted with a single `vx_enqueue_draw` (one doorbell, one completion;
see [`command_processor.md`](command_processor.md) §8.1). The batch is
the nine front-end stage launches + FF DCR writes, drained in order by
the CP's launch-barrier:

1. **`expand_k`** — VS assembly, one thread per vertex: runs the
   translated VS and writes `setup_vertex_t` records (resident). Indexed
   draws resolve the index here (VS *output* slot = sequential `vraw`
   for in-order assembly).
2. **`setup_k`** — near-plane clip (Sutherland-Hodgman, ≤2 subtris) +
   front/back cull + fixed-point plane-equation setup → `rast_prim_t`
   (120 B: `edges[3]` + the affine attribs `{z,r,g,b,a,u,v}`) + per-prim
   counts.
3. **`binning_k`** — exact-sized parallel sort-middle (count→scan→emit,
   no overflow path) → dense primbuf + 12 B `rast_bin_header_t` + PID
   array.

Colour/depth/texture are **render-pass-resident (pinned-PA)** and reached
by the FF units through their DCRs; the on-device front end binds them,
not a host round-trip. (Residency-boundary host copies that still remain
— colour seed/readback — are tracked as `R2/R3` in the master plan.)

### 3.3 FF configuration + RASTER launch

`vp_raster_draw` then programs the RASTER + OM + (optional) TEX DCRs and
lets the RASTER engine launch the fragment shader itself:

1. **Program RASTER DCRs**: tile/prim buffer block-addresses + strides,
   scissor, and the **fragment-shader launch descriptor**
   (`VX_DCR_RASTER_FRAG_PC_LO/HI`, `FRAG_ENTRY`, `FRAG_PARAM`) — so the
   raster engine self-launches the FS with no host KMU grid.
2. **Program OM DCRs**: colour + depth buffer addresses/pitches,
   depth-compare + write-mask (bound DSA cso), colour-write-mask + blend
   mode/func (blend cso), stencil state, and the per-draw `EARLYZ_SAFE`
   gate.
3. **Program TEX DCRs** (if a sampler is bound): the resident texture's
   `VX_DCR_TEX_{ADDR, LOGDIM, FORMAT, FILTER, WRAP, MIPOFF}` for the stage.
4. **RASTER runs** the fixed-function walker → early-Z → packer →
   **dispatch**, which *launches* a bare 1-warp fragment CTA per
   covered-quad wave on the core-local KMU (pure-DCR — there is **no host
   FS grid launch**; the raster engine self-kicks). Each fragment CTA runs
   the FS wrapper **once** (§3.4).

### 3.4 The fragment shader (FWD-5 push, run-once)

The emitted `kernel_main` wrapper (`emit_fs_wrapper`,
[`vp_nir_to_llvm.c`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c))
runs once per launched wave:

```
frag         = vx_frag_load()                       // GETWS, slot = block_idx
prim         = arg[0] + frag.pid * 120
(qx,qy,mask) = decode(frag.pos_mask)
for each covered sub-pixel i:
    (f0,f1,f2) = recompute edge values  a·X + b·Y + c  at pixel (X,Y)
    dx = f0/(f0+f1+f2);  dy = f1/(f0+f1+f2)
    interpolate(prim.rast_attribs, dx, dy) → fs_in
    fs_main(fs_in, fs_out, texstate)                // vx_tex4  | gfx_tex_sample_sw
    rgba  = pack(fs_out)
    depth = fixed24(plane_z(prim, X, Y))
    vx_om4(frag.pos_mask | face<<31, om_slot_base)  // vx_om4   | gfx_om_fragment_sw
```

There is **no bcoord CSR read and no `vx_rast`/`vx_om` pull** — the payload was
seeded at launch and the edge values are recomputed from the primitive. `vx_om4`
submits the covered quad to the OM unit, which depth-tests / blends / writes
colour+depth at the DCR-configured PAs; the FS never sees the attachment
addresses. Same-pixel ordering is correct by construction (one screen tile →
one warp). When a unit is routed to software (§2.3.3) the wrapper calls the
matching `gfx_*_sw` in place of the FF op. The colour attachment stays
device-resident; only present copies it out.

### 3.5 Fallback paths (also valid in the same code)

If `gfx_hw` is false (the device lacks RASTER+OM, or
`$VORTEXPIPE_SW_RASTER` is set, or the FS shader isn't translatable):

- **VS on Vortex / raster on llvmpipe** (`vp_draw_passthrough`,
  [`vp_context.c:1057`](../../src/gallium/drivers/vortexpipe/vp_context.c#L1057)).
  The transformed-vertex buffer is presented to llvmpipe through a
  cached passthrough VS + matching `pipe_vertex_elements_state`, and
  llvmpipe's rasterizer takes over.
- **Everything on llvmpipe** — the original `lp_draw_vbo` is called.
  This is the path STRICT mode refuses.

### 3.6 TEX conformance gaps (current implementation vs. the spec)

The Vortex TEX block is a fixed-function 2D sampler: one stage, mip 0
only, A8R8G8B8 texels, S.23 fixed-point UV, point/bilinear filtering,
three wrap modes (CLAMP / REPEAT / MIRROR), no compare, no LOD bias,
no derivatives, no integer formats, no array/3D/cube/multisample.
Vulkan's `VkSampler` + `VkImageView` cover far more than that. Today
vortexpipe handles the gap with a mixture of gated fallback (correct)
and silent collapse (wrong-pixels-on-conforming-input).

| Spec input                              | Gfx-v1 capability     | What the code does                                              | Conformant?                |
|-----------------------------------------|-----------------------|------------------------------------------------------------------|----------------------------|
| Device lacks `VX_ISA_EXT_TEX`           | n/a                   | `vp_draw_vbo` clears `gfx_hw`, draw goes to llvmpipe              | **Yes** — gated            |
| NIR op other than `nir_texop_tex`       | not supported         | `emit_tex` sets `t->ok=false`, FS .vxbin null, draw goes to llvmpipe | **Yes** — gated         |
| Second sampler / multi-stage            | one stage             | covered incidentally — multiple derefs aren't emitted             | Incidental                 |
| Non-power-of-two texture dims           | POT only              | `vp_draw_vbo` logs warning, skips hardware RASTER+OM path         | **Yes** — gated            |
| Mipmap filter (`MIPMAP_NEAREST/LINEAR`) | LOD 0 only            | `vp_vx_filter` silently collapses to POINT                        | **No** — silent collapse   |
| Anisotropic, cubic                      | not supported         | `vp_vx_filter` silently collapses to POINT                        | **No** — silent collapse   |
| Clamp-to-edge / clamp-to-border         | CLAMP_TO_EDGE-ish     | `vp_vx_wrap` silently collapses to CLAMP                          | **No** — silent collapse   |
| Mirror-clamp-to-edge                    | not supported         | `vp_vx_wrap` silently collapses to CLAMP                          | **No** — silent collapse   |
| Non-RGBA8 colour formats (R16F, sRGB…)  | A8R8G8B8 only         | `vp_resource_rw` memcpys raw 32-bpp; no format check              | **No** — silent collapse   |
| Compare/shadow sampler                  | not supported         | covered incidentally — lowers to non-`texop_tex` NIR              | Incidental                 |
| Texel-fetch (`texelFetch`)              | not supported         | gated — `nir_texop_txf` rejected by emit_tex                      | **Yes** — gated            |
| `textureSize` / `textureQueryLod`       | not supported         | gated — `nir_texop_txs`/`lod` rejected by emit_tex                | **Yes** — gated            |
| 1D / 3D / cube / array / multisample    | 2D only               | gated indirectly — the deref+coord shapes get rejected            | Mostly incidental          |

The "Incidental" rows are conformance-correct **today** only because
lavapipe's NIR lowering happens to emit non-`texop_tex` ops or extra
derefs for those cases, which Layer 2 then refuses. They are not
explicitly tested in vortexpipe; a future lavapipe change could
silently turn them into accepted-but-wrong calls.

#### Closing the silent-collapse holes

To make gfx-v1 actually CTS-clean, three additions are needed (none
of them require new hardware):

1. **Sampler-state gate** in `vp_create_texture_handle`
   ([`vp_context.c:429-449`](../../src/gallium/drivers/vortexpipe/vp_context.c#L429)).
   Refuse — i.e. set a "sampler unsupported" flag on the captured
   sampler — when any of these is set in the `pipe_sampler_state`:
   `min_mip_filter != PIPE_TEX_MIPFILTER_NONE`, `max_anisotropy > 1`,
   `compare_mode != PIPE_TEX_COMPARE_NONE`, or `wrap_{s,t}` is one of
   `PIPE_TEX_WRAP_CLAMP_TO_BORDER` / `_MIRROR_CLAMP_*`. `vp_draw_vbo`
   already has the check shape — extend the existing `gfx_hw &&
   tex_needed && !vps->has_tex` clause to also test for this flag.
2. **Texture-format gate** in the same draw block, before
   `vp_resource_rw`. Accept only `PIPE_FORMAT_R8G8B8A8_UNORM`,
   `B8G8R8A8_UNORM`, and their `_SRGB` siblings (with sRGB→linear
   handled either by the readback or by the FS wrapper); everything
   else triggers the same llvmpipe fallback the POT check does.
3. **`emit_tex` tightening**
   ([`vp_nir_to_llvm.c:907-944`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L907)).
   Explicitly reject `tex->sampler_dim != GLSL_SAMPLER_DIM_2D`,
   `tex->is_array`, `tex->is_shadow`, and any source whose type isn't
   `nir_tex_src_coord` / `_texture_deref` / `_sampler_deref` (LOD
   bias, ddx/ddy, ms_index, etc.). This makes Layer 2 deliberate
   instead of relying on lavapipe's lowering choices.

Each addition is local: ~10 LOC in `vp_context.c`, ~10 LOC in
`vp_nir_to_llvm.c`, no runtime/ISA change. The result is that
every Vulkan call that the gfx-v1 TEX block cannot represent
either runs correctly on llvmpipe (default) or fails the test
loudly (STRICT mode) — never silently wrong.

### 3.7 OM conformance gaps

The Vortex OM block covers a substantial subset of the Vulkan
output-merger fixed-function surface — 8 depth-compares, 5 blend
equations (ADD / SUB / REV_SUB / MIN / MAX), 12 blend factors, RGBA
write-mask — but the Gallium surface is much bigger, and almost none
of the deltas are detected. The current encoding sits in
[`vp_create_dsa_state`](../../src/gallium/drivers/vortexpipe/vp_context.c#L676-L690)
and
[`vp_create_blend_state`](../../src/gallium/drivers/vortexpipe/vp_context.c#L720-L750);
both quietly drop fields they don't understand.

| Spec input                                  | Gfx-v1 capability    | What the code does                                                                  | Conformant?              |
|---------------------------------------------|----------------------|--------------------------------------------------------------------------------------|--------------------------|
| Device lacks `VX_ISA_EXT_OM`                | n/a                  | `gfx_hw = false`, draw goes to llvmpipe                                              | **Yes** — gated          |
| Stencil test/write                          | not supported        | `s->stencil[0/1].enabled` silently ignored ("gfx-v1 deferred")                       | **No** — silent collapse |
| Alpha test                                  | not supported        | `s->alpha_enabled` silently ignored                                                  | **No** — silent collapse |
| Depth-bounds test                           | not supported        | `s->depth_bounds_test` silently ignored                                              | **No** — silent collapse |
| Logic op                                    | not supported        | `s->logicop_enable` / `logicop_func` ignored                                         | **No** — silent collapse |
| MRT (`nr_cbufs > 1`)                        | RT 0 only            | only `rt[0]` read; extra RTs silently dropped                                        | **No** — silent collapse |
| Independent blend per RT                    | shared (RT 0 only)   | `s->independent_blend_enable` ignored                                                | **No** — silent collapse |
| Independent RGB/alpha blend equation        | shared equation      | `blend_mode = (m<<16) \| (m<<0)` — `rt->alpha_func` dropped                          | **No** — silent collapse |
| Dual-source blend (`SRC1_*`)                | not supported        | `vp_vx_blend_factor` `default:` returns `ONE`                                        | **No** — silent collapse |
| `CONSTANT_ALPHA` / `INV_CONSTANT_ALPHA`     | only `CONST_RGB`     | same `default:` → `ONE`                                                              | **No** — silent collapse |
| `alpha_to_coverage` / `alpha_to_one`        | not supported        | ignored                                                                              | **No** — silent collapse |
| sRGB write (`_SRGB` colour attachment)      | linear A8R8G8B8 only | OM writes linear bytes into the sRGB surface — curve double-applied by presentation  | **No** — silent collapse |
| Depth-clamp / depth-clip-disable            | not captured         | `pipe_rasterizer_state.depth_clip_*` not read                                        | **No** — silent collapse |
| Multisample (`samples > 1`)                 | single-sample only   | no check rejects multi-sample framebuffers                                           | **No** — silent collapse |
| Depth funcs (NEVER/LESS/EQ/LEQ/GR/NEQ/GEQ/ALWAYS) | all 8           | full 1:1 mapping in `vp_vx_depth_func`                                               | **Yes** — covered        |
| Blend ADD/SUB/REV_SUB/MIN/MAX               | all 5                | full 1:1 mapping in `vp_vx_blend_mode`                                               | **Yes** — covered        |
| Blend factors {ZERO,ONE,SRC_RGB,INV_SRC_RGB,DST_RGB,INV_DST_RGB,SRC_A,INV_SRC_A,DST_A,INV_DST_A,CONST_RGB,INV_CONST_RGB} | all 12 | full 1:1 mapping in `vp_vx_blend_factor`                                            | **Yes** — covered        |
| Colour write-mask (RGBA bits)               | 4 bits               | captured as `rt->colormask`                                                          | **Yes** — covered        |

#### Closing the OM silent-collapse holes

Three additions, all local and structurally identical to the TEX
fixes:

1. **DSA-state gate** in `vp_create_dsa_state`
   ([`vp_context.c:676-690`](../../src/gallium/drivers/vortexpipe/vp_context.c#L676-L690)).
   Capture an `unsupported` flag on `vp_dsa_cso` when **any** of
   `s->stencil[0].enabled`, `s->stencil[1].enabled`, `s->alpha_enabled`,
   `s->depth_bounds_test` is true. `vp_draw_vbo` already groups all
   the hardware-RASTER gates together — extend `gfx_hw` to also
   require `!cur_dsa->unsupported`.
2. **Blend-state gate** in `vp_create_blend_state`
   ([`vp_context.c:720-750`](../../src/gallium/drivers/vortexpipe/vp_context.c#L720-L750)).
   Same shape: mark `vp_blend_cso::unsupported` if **any** of
   `s->logicop_enable`, `s->independent_blend_enable`,
   `s->alpha_to_coverage`, `s->alpha_to_one`, `rt->rgb_func != alpha_func`,
   the rt[0] factors include `SRC1_*` / `CONSTANT_ALPHA`, or
   the host call binds more than one render target. Same `gfx_hw &&
   !cur_blend->unsupported` gate.
3. **Framebuffer-format gate** in `vp_set_framebuffer_state`
   ([`vp_context.c:524-535`](../../src/gallium/drivers/vortexpipe/vp_context.c#L524-L535)).
   Accept only `PIPE_FORMAT_R8G8B8A8_UNORM` / `B8G8R8A8_UNORM`
   (with explicit sRGB→linear handling if `_SRGB` is allowed at all),
   reject `samples > 1` and `nr_cbufs > 1`. Record a single
   `fb_unsupported` bit consulted by `vp_draw_vbo` next to the
   POT/TEX checks.

The hardware encoding itself doesn't change — these gates only
**refuse** the hardware path when the state isn't representable.

### 3.8 Rasterizer precision & geometric limits

The host-side `Binning`
([`sw/runtime/graphics.cpp`](../../../../sw/runtime/graphics.cpp))
and the on-chip RASTER both work in fixed-point:

- **Edge equations**: normalized by `1/maxVal` then stored as
  **Q15.16** (`EdgeToFixed`,
  [`graphics.cpp:137-151`](../../../../sw/runtime/graphics.cpp#L137-L151)).
  Sub-pixel resolution = 1/65536.
- **Scissor DCR**: `(width << 16) | y` and `(height << 16) | y`
  ([`vp_raster.cpp:202-203`](../../src/gallium/drivers/vortexpipe/vp_raster.cpp#L202-L203))
  — width/height live in 16 bits.
- **Tile header**: `tile_x`/`tile_y` are `uint16_t`, `pids_count` is
  `uint16_t`. With `RASTER_TILE_LOGSIZE = 5` (32-px tiles) the implied
  hard framebuffer ceiling is `65535 × 32 ≈ 2.1 M px`.
- **Barycentrics in the FS wrapper**: `VX_CSR_RASTER_BCOORD_{X,Y,Z}{0..3}`
  read as `fixed16`
  ([`emit_fs_wrapper`,
  `vp_nir_to_llvm.c`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L1393)).

| Limit / input                                    | Gfx-v1 capability       | What the code does                                                                                                  | Conformant?              |
|--------------------------------------------------|-------------------------|----------------------------------------------------------------------------------------------------------------------|--------------------------|
| Degenerate triangle (`det == 0`)                 | n/a                     | `EdgeEquation` returns false, `continue` + warning                                                                   | **Yes** — gated          |
| Off-screen after viewport clamp                  | n/a                     | `bbox.right <= bbox.left \|\| bbox.bottom <= bbox.top` → primitive skipped                                           | **Yes** — gated          |
| Zero tiles overall                               | n/a                     | `vp_raster_draw` returns false → llvmpipe fallback                                                                   | **Yes** — gated          |
| Framebuffer width/height > 65535                 | scissor packs 16 bits   | not checked — DCR wraps silently                                                                                     | **No** — silent overflow |
| Framebuffer dims s.t. tile count > 65535         | tile header packs 16 b  | not checked — `tile_header->tile_x/y` wraps silently                                                                 | **No** — silent overflow |
| Per-tile primitive count > 65535                 | `pids_count` is u16     | not checked — wraps silently                                                                                         | **No** — silent overflow |
| Triangle far outside viewport (no guardband)     | n/a                     | clipped only by screen-space bbox clamp; degenerate w produces sheared edges                                         | **No** — silent collapse |
| Near-plane intersection (w → 0)                  | n/a                     | no w-clip before binning — `ClipToScreen` divides by w, producing huge ps0/ps1/ps2                                   | **No** — silent collapse |
| Very large triangle (sub-pixel < advertised)     | Q15.16 sub-pixel        | `EdgeToFixed`'s `1/maxVal` normalization shrinks small edges; nothing rejects below `subPixelPrecisionBits`          | **No** — silent collapse |
| Attribute delta precision (`FloatA(a0 - a2)`)    | fixed-point             | not checked — large attribute magnitudes silently lose precision                                                     | **No** — silent collapse |
| Primitive type ≠ triangle list                   | triangle list only      | `vp_raster.cpp:106-107` emits `{i, i+1, i+2}` triples regardless of `info->mode`                                     | **No** — silent collapse |

#### Closing the rasterizer holes

All checks are host-side, in `vp_raster_draw` or `vp_draw_vbo`:

1. **Framebuffer-size gate** in `vp_set_framebuffer_state` (alongside
   the format gate proposed in §3.7): refuse `width > 65535` or
   `height > 65535`, and refuse anything that would produce
   `(width >> TILE_LOGSIZE) > 65535` or
   `(height >> TILE_LOGSIZE) > 65535`. These are the actual hardware
   limits the DCR encoding implies; today nothing enforces them.
2. **Primitive-mode gate** in `vp_draw_vbo`: refuse anything other
   than `PIPE_PRIM_TRIANGLES` until strips/fans/lines/points are
   wired through `vp_raster.cpp`'s primitive expansion. The
   silent-pretend-it's-a-list behaviour today produces wrong
   pixels, not refused draws.
3. **Per-tile pid-count guard** in `Binning`: when a single tile
   accumulates > 65535 primitives, return `0` from `Binning` so
   `vp_raster_draw` falls back. `tilebuf.resize` + the header writes
   already assume 16-bit counts; turning that assumption into an
   explicit check is ~5 LOC.
4. **Sub-pixel-precision guard**: after `EdgeToFixed`, the smallest
   normalized edge component sets the effective sub-pixel resolution.
   When `min(edge.x, edge.y)` < `1 / (1 << subPixelPrecisionBits)` of
   the largest component, the triangle's barycentrics fall below the
   advertised precision. Reject (skip the primitive) and fall back
   the whole draw if any primitive trips this — or, if performance
   matters, only the offending tile.
5. **W-clipping before binning**: run a Sutherland–Hodgman clip in
   homogeneous clip space against `w ≥ epsilon` before
   `ClipToScreen`, so near-plane crossings become 1–2 sub-triangles
   on the visible side instead of a single triangle with one vertex
   at infinity. This is the right long-term fix; it eliminates the
   silent-shear class entirely and is what the guardband would be
   protecting against anyway.

Items 1–3 are pure refusal gates (a few LOC each, no math change).
Items 4–5 are the substantive precision fixes; either of them — and
ideally both — is what a Vulkan-CTS-passing gfx-v1 actually needs.

The graphic on the next page summarises the full draw timeline:

```
   host                              │     device
                                     │
   vp_draw_vbo                        │
     ├─ eligibility check             │
     ├─ (indexed) upload index buf    │
     │                                │
     └─ vp_raster_draw                │    ── one vx_enqueue_draw batch (OP_DRAW) ──
         ├─ build DrawCommands batch  │
         ├─ program RASTER/OM/TEX DCRs│ ──► FF config + FS launch descriptor (FRAG_PC)
         ├─ vx_enqueue_draw  ───────► │ ──► CP expands the draw device-side:
         │                            │       expand_k  (VS assembly)     → setup_vertex_t
         │                            │       setup_k   (clip+cull+setup) → rast_prim_t
         │                            │       binning_k (sort-middle)     → primbuf + headers
         │                            │       RASTER walker→earlyZ→packer→dispatch
         │                            │         └ LAUNCH 1-warp frag CTA per wave (pure-DCR)
         │                            │            FS wrapper (run-once):
         │                            │              frag = vx_frag_load()        (GETWS)
         │                            │              recompute edges; interpolate
         │                            │              fs_main(in, out, texstate)   (vx_tex4 | sw)
         │                            │              vx_om4(pos_mask|face, base)   (OM | sw)
         │                            │                        └─► OM AXI master writes cbuf/zbuf
         └─ vx_queue_finish           │    colour/depth stay resident (present = only egress)
                                      │
```

---

## 4. Cross-references in this repository

- The Vortex SDK headers the translator and runtime use:
  `sw/runtime/include/vortex2.h`, `sw/runtime/include/graphics.h`,
  `sw/kernel/include/vx_graphics.h`.
- The shared on-wire graphics types (`fixed_t`, `vec2e_t`, `vec3e_t`,
  `rast_*_t`) live in `sw/common/`; how those buffers are pinned under VM,
  and the TEX/RASTER/OM hardware they feed, is documented in
  [`graphics_hardware_stack.md`](graphics_hardware_stack.md).
- Generated CSR / DCR numbers come from `VX_types.toml` →
  `sw/VX_types.h` + `hw/VX_types.vh`.
- The build artefacts the launcher consumes ship from `libvortex2.a`
  (`sw/kernel/`) and `libvortex.so` (`sw/runtime/`); see
  `Makefile.in` for the install layout (`$VORTEX_PATH/kernel/`,
  `$VORTEX_PATH/runtime/`).
- The hand-written test kernels that exercise the same TEX / RASTER /
  OM hardware mesa drives end up using: `tests/regression/gfx_tex`,
  `gfx_raster`, `gfx_om`, `gfx_draw3d`.

---

## 5. Design invariants & conformance model

These are the load-bearing policies the shipped driver embodies (the
fallback contract in §1.5 *is* the conformance model below). They were
established by `vulkan_support_proposal.md`, which this document now
supersedes.

### 5.1 Invariants

1. **Graphics fixed-function hardware is exactly RASTER, TEX, and OM.**
   Everything else — vertex/fragment/compute shading, binning glue, and
   (historically) ray tracing — runs on the SIMT cores. There is no
   general-purpose "graphics" co-processor beyond those three units.
   *(See the RTU reconciliation in §6.3 — this invariant has since been
   relaxed for ray tracing.)*
2. **The R/T/O datapaths are fixed-point (gfx-v1).** Floating-point work
   runs on the SIMT cores; native FP inside the fixed-function units is a
   gfx-v2 item (§6 of [`graphics_hardware_stack.md`](graphics_hardware_stack.md)).
3. **The driver targets SimX-modeled / synthesizable hardware** — there is
   no separate software-only graphics path; the fallback is llvmpipe CPU
   execution (§1.5), not a divergent Vortex path.

### 5.2 Conformance model — inherit and accelerate

vortexpipe inherits lavapipe's full Vulkan surface and **accelerates a
subset** onto Vortex, falling back to lavapipe CPU execution for anything
not yet offloaded (§1.5). lavapipe is therefore both the unimplemented-
feature fallback **and** the correctness oracle: any Vortex-accelerated
result must match what lavapipe would have produced. The practical
commitment target is Vulkan 1.3 + the ray-tracing extension family, while
the advertised surface remains lavapipe's (currently 1.4). The
silent-collapse audits in §3.6–§3.8 exist precisely to keep "accelerated"
from quietly meaning "wrong" — a gated fallback (§1.5.1) is always
preferable to a unit silently producing a non-conformant result.

---

## 6. Design history and open directions

### 6.1 Rejected compiler shapes (why Shape C)

The Shape C scalar `NIR→LLVM-IR→RISC-V` translator (§2) was chosen after
two alternatives were spiked and rejected:

- **Shape A — fork llvmpipe's SoA codegen** (~238 KB of vectorized
  codegen): rejected as too large to own and maintain.
- **Shape B — SPIR-V round-trip** (`NIR→SPIR-V→llvm-spirv→LLVM`): rejected
  because `llvm-spirv` rejects Vulkan-flavored SPIR-V.

These are recorded so the alternatives are not re-litigated.

### 6.2 HW-unit acceleration roadmap

The forward roadmap of fixed-function enhancements for Vulkan-class
workloads (Hi-Z / early-Z, quad-rate `vx_tex4`/`vx_om4`, MRT, MSAA,
compressed formats, anisotropic filtering, bindless, deeper queues) is
**not implemented** — the units are still gfx-v1. It is tracked in the
"Proposed but not yet implemented" section of
[`graphics_hardware_stack.md`](graphics_hardware_stack.md).

### 6.3 Ray tracing — the RTU path

Ray tracing runs on the **PRISM RTU** (a fixed-function hardware ray-tracing
unit), not the original SIMT-compute traversal. The driver RT path:

- **Lowering.** `vp_nir_lower_ray_tracing_to_rtu.c` lowers Vulkan `rayQueryEXT`
  (`rq_*`) opcodes to the RTU **ISA-v2 window ops** (`TRACE2`/`WAIT2`/`GETW`/
  `CB_RET`, CUSTOM1 funct3=6/7). It runs at NIR-finalize for **any** stage
  (`vp_screen.c`), so a ray query is compilable in a fragment shader as well as
  compute — though the fragment "fusion" case is not yet proven (below).
- **Acceleration structure.** `vp_transcode_as` transcodes the Vulkan AS to the
  RTU's **CW-BVH4** layout (the host builder is the SDK `vortex::raytrace` lib),
  and the `VP_DESC_AS` relocation copies it resident. The RTU consumes that byte
  format directly.
- **Dispatch.** RT rides the compute path (`vp_launch_grid` → `vp_launch`), not
  the CP `OP_DRAW` batch — there is no `OP_TRACE`/`OP_DISPATCH` yet.

Current gaps (tracked in the master plan): the BVH is **rebuilt every dispatch**
(no AS residency) and RT/compute modules are re-loaded per dispatch (no module
residency — compute shares the FS load slot); **ray-query-in-fragment-shader
fusion** is blocked by the shared 32-slot window (the gfx frag payload overlaps
the RTU object-ray/hit slots); and `rtquery` still **silently falls back to
llvmpipe** for the AS-build shaders (STRICT=0). Native `tests/raytracing/
rtu_smoke_*` validate the RTU directly on-device (25/25 simx, 19/19 rtlsim).

The RTU hardware/ISA/ABI microarchitecture is documented in
[`ray_tracing_unit.md`](ray_tracing_unit.md). Invariant 5.1.1 ("no RT hardware
unit") is relaxed for ray tracing.

---

## 7. Source

This document now also subsumes `vulkan_support_proposal.md` (the Vulkan-
on-Vortex strategy, conformance model, and design invariants), which has
been removed from `docs/proposals/`. The hardware-unit improvement
roadmap it proposed is preserved in
[`graphics_hardware_stack.md`](graphics_hardware_stack.md).
