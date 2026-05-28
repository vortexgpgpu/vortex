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
ISA mnemonics (`vx_rast`, `vx_om`, `vx_tex`, `vx_barrier`,
`vx_rast_begin`) and CSR numbers come from
`sw/kernel/include/vx_graphics.h` + `sw/VX_types.h` in this repo.

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
| fragment  | `void fs_main(ptr %in, ptr %out)` wrapped by an emitted `kernel_main` that runs the raster poll-loop (`emit_fs_wrapper`, [`vp_nir_to_llvm.c:1393`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L1393)) | the wrapper's `kernel_main` |

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
  CSR (CTA thread / block IDs, RASTER barycentrics, tmask, etc.).
- `emit_vx_barrier(t)` — `custom-0 funct3=4` with the CTA id as
  barrier id and the CTA's warp count as the count, matching
  `vx_spawn2.h::__syncthreads()`.
- `emit_vx_tex(t, u, v, lod)` — `custom-1 funct3=1 R4-type funct2=0`
  (stage 0). Returns the filtered texel as a packed `A8R8G8B8` i32.
- `emit_vx_rast(t)` — `custom-1 funct3=3 R-type`. Pops a quad from
  the cluster raster_core; returns `pos_mask` (0 means drained).
- `emit_vx_rast_begin(t)` — `custom-1 funct3=4 R-type`. Per-frame
  trigger that tells the raster_core to fetch its tile/prim
  buffers from the currently-programmed DCRs. Idempotent in hardware;
  the FS wrapper emits it exactly once before the poll loop.
- `emit_vx_om(t, pos_face, color, depth)` — `custom-1 funct3=2
  R4-type`. Submits one shaded fragment to the OM unit, which does
  depth-stencil + blend and writes the colour/depth buffers via its
  own AXI master.

### 2.3 How the compiler **detects and selects** Vortex graphics ISA

There is no per-instruction "should I use TEX?" decision in the
translator — the selection happens at three earlier and clearer
points.

#### 2.3.1 Selection by shader stage

The translator routes once on `nir->info.stage`
([`vp_nir_to_llvm.c:1547-1548,
1583`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L1547)):
compute and vertex stages become a plain `kernel_main(ptr %arg)`;
fragment becomes `fs_main(ptr %in, ptr %out)` and an emitted
`kernel_main` wrapper runs the **raster poll-loop**. So `vx_rast`,
`vx_rast_begin`, and `vx_om` are emitted only on the fragment path,
inside `emit_fs_wrapper` — they are wired up because the stage is FS,
not because a NIR opcode asked for them. Every fragment shader uses
the hardware raster: that is the only fragment-stage code path
vortexpipe knows how to emit.

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
- `nir_tex_instr` with `op == nir_texop_tex` → `emit_vx_tex` after
  converting the float UVs to S.23 fixed-point
  ([`vp_nir_to_llvm.c:907-944`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L907)).
  No other tex op is currently lowered; anything else falls back.

If a NIR opcode has no mapping, the translator sets `t->ok = false`
and the whole shader fails translation (`vp_nir_to_llvm` returns
`false`), at which point the consumer call site
(`vp_create_compute_state` / `_vs_state` / `_fs_state`) keeps the
llvmpipe CSO around without a `vxbin` and the per-call fallback at
`launch_grid` / `draw_vbo` kicks in.

#### 2.3.3 Selection by device capability — the runtime ISA gate

The translator can emit a graphics-ISA-using shader, but the *runtime*
decides whether to actually dispatch it on the hardware raster path.
That gate sits in `vp_draw_vbo`:

```c
struct vp_screen *vps = vp_reg_get(pipe->screen);
bool gfx_hw = vps && vps->has_raster && vps->has_om;
bool tex_needed = vp->cur_tex != NULL;
if (gfx_hw && tex_needed && !vps->has_tex) {
    mesa_logw("...device lacks TEX extension; ...skipping hardware "
              "RASTER+OM path");
    gfx_hw = false;
}
```
([`vp_context.c:967-975`](../../src/gallium/drivers/vortexpipe/vp_context.c#L967))

`has_raster` / `has_om` / `has_tex` are the cached
`VX_ISA_EXT_RASTER` / `_OM` / `_TEX` bits from the device's
`VX_CAPS_ISA_FLAGS`. A device built without these extensions takes
the **VS-on-Vortex / raster-on-llvmpipe** fallback path
(`vp_draw_passthrough`) so the kernel never executes a `vx_rast` /
`vx_om` / `vx_tex` that would trap as an illegal instruction.

A second env-var knob `$VORTEXPIPE_SW_RASTER` forces that same
fallback even on a fully-capable device, useful for A/B'ing the
hardware raster against the llvmpipe one.

#### 2.3.4 Selection by encoding constants

Vortex's graphics ISA uses the **RISC-V custom-1 opcode** (43 decimal
= 0x2B). `vp_nir_to_llvm` emits the instructions through LLVM inline
asm with `.insn r 43, funct3, …` / `.insn r4 43, funct3, …`
templates. The `funct3` values used are:

| `funct3` | Form     | Mnemonic        | What it does                                |
|----------|----------|-----------------|---------------------------------------------|
| 1        | R4       | `vx_tex`        | sample TEX stage 0 (`stage = funct2`)       |
| 2        | R4       | `vx_om`         | submit a fragment to OM                     |
| 3        | R        | `vx_rast`       | pop a quad from the cluster raster_core     |
| 4        | R        | `vx_rast_begin` | per-frame raster fetch trigger              |

The `funct3` numbering here matches the Vortex kernel SDK
(`sw/kernel/include/vx_graphics.h` in this repo) so that hand-written
test kernels (e.g. `tests/regression/gfx_draw3d`) and the translated
mesa shaders produce byte-identical encodings.

`vx_barrier` is on custom-0 (opcode 11, `VP_RISCV_CUSTOM0` in
[`vp_nir_to_llvm.c:52`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L52)),
since custom-1 is reserved for graphics.

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

vortexpipe only takes the Vortex path for a **simple direct,
non-indexed, non-instanced single draw**:

```c
bool simple = vp->dev && vs && vs->vxbin && vs->vs_layout.stride &&
              !indirect && num_draws == 1 && info->index_size == 0 &&
              !info->primitive_restart && info->instance_count == 1 &&
              draws[0].count > 0;
```
([`vp_context.c:929-933`](../../src/gallium/drivers/vortexpipe/vp_context.c#L929))

Anything else (indexed, instanced, multi-draw, indirect, prim
restart, no VS, untranslatable VS) takes the wholesale llvmpipe
fallback (or fails loudly in STRICT mode).

### 3.2 Stage 1 — vertex shader on Vortex (`vp_launch_vs`)

For a simple draw:

1. Allocate a host buffer of `count × vs_layout.stride` bytes for the
   transformed-vertex records. Slot 0 of each record is the clip-
   space `gl_Position` (vec4); slots 1.. are the generic varyings
   declared by the VS, padded to vec4 (`stride` = `16 × (1 +
   num_varyings)`).
2. If the VS reads vertex-buffer inputs, gather the bound vertex
   buffer + per-attribute offsets/strides into a `vp_vertex_input`
   record (`vp_gather_vertex_input`).
3. Build a `.vxbin` argument block:
   - Slot 0 = device address of the output vertex-record buffer.
   - Slot 1 = device address of the attribute table (or 0 for a
     self-contained VS).
4. Launch the VS kernel as one CTA of `vertex_count` threads:
   `grid = {1,1,1}`, `block = {vertex_count, 1, 1}`. Each thread
   reads `gl_VertexIndex` from `VX_CSR_CTA_THREAD_ID_X` (= `t->vid`),
   fetches its inputs, runs the user shader, and writes its output
   record to `out_base + vid × stride`.
5. Read the output buffer back into host memory.

### 3.3 Stage 2 — binning (host CPU, but on Vortex install path)

`vp_raster_draw`
([`vp_raster.cpp`](../../src/gallium/drivers/vortexpipe/vp_raster.cpp))
takes over once the transformed vertices are back on the host:

1. Build `graphics::vertex_t` records from the transformed-vertex
   blob — slot 0 is the clip-space position; each varying is routed
   to either the colour plane (3-/4-component) or the texcoord plane
   (2-component) by component count
   ([`vp_raster.cpp:84-105`](../../src/gallium/drivers/vortexpipe/vp_raster.cpp#L84)).
   This is the gfx-v1 fixed-function varying mapping the FS
   translator also assumes.
2. Build `graphics::primitive_t` triples — gfx-v1 only handles
   triangle lists today.
3. Call `vortex::graphics::Binning(tilebuf, primbuf, verts, prims,
   width, height, 0.0f, 1.0f, RASTER_TILE_LOGSIZE)`. This is shared
   code linked from the Vortex SDK (`graphics.cpp` /
   `sw/runtime/graphics.cpp`): it does triangle setup +
   tile-binning, producing two byte blobs in the on-wire
   `rast_tile_header_t` + `rast_prim_t` layouts the RASTER hardware
   reads directly.

### 3.4 Stage 3 — hardware rasterization + fragment shading

The remainder of `vp_raster_draw` programs the RASTER + OM + (optional)
TEX DCRs and dispatches the fragment shader kernel:

1. **Upload everything**: tile buffer, primitive buffer, the
   (already-cleared) colour attachment, and a freshly-allocated depth
   attachment (cleared to `0x00` for `GREATER`/`GEQUAL` compares,
   `0xFF` for everything else).
2. **Program RASTER DCRs**
   ([`vp_raster.cpp:197-203`](../../src/gallium/drivers/vortexpipe/vp_raster.cpp#L197)):

   ```
   VX_DCR_RASTER_TBUF_ADDR   <- tile_dev   / 64    (64-byte block index)
   VX_DCR_RASTER_TILE_COUNT  <- num_tiles
   VX_DCR_RASTER_PBUF_ADDR   <- prim_dev   / 64
   VX_DCR_RASTER_PBUF_STRIDE <- VP_RAST_PRIM_STRIDE  (120 bytes)
   VX_DCR_RASTER_SCISSOR_X   <- (width  << 16) | 0
   VX_DCR_RASTER_SCISSOR_Y   <- (height << 16) | 0
   ```
3. **Program OM DCRs** ([`vp_raster.cpp:207-226`](../../src/gallium/drivers/vortexpipe/vp_raster.cpp#L207)):
   colour + depth buffer addresses and pitches, depth-compare
   function and write-mask from the bound `vp_dsa_cso`,
   colour-write-mask + blend mode + blend func from the bound
   `vp_blend_cso`. Stencil is hard-disabled (gfx-v1 doesn't ship
   stencil).
4. **Program TEX DCRs** (only if a sampler is bound,
   [`vp_raster.cpp:234-263`](../../src/gallium/drivers/vortexpipe/vp_raster.cpp#L234)):
   convert the Vulkan-side R8G8B8A8 host pixels to the A8R8G8B8 word
   the TEX unit unpacks, upload as mip 0, then program
   `VX_DCR_TEX_{STAGE, LOGDIM, FORMAT, FILTER, WRAP, ADDR,
   MIPOFF_BASE}` for stage 0.
5. **Dispatch the FS kernel across the whole device**:

   ```c
   grid_dim  = { num_cores, 1, 1 };
   block_dim = { num_threads * num_warps, 1, 1 };
   ```
   ([`vp_raster.cpp:274-285`](../../src/gallium/drivers/vortexpipe/vp_raster.cpp#L274)).
   Every warp on every core enters the FS wrapper's
   `vx_rast_begin` → poll-loop, so the whole device races for
   quad pops.

6. **The FS wrapper inside the kernel**
   ([`emit_fs_wrapper`,
   `vp_nir_to_llvm.c:1393-1526`](../../src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c#L1393))
   loops:
   ```
   loop:
       pos_mask = vx_rast()
       if pos_mask == 0: return
       pid       = csrr VX_CSR_RASTER_PID
       prim      = arg[0] + pid * 120
       (qx, qy)  = unpack(pos_mask >> 4)
       mask      = pos_mask & 0xf
       for each covered sub-pixel i in {0,1,2,3}:
           (f0,f1,f2) = csrr VX_CSR_RASTER_BCOORD_{X,Y,Z}0+i (fixed16)
           dx = f0/(f0+f1+f2)
           dy = f1/(f0+f1+f2)
           interpolate(prim.rast_attribs, dx, dy) → fs_in (varyings)
           fs_main(fs_in, fs_out)
           rgba   = pack(fs_out as A8R8G8B8)
           depth  = fixed24(interp(prim.attribs.z, dx, dy))
           pos_face = ((qy << 1 | (i >> 1)) << 16)
                    | ((qx << 1 | (i &  1)) << 1)
           vx_om(pos_face, rgba, depth)
   ```
   `vx_om` triggers the OM unit's AXI master, which depth-tests
   against the depth buffer at the configured PA, blends against the
   current colour pixel, and writes both back. The FS kernel is
   never directly aware of the colour / depth buffer addresses — it
   just feeds the OM.

7. After `vx_queue_finish`, the colour buffer is copied back into
   the framebuffer attachment via `vp_fb_color_write`.

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
     │                                │
     ├─ vp_launch_vs                  │       VS kernel (one CTA, count threads)
     │   ├─ alloc out-buf             │ ──┐
     │   ├─ build arg block           │   │ vx_enqueue_write   (vbuf, attrtab)
     │   ├─ build attr table          │   │
     │   ├─ launch ────────────────── │ ──┘ vx_enqueue_launch  (grid=1, block=count)
     │   └─ read back out-buf  ◄───── │ ◄── vx_enqueue_read    (out)
     │                                │
     ├─ graphics::Binning             │
     │     produces tilebuf + primbuf │
     │                                │
     ├─ vp_raster_draw                │
     │   ├─ vx_enqueue_write   ─────► │ ──► tiles + prims + cbuf + zbuf [+ tex]
     │   ├─ vx_enqueue_dcr_write x N  │ ──► RASTER + OM [+ TEX] DCRs
     │   │                            │
     │   ├─ vx_enqueue_launch  ─────► │ ──► FS kernel (one CTA per core, n_warps*n_threads each)
     │   │                            │       loop:
     │   │                            │         pos_mask = vx_rast()
     │   │                            │         if drained: exit
     │   │                            │         interpolate varyings via CSRs
     │   │                            │         fs_main(in, out)
     │   │                            │         vx_om(pos_face, rgba, depth)  ─► OM AXI master
     │   │                            │                                          writes cbuf/zbuf
     │   ├─ vx_enqueue_read  ◄─────── │ ◄── colour attachment
     │   └─ vx_queue_finish           │
     │                                │
     └─ vp_fb_color_write             │
                                      │
```

---

## 4. Cross-references in this repository

- The Vortex SDK headers the translator and runtime use:
  `sw/runtime/include/vortex2.h`, `sw/runtime/include/graphics.h`,
  `sw/kernel/include/vx_graphics.h`.
- The shared on-wire graphics types (`fixed_t`, `vec2e_t`, `vec3e_t`,
  `rast_*_t`) live in `sw/common/` (see
  [`gfx_vm_pinned_buffers_proposal.md`](../proposals/gfx_vm_pinned_buffers_proposal.md)
  for how those buffers are pinned under VM).
- Generated CSR / DCR numbers come from `VX_types.toml` →
  `sw/VX_types.h` + `hw/VX_types.vh`.
- The build artefacts the launcher consumes ship from `libvortex2.a`
  (`sw/kernel/`) and `libvortex.so` (`sw/runtime/`); see
  `Makefile.in` for the install layout (`$VORTEX_PATH/kernel/`,
  `$VORTEX_PATH/runtime/`).
- The hand-written test kernels that exercise the same TEX / RASTER /
  OM hardware mesa drives end up using: `tests/regression/gfx_tex`,
  `gfx_raster`, `gfx_om`, `gfx_draw3d`.
