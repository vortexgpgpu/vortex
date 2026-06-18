// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Vortex public host-side graphics API. Triangle setup + tile binning
// for the RASTER hardware unit, plus DCR address helpers. Self-contained:
// the only external dependency is <vx_graphics.h> (canonical on-wire
// ABI) and <VX_types.h> (generated constants).
//
// Consumers: mesa-vortex (vortexpipe Gallium driver), regression tests,
// and any downstream tool that programs the RASTER unit. Implementation
// ships in libvortex.so.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <vx_graphics.h>
#include <VX_types.h>
#include <vortex2.h>

namespace vortex {
namespace graphics {

///////////////////////////////////////////////////////////////////////////////
// DCR address → state-index helpers
//
// VX_types.toml only emits scalar `#define`s, so function-like helpers
// must be declared here rather than in VX_types.h.
///////////////////////////////////////////////////////////////////////////////

#ifndef VX_DCR_TEX_STATE
#define VX_DCR_TEX_STATE(addr)    ((addr) - VX_DCR_TEX_STATE_BEGIN)
#endif
#ifndef VX_DCR_RASTER_STATE
#define VX_DCR_RASTER_STATE(addr) ((addr) - VX_DCR_RASTER_STATE_BEGIN)
#endif
#ifndef VX_DCR_OM_STATE
#define VX_DCR_OM_STATE(addr)     ((addr) - VX_DCR_OM_STATE_BEGIN)
#endif
#ifndef VX_DCR_TEX_MIPOFF
#define VX_DCR_TEX_MIPOFF(lod)    (VX_DCR_TEX_MIPOFF_BASE + (lod))
#endif

///////////////////////////////////////////////////////////////////////////////
// Triangle-setup + tile-binning input types
///////////////////////////////////////////////////////////////////////////////

// One vertex in clip space, plus RASTER-interpolable attributes.
struct vertex_t {
  float pos[4];        // clip-space x, y, z, w
  float color[4];      // r, g, b, a in [0, 1]
  float texcoord[2];   // u, v
};

// One triangle, by vertex index into the caller's vertex container.
struct primitive_t {
  uint32_t i0, i1, i2;
};

///////////////////////////////////////////////////////////////////////////////
// Binning: triangle setup + tile assignment. HOST COVERAGE REFERENCE.
//
// Produces two device-ready buffers:
//   - primbuf: contiguous array of rast_prim_t (edge equations + attribute
//     deltas in Q15.16 / Q23.8 fixed-point form the RASTER unit reads).
//   - tilebuf: the gfx_v2 §6.3 coarse-bin layout — a dense rast_bin_header_t
//     block (each header's pids_offset an ABSOLUTE index into the pid array)
//     followed by the sorted primitive-ID array.
//
// This is a COVERAGE REFERENCE, NOT a bit-exact model of the on-device front
// end: it back-face *flips* winding rather than culling, and does not apply the
// device front end's SETUP_CULL_* modes or its near-plane sub-triangle clip
// (sw/gfx/pipe_frontend.h `pipe_clip_and_setup`). For front-facing,
// within-near-plane geometry it agrees with the device bit-for-bit — the
// gfx_pipeline_* cross-checks rely on that — but for culled / near-clipped
// scenes the on-device front end is authoritative.
//
// Returns the number of tiles produced (>= 0). Width / height are the
// render-target dimensions in pixels. near / far are the depth-range
// extents in [0, 1]. tileLogSize is log2 of the RASTER bin size; must match
// VX_CFG_RASTER_BIN_LOGSIZE (gfx_v2 coarse bins, typically 7 → 128px).
///////////////////////////////////////////////////////////////////////////////

uint32_t Binning(std::vector<uint8_t>& tilebuf,
                 std::vector<uint8_t>& primbuf,
                 const std::unordered_map<uint32_t, vertex_t>& vertices,
                 const std::vector<primitive_t>& primitives,
                 uint32_t width,
                 uint32_t height,
                 float near,
                 float far,
                 uint32_t tileLogSize);

///////////////////////////////////////////////////////////////////////////////
// DrawCommands — host-side builder for a self-contained device draw / pass.
//
// Accumulate the ordered kernel launches and fixed-function (DCR) config
// writes, then submit() the whole sequence as a single CP ring batch
// (vx_enqueue_commands): one doorbell, one completion event, the host
// untouched between stages. The CP's launch-drain serialization provides the
// device-wide inter-stage barrier the sort-middle front end needs (setup →
// binning → raster). This is the charter §6.4 graphics command-sequence
// builder; vortexpipe and the regression tests emit a draw through it instead
// of issuing per-stage launches with a host round-trip each.
//
// The builder copies each launch's argument blob and owns the launch
// descriptors, so the only valid arg-pointer lifetime callers must respect is
// the duration of the launch() call itself. Reusable across draws: build once,
// submit() per pass.
///////////////////////////////////////////////////////////////////////////////

class DrawCommands {
public:
  // Append a kernel launch. `args` / `args_size` is the kernel-argument blob
  // (copied immediately); pass nullptr / 0 for the legacy escape hatch (caller
  // pre-programmed the ARG DCRs). grid / block are `ndim`-element arrays
  // (ndim in 1..3). Returns *this for chaining.
  DrawCommands& launch(vx_kernel_h kernel,
                       const void* args, size_t args_size,
                       uint32_t ndim,
                       const uint32_t grid[3], const uint32_t block[3]);

  // Append one fixed-function / device-config-register write.
  DrawCommands& dcr_write(uint32_t addr, uint32_t value);

  // Number of commands accumulated so far.
  size_t size() const { return entries_.size(); }

  // Drop all accumulated commands (reuse the builder for the next pass).
  void clear() { entries_.clear(); }

  // Encode the accumulated sequence and submit it as one CP batch on `q`.
  // Wait-list + single completion event semantics match vx_enqueue_commands.
  vx_result_t submit(vx_queue_h q,
                     uint32_t n_wait_events, const vx_event_h* wait_events,
                     vx_event_h* out_event);

private:
  struct Entry {
    bool        is_launch = false;
    uint32_t    dcr_addr  = 0;        // VX_COMMAND_DCR_WRITE
    uint32_t    dcr_value = 0;
    vx_kernel_h kernel    = nullptr;  // VX_COMMAND_LAUNCH
    uint32_t    ndim      = 0;
    uint32_t    grid[3]   = {1, 1, 1};
    uint32_t    block[3]  = {1, 1, 1};
    std::vector<uint8_t> args;
  };
  std::vector<Entry> entries_;
};

///////////////////////////////////////////////////////////////////////////////
// FrontEndPool — the on-device setup+binning working set + launch emitter.
//
// Owns the device-resident scratch + output buffers the nine-stage front end
// reads/writes (the charter §4.1 tiling pool), allocated once sized to a
// triangle / key high-water mark. Per draw, append() emits the front end's
// nine launches into a DrawCommands, reading the runtime counts from pipe_arg_t
// so the launch dims stay static. The pool is OVERWRITTEN by each draw and so
// is reused across a frame's draws (reset-by-reuse): the prior draw's RASTER
// pass has fully drained before the next draw's front end runs (CP launch-drain
// serialization), so one pool backs an entire pass. primbuf/tilebuf are the
// pinned outputs the RASTER unit consumes; num_bins() is the host-static tile
// count RASTER walks.
//
// This is the §6.1/§6.2 front end as a runtime-provided capability: callers
// (regression tests, vortexpipe) get the front end without re-deriving its
// ~16-buffer layout or stage schedule. The two kernel entries are supplied by
// the caller (app-loaded module) until the front-end kernels ship in the
// kernel library.
///////////////////////////////////////////////////////////////////////////////

class FrontEndPool {
public:
  FrontEndPool() = default;
  ~FrontEndPool();
  FrontEndPool(const FrontEndPool&) = delete;
  FrontEndPool& operator=(const FrontEndPool&) = delete;

  // Allocate the working set on `dev` for up to `max_tris` triangles into a
  // (width x height) target binned at `tile_log` (log2 tile edge in px), with
  // `keys_cap` bounding the per-draw (tile,prim)-pair count. `setup_k` /
  // `binning_k` are the front-end kernel entries. `block_dim` / `grid_dim` are
  // the device-filling launch dims (e.g. from vx_device_max_occupancy_grid).
  // Re-init releases the prior pool first.
  vx_result_t init(vx_device_h dev, vx_kernel_h setup_k, vx_kernel_h binning_k,
                   uint32_t max_tris, uint32_t width, uint32_t height,
                   uint32_t tile_log, uint32_t keys_cap,
                   uint32_t block_dim, uint32_t grid_dim);

  // Append the nine front-end launches for `num_tris` triangles sourced from
  // `verts_addr` (a device setup_vertex_t[3*num_tris]) into `dc`. num_tris must
  // not exceed the max_tris the pool was sized for.
  vx_result_t append(DrawCommands& dc, uint64_t verts_addr, uint32_t num_tris) const;

  uint64_t primbuf_addr() const;   // dense rast_prim_t output (pinned)
  uint64_t tilebuf_addr() const;   // dense tile headers + sorted pids (pinned)
  uint32_t num_bins()     const;   // dense tile-grid count == RASTER tile_count
  uint32_t bin_cols()     const;

  // Output buffer handles, owned by the pool (do NOT release) — exposed for
  // host readback / inspection; RASTER consumes them via *_addr() above.
  vx_buffer_h prim_buffer() const;
  vx_buffer_h tile_buffer() const;

  void release();

private:
  struct Impl;
  Impl* impl_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////////
// Fixed-function unit state + register emitters (the genxml / si_emit pattern).
//
// Each *_state_t is the canonical host-side configuration for one FF unit — the
// single source of truth a driver or test fills, which program_* translates to
// that unit's VX_DCR_* writes. This centralizes the register packing that was
// otherwise duplicated inline across the vortexpipe driver and every gfx test
// (the radeonsi si_emit_* / Intel genxml *_pack model): libvortex owns the
// hardware register knowledge, callers own only the state. Buffer addresses are
// plain byte device addresses (as returned by vx_buffer_address); the emitter
// converts to the 64-byte block index each DCR encodes.
//
// Two emit targets share one sequence:
//   - program_*(vx_queue_h, ...)    immediate: one vx_enqueue_dcr_write per reg.
//   - program_*(DrawCommands&, ...) batched:   appended to a CP command batch.
///////////////////////////////////////////////////////////////////////////////

struct raster_state_t {
  uint64_t tbuf_addr   = 0;   // tile-buffer byte address
  uint32_t tile_count  = 0;
  uint64_t pbuf_addr   = 0;   // primitive-buffer byte address
  uint32_t pbuf_stride = 0;
  uint32_t width       = 0;   // scissor extents (pixels)
  uint32_t height      = 0;
};

struct om_state_t {
  // colour target
  uint64_t cbuf_addr  = 0;    // colour-buffer byte address
  uint32_t cbuf_pitch = 0;
  uint32_t colormask  = 0xF;
  // depth/stencil target — z-buffer registers emitted only when zbuf_addr != 0
  uint64_t zbuf_addr  = 0;    // depth-buffer byte address
  uint32_t zbuf_pitch = 0;
  // depth (defaults disable: ALWAYS-pass, no writes)
  uint32_t depth_func      = VX_OM_DEPTH_FUNC_ALWAYS;
  uint32_t depth_writemask = 0;
  // stencil (defaults disable)
  uint32_t stencil_func      = VX_OM_DEPTH_FUNC_ALWAYS;
  uint32_t stencil_zpass     = VX_OM_STENCIL_OP_KEEP;
  uint32_t stencil_zfail     = VX_OM_STENCIL_OP_KEEP;
  uint32_t stencil_fail      = VX_OM_STENCIL_OP_KEEP;
  uint32_t stencil_ref       = 0;
  uint32_t stencil_mask      = VX_OM_STENCIL_MASK;
  uint32_t stencil_writemask = 0;
  // blend / logic-op (defaults: opaque src-copy, no logic-op)
  uint32_t blend_mode  = (VX_OM_BLEND_MODE_ADD << 16) | VX_OM_BLEND_MODE_ADD;
  uint32_t blend_func  = (VX_OM_BLEND_FUNC_ZERO << 24) | (VX_OM_BLEND_FUNC_ZERO << 16)
                       | (VX_OM_BLEND_FUNC_ONE  << 8)  |  VX_OM_BLEND_FUNC_ONE;
  uint32_t blend_const = 0;
  uint32_t logic_op    = 0;
};

struct tex_state_t {
  uint32_t        stage       = 0;
  uint64_t        addr        = 0;    // texture byte address
  uint32_t        logwidth    = 0;
  uint32_t        logheight   = 0;
  uint32_t        format      = 0;    // VX_TEX_FORMAT_*
  uint32_t        filter      = VX_TEX_FILTER_POINT;
  uint32_t        wrap_u      = 0;
  uint32_t        wrap_v      = 0;
  const uint32_t* mip_offsets = nullptr;  // num_mips entries; null => only mip 0 at base
  uint32_t        num_mips    = 0;
};

// Immediate emit (one vx_enqueue_dcr_write per register). Returns the first
// non-VX_SUCCESS status, or VX_SUCCESS.
vx_result_t program_raster(vx_queue_h q, const raster_state_t& s);
vx_result_t program_om    (vx_queue_h q, const om_state_t& s);
vx_result_t program_tex   (vx_queue_h q, const tex_state_t& s);

// Batched emit (append to a CP command batch).
void program_raster(DrawCommands& dc, const raster_state_t& s);
void program_om    (DrawCommands& dc, const om_state_t& s);
void program_tex   (DrawCommands& dc, const tex_state_t& s);

///////////////////////////////////////////////////////////////////////////////
// draw — thin "flush + dispatch" for the simple single-pass consumer.
//
// Programs the FF units from the supplied state, then dispatches the
// fragment-shader kernel. The si_draw_vbo / nvk_CmdDraw composition point: the
// caller owns buffer allocation/upload and the (kernel-ABI-specific) argument
// blob; richer consumers call program_* directly and dispatch themselves. tex
// == nullptr leaves the TEX unit unprogrammed (untextured draw). grid / block
// are ndim-element arrays (ndim in 1..3); a null/short array defaults to 1.
///////////////////////////////////////////////////////////////////////////////

vx_result_t draw(vx_queue_h q, vx_kernel_h fs_kernel,
                 const raster_state_t& raster,
                 const om_state_t& om,
                 const tex_state_t* tex,
                 const void* args, size_t args_size,
                 uint32_t ndim, const uint32_t grid[3], const uint32_t block[3]);

} // namespace graphics
} // namespace vortex
