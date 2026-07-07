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

// Implementation of vortex::graphics::Binning. Self-contained: uses
// only stdlib + the public graphics.h header. No cocogfx dependency
// (intentional — graphics.h ships in the install tree and downstream
// tools build only against $VORTEX_PATH/runtime/include).

#include <graphics.h>
#include <gfx_frontend_abi.h>   // pipe_arg_t, PIPE_STAGE_*, setup types
#include <gfx_setup.h>          // gfx_setup::setup_triangle (SSOT, shared w/ device setup_k)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace vortex {
namespace graphics {


// The bin-sort key packs the pid into its low PIPE_PRIM_BITS; the RASTER unit
// then addresses that pid with a VX_RASTER_PID_BITS-wide field. The key field
// must be at least as wide as the consumer pid, and the real visible-prim cap is
// the tighter VX_RASTER_PID_BITS (enforced at runtime in Binning / append).
static_assert(PIPE_PRIM_BITS >= VX_RASTER_PID_BITS,
              "bin-sort key pid field (PIPE_PRIM_BITS) must hold a RASTER pid");

///////////////////////////////////////////////////////////////////////////////
// Binning: scan primitives, build per-tile primitive-ID lists, emit the
// device-ready tile-header + primitive-record buffers.
///////////////////////////////////////////////////////////////////////////////

uint32_t Binning(std::vector<uint8_t>& tilebuf,
                 std::vector<uint8_t>& primbuf,
                 const std::unordered_map<uint32_t, vertex_t>& vertices,
                 const std::vector<primitive_t>& primitives,
                 uint32_t width,
                 uint32_t height,
                 float near,
                 float far,
                 uint32_t tileLogSize) {
  std::map<std::pair<uint16_t, uint16_t>, std::vector<uint32_t>> tiles;

  std::vector<rast_prim_t> rast_prims;
  rast_prims.reserve(primitives.size());

  uint32_t total_prims = 0;

  for (auto& primitive : primitives) {
    auto it0 = vertices.find(primitive.i0);
    auto it1 = vertices.find(primitive.i1);
    auto it2 = vertices.find(primitive.i2);
    if (it0 == vertices.end() || it1 == vertices.end() || it2 == vertices.end()) {
      printf("warning: primitive references missing vertex...\n");
      continue;
    }
    const vertex_t& v0 = it0->second;
    const vertex_t& v1 = it1->second;
    const vertex_t& v2 = it2->second;

    // Per-triangle setup via the shared SSOT (gfx_setup::setup_triangle) — the
    // same code the device front end (setup_k) runs, so this host coverage
    // reference stays bit-exact with the device for front-facing, within-near-
    // plane geometry. Two-sided (SETUP_CULL_NONE) and no near-plane sub-triangle
    // clip: Binning is a coverage oracle, not the device's culled/clipped path.
    // setup_vertex_t is byte-identical to vertex_t (gfx_frontend_abi.h).
    auto to_sv = [](const vertex_t& v) {
      setup_vertex_t s;
      s.pos[0] = v.pos[0]; s.pos[1] = v.pos[1]; s.pos[2] = v.pos[2]; s.pos[3] = v.pos[3];
      s.color[0] = v.color[0]; s.color[1] = v.color[1];
      s.color[2] = v.color[2]; s.color[3] = v.color[3];
      s.texcoord[0] = v.texcoord[0]; s.texcoord[1] = v.texcoord[1];
      return s;
    };

    rast_prim_t rast_prim{};
    setup_bbox_t bb{};
    if (!gfx_setup::setup_triangle(to_sv(v0), to_sv(v1), to_sv(v2),
                                   static_cast<int>(width), static_cast<int>(height),
                                   near, far, rast_prim, bb, SETUP_CULL_NONE))
      continue;  // degenerate or empty bbox

    uint32_t p = static_cast<uint32_t>(rast_prims.size());
    rast_prims.push_back(rast_prim);

    // Tile coverage from the setup bbox
    {
      uint32_t tileSize  = 1u << tileLogSize;
      uint32_t minTileX  = bb.bbL >> tileLogSize;
      uint32_t maxTileX  = (bb.bbR + tileSize - 1) >> tileLogSize;
      uint32_t minTileY  = bb.bbT >> tileLogSize;
      uint32_t maxTileY  = (bb.bbB + tileSize - 1) >> tileLogSize;

      for (uint32_t ty = minTileY; ty < maxTileY; ty++) {
        for (uint32_t tx = minTileX; tx < maxTileX; tx++) {
          tiles[{static_cast<uint16_t>(tx), static_cast<uint16_t>(ty)}].push_back(p);
          ++total_prims;
        }
      }
    }
  }

  // pid aliasing guard: the RASTER unit addresses primitives with a
  // VX_RASTER_PID_BITS-wide pid (the rast_prim index). A scene with more visible
  // primitives than that field can hold would silently alias on the device, so
  // reject it loudly here rather than corrupt the frame. (PIPE_PRIM_BITS — the
  // bin-sort key's pid field — is >= this; see the static_assert above.)
  if (rast_prims.size() > (size_t(1) << VX_RASTER_PID_BITS)) {
    printf("error: Binning: %zu visible primitives exceed the %u-bit RASTER pid "
           "field (max %u)\n", rast_prims.size(), (unsigned)VX_RASTER_PID_BITS,
           (1u << VX_RASTER_PID_BITS));
    return 0;
  }

  // Emit primbuf (contiguous rast_prim_t array)
  primbuf.resize(rast_prims.size() * sizeof(rast_prim_t));
  if (!rast_prims.empty())
    std::memcpy(primbuf.data(), rast_prims.data(), primbuf.size());

  // Emit tilebuf in the gfx_v2 §6.3 coarse-bin layout the RASTER front end
  // reads: a dense rast_bin_header_t block followed by the sorted-pid array,
  // each bin's pids_offset an ABSOLUTE index into that array (not relative to
  // its own header). bin_x/bin_y are bin indices — the RASTER core scales them
  // by the bin size (1 << VX_CFG_RASTER_BIN_LOG_SIZE), so the caller must bin at
  // that granularity (pass tileLogSize = VX_CFG_RASTER_BIN_LOG_SIZE).
  const size_t num_bins = tiles.size();
  tilebuf.resize(num_bins * sizeof(rast_bin_header_t)
                 + (size_t)total_prims * sizeof(uint32_t));
  auto* bin_header  = reinterpret_cast<rast_bin_header_t*>(tilebuf.data());
  auto* sorted_pids = reinterpret_cast<uint32_t*>(
      tilebuf.data() + num_bins * sizeof(rast_bin_header_t));
  uint32_t pid_cursor = 0;   // absolute index into sorted_pids
  for (auto& it : tiles) {
    bin_header->bin_x       = it.first.first;
    bin_header->bin_y       = it.first.second;
    bin_header->pids_offset = pid_cursor;
    bin_header->pids_count  = static_cast<uint32_t>(it.second.size());
    ++bin_header;
    std::memcpy(sorted_pids + pid_cursor, it.second.data(),
                it.second.size() * sizeof(uint32_t));
    pid_cursor += static_cast<uint32_t>(it.second.size());
  }

  return static_cast<uint32_t>(num_bins);
}

///////////////////////////////////////////////////////////////////////////////
// DrawCommands — assemble + submit a draw as one CP ring batch (charter §6.4).
///////////////////////////////////////////////////////////////////////////////

DrawCommands& DrawCommands::launch(vx_kernel_h kernel,
                                   const void* args, size_t args_size,
                                   uint32_t ndim,
                                   const uint32_t grid[3],
                                   const uint32_t block[3],
                                   uint32_t lmem_size) {
  Entry e;
  e.is_launch = true;
  e.kernel    = kernel;
  e.ndim      = ndim;
  e.lmem_size = lmem_size;
  for (uint32_t i = 0; i < ndim && i < 3; ++i) {
    if (grid)  e.grid[i]  = grid[i];
    if (block) e.block[i] = block[i];
  }
  if (args && args_size > 0) {
    const uint8_t* p = static_cast<const uint8_t*>(args);
    e.args.assign(p, p + args_size);
  }
  entries_.push_back(std::move(e));
  return *this;
}

DrawCommands& DrawCommands::dcr_write(uint32_t addr, uint32_t value) {
  Entry e;
  e.is_launch  = false;
  e.dcr_addr   = addr;
  e.dcr_value  = value;
  entries_.push_back(std::move(e));
  return *this;
}

vx_result_t DrawCommands::submit(vx_queue_h q, uint32_t nw,
                                 const vx_event_h* w, vx_event_h* out) {
  // Build the launch-info storage first, reserved to the launch count so it
  // never reallocates — the command list takes stable pointers into it.
  // args_host points into each Entry's owned blob, stable for this call.
  std::vector<vx_launch_info_t> linfos;
  linfos.reserve(entries_.size());
  std::vector<vx_command_t> cmds;
  cmds.reserve(entries_.size());

  for (const Entry& e : entries_) {
    vx_command_t c{};
    if (e.is_launch) {
      vx_launch_info_t li{};
      li.struct_size = sizeof(li);
      li.kernel      = e.kernel;
      li.args_host   = e.args.empty() ? nullptr : e.args.data();
      li.args_size   = e.args.size();
      li.ndim        = e.ndim;
      li.lmem_size   = e.lmem_size;
      for (uint32_t i = 0; i < 3; ++i) {
        li.grid_dim [i] = e.grid[i];
        li.block_dim[i] = e.block[i];
      }
      linfos.push_back(li);
      c.type        = VX_COMMAND_LAUNCH;
      c.data.launch = &linfos.back();   // stable — linfos is reserved
    } else {
      c.type           = VX_COMMAND_DCR_WRITE;
      c.data.dcr.addr  = e.dcr_addr;
      c.data.dcr.value = e.dcr_value;
    }
    cmds.push_back(c);
  }

  // One device-orchestrated draw: the CP expands the resident descriptor and
  // sequences VS→setup→bin→FF-config→FS on-device with no host round-trip.
  return vx_enqueue_draw(q, cmds.data(),
                         static_cast<uint32_t>(cmds.size()), nw, w, out);
}

///////////////////////////////////////////////////////////////////////////////
// FrontEndPool — own the §4.1 tiling pool + emit the nine front-end launches.
///////////////////////////////////////////////////////////////////////////////

struct FrontEndPool::Impl {
  vx_device_h dev       = nullptr;
  vx_kernel_h setup_k   = nullptr;
  vx_kernel_h binning_k = nullptr;
  uint32_t    width = 0, height = 0;
  uint32_t    max_tris = 0, block_dim = 0, grid_dim = 0;
  uint32_t    bin_cols = 0, bin_rows = 0, num_bins = 0, bin_stripe = 0;

  std::vector<vx_buffer_h> bufs;   // every allocation, for release()
  vx_buffer_h prim_buf = nullptr;  // == prim address' buffer (readback handle)
  vx_buffer_h tile_buf = nullptr;  // == tilebuf address' buffer (readback handle)

  // Scratch + output addresses (device byte addresses).
  uint64_t slot_prim = 0, slot_bbox = 0, keep = 0, offset = 0, tsum = 0;
  uint64_t prim = 0, bbox = 0, bcount = 0, boffset = 0, keys = 0, btsum = 0;
  uint64_t thist = 0, bincount = 0, binbase = 0, tilebuf = 0, meta = 0;
};

FrontEndPool::~FrontEndPool() { release(); }

void FrontEndPool::release() {
  if (!impl_) return;
  for (vx_buffer_h b : impl_->bufs)
    if (b) vx_buffer_release(b);
  delete impl_;
  impl_ = nullptr;
}

vx_result_t FrontEndPool::init(vx_device_h dev, vx_kernel_h setup_k,
                               vx_kernel_h binning_k, uint32_t max_tris,
                               uint32_t width, uint32_t height, uint32_t tile_log,
                               uint32_t keys_cap, uint32_t block_dim,
                               uint32_t grid_dim) {
  release();
  if (!dev || !setup_k || !binning_k || max_tris == 0 ||
      block_dim == 0 || grid_dim == 0 || keys_cap == 0)
    return VX_ERR_INVALID_VALUE;

  impl_ = new Impl();
  impl_->dev = dev; impl_->setup_k = setup_k; impl_->binning_k = binning_k;
  impl_->width = width; impl_->height = height;
  impl_->max_tris = max_tris; impl_->block_dim = block_dim; impl_->grid_dim = grid_dim;

  const uint32_t ts = 1u << tile_log;
  impl_->bin_cols  = (width  + ts - 1) / ts;
  impl_->bin_rows  = (height + ts - 1) / ts;
  impl_->num_bins  = impl_->bin_cols * impl_->bin_rows;
  impl_->bin_stripe = (impl_->num_bins + grid_dim - 1) / grid_dim;

  const uint64_t MS      = SETUP_MAX_SUB;
  const uint64_t PRIM_SZ = sizeof(rast_prim_t);
  const uint64_t HDR_SZ  = sizeof(rast_bin_header_t);  // gfx_v2 coarse-bin header (§6.3)
  const uint64_t BBOX_SZ = sizeof(setup_bbox_t);
  const uint64_t T = block_dim, B = impl_->num_bins, NT = max_tris;

  // Allocate one buffer; record it for release and return its device address.
  auto mk = [&](uint64_t bytes, uint32_t flags, uint64_t* out_addr) -> vx_result_t {
    vx_buffer_h b = nullptr;
    auto r = vx_buffer_create(dev, bytes, flags, &b);
    if (r != VX_SUCCESS) return r;
    impl_->bufs.push_back(b);
    return vx_buffer_address(b, out_addr);
  };
  const uint32_t W = VX_MEM_WRITE;                              // host-write-intent scratch
  const uint32_t RWP = VX_MEM_READ | VX_MEM_WRITE | VX_MEM_PHYS; // pinned FF-read outputs

  vx_result_t r;
  // No slot_prim/slot_bbox scratch: setup_k counts survivors and EMIT recomputes
  // clip+setup straight into the dense primbuf, so the 120-byte rast_prim_t is
  // never staged to a per-triangle slot (slot_*_addr stay 0 in pipe_arg_t).

  // Two-heap residency split (§5.5): the device-only shader scratch regions live
  // in ONE pooled slab (collapsing 12 separate allocations into a single resident
  // buffer), while the FF-pinned-PA outputs the raster unit reads (prim, tilebuf)
  // keep their own VX_MEM_PHYS buffers — both because they need the pinned heap and
  // because callers read them back at offset 0 (prim_buffer()/tile_buffer()).
  constexpr uint64_t SLAB_ALIGN = 64;  // sub-region alignment (cache line)
  auto align_up = [](uint64_t v, uint64_t a) { return (v + a - 1) / a * a; };

  struct Region { uint64_t* addr; uint64_t bytes; };
  const Region scratch[] = {
    { &impl_->keep,     NT * 4 },
    { &impl_->offset,   (NT + 1) * 4 },
    { &impl_->tsum,     T * 4 },
    { &impl_->bbox,     NT * MS * BBOX_SZ },
    { &impl_->bcount,   NT * MS * 4 },
    { &impl_->boffset,  (NT * MS + 1) * 4 },
    { &impl_->keys,     (uint64_t)keys_cap * 4 },
    { &impl_->btsum,    T * 4 },
    { &impl_->thist,    T * B * 4 },
    { &impl_->bincount, B * 4 },
    { &impl_->binbase,  B * 4 },
    { &impl_->meta,     3 * 4 },
  };
  uint64_t slab_bytes = 0;
  for (const auto& rg : scratch) slab_bytes = align_up(slab_bytes, SLAB_ALIGN) + rg.bytes;

  uint64_t slab_base = 0;
  if ((r = mk(slab_bytes, W, &slab_base)) != VX_SUCCESS) return r;
  uint64_t off = 0;
  for (const auto& rg : scratch) {
    off = align_up(off, SLAB_ALIGN);
    *rg.addr = slab_base + off;
    off += rg.bytes;
  }

  // FF-pinned-PA outputs (read by the raster unit; read back by the host at off 0).
  if ((r = mk(NT * MS * PRIM_SZ, RWP, &impl_->prim)) != VX_SUCCESS) return r;
  impl_->prim_buf = impl_->bufs.back();
  if ((r = mk(B * HDR_SZ + (uint64_t)keys_cap * 4, RWP, &impl_->tilebuf)) != VX_SUCCESS) return r;
  impl_->tile_buf = impl_->bufs.back();
  return VX_SUCCESS;
}

vx_result_t FrontEndPool::append(DrawCommands& dc, uint64_t verts_addr,
                                 uint32_t num_tris) const {
  if (!impl_ || num_tris > impl_->max_tris) return VX_ERR_INVALID_VALUE;
  // pid aliasing guard (see Binning): the RASTER pid field is VX_RASTER_PID_BITS
  // wide, so more primitives than it can address would silently alias.
  if (num_tris > (uint32_t(1) << VX_RASTER_PID_BITS)) return VX_ERR_INVALID_VALUE;

  // One pipe_arg_t, mutated per stage. DrawCommands::launch copies the blob on
  // each call, so the nine launches each capture their own stage value.
  pipe_arg_t pa{};
  pa.num_tris   = num_tris;
  pa.width      = impl_->width;
  pa.height     = impl_->height;
  pa.bin_stripe = impl_->bin_stripe;
  pa.bin_cols   = impl_->bin_cols;
  pa.num_bins   = impl_->num_bins;
  pa.verts_addr     = verts_addr;
  pa.slot_prim_addr = impl_->slot_prim;
  pa.slot_bbox_addr = impl_->slot_bbox;
  pa.keep_addr      = impl_->keep;
  pa.offset_addr    = impl_->offset;
  pa.tsum_addr      = impl_->tsum;
  pa.prim_addr      = impl_->prim;
  pa.bbox_addr      = impl_->bbox;
  pa.bcount_addr    = impl_->bcount;
  pa.boffset_addr   = impl_->boffset;
  pa.keys_addr      = impl_->keys;
  pa.btsum_addr     = impl_->btsum;
  pa.thist_addr     = impl_->thist;
  pa.bincount_addr  = impl_->bincount;
  pa.binbase_addr   = impl_->binbase;
  pa.tilebuf_addr   = impl_->tilebuf;
  pa.meta_addr      = impl_->meta;

  const uint32_t block[3] = { impl_->block_dim, 1, 1 };
  for (uint32_t s = 0; s < 9; ++s) {
    pa.stage = s;
    // SCAN / BSCAN / BBASE are single-CTA reductions; the rest device-fill.
    const bool single = (s == PIPE_STAGE_SCAN || s == PIPE_STAGE_BSCAN ||
                         s == PIPE_STAGE_BBASE);
    const uint32_t grid[3] = { single ? 1u : impl_->grid_dim, 1, 1 };
    vx_kernel_h k = (s < PIPE_STAGE_BCOUNT) ? impl_->setup_k : impl_->binning_k;
    dc.launch(k, &pa, sizeof(pa), 1, grid, block);
  }
  return VX_SUCCESS;
}

uint64_t FrontEndPool::primbuf_addr() const { return impl_ ? impl_->prim : 0; }
uint64_t FrontEndPool::tilebuf_addr() const { return impl_ ? impl_->tilebuf : 0; }
uint32_t FrontEndPool::num_bins()    const { return impl_ ? impl_->num_bins : 0; }
uint32_t FrontEndPool::bin_cols()    const { return impl_ ? impl_->bin_cols : 0; }
vx_buffer_h FrontEndPool::prim_buffer() const { return impl_ ? impl_->prim_buf : nullptr; }
vx_buffer_h FrontEndPool::tile_buffer() const { return impl_ ? impl_->tile_buf : nullptr; }

///////////////////////////////////////////////////////////////////////////////
// Fixed-function register emitters (genxml / si_emit pattern). Each emit_*
// drives one DCR sequence through a sink W invoked as w(addr, value); the
// immediate (queue) and batched (DrawCommands) public forms below share it, so
// the register layout for a unit lives in exactly one place.
///////////////////////////////////////////////////////////////////////////////

namespace {

// 64-byte block index the *_ADDR DCRs encode.
inline uint32_t block_addr(uint64_t byte_addr) {
  return static_cast<uint32_t>(byte_addr / 64);
}

template <class W>
void emit_raster(const raster_state_t& s, W&& w) {
  w(VX_DCR_RASTER_TBUF_ADDR,   block_addr(s.tbuf_addr));
  w(VX_DCR_RASTER_TILE_COUNT,  s.tile_count);
  w(VX_DCR_RASTER_PBUF_ADDR,   block_addr(s.pbuf_addr));
  w(VX_DCR_RASTER_PBUF_STRIDE, s.pbuf_stride);
  w(VX_DCR_RASTER_SCISSOR_X,   (s.width  << 16) | 0u);
  w(VX_DCR_RASTER_SCISSOR_Y,   (s.height << 16) | 0u);
  // Fragment-shader dispatch descriptor: the raster work distributor launches the FS
  // at this entry with this arg, on-device (true-GPU pixel dispatch).
  w(VX_DCR_RASTER_FRAG_ENTRY_LO, static_cast<uint32_t>(s.frag_entry));
  w(VX_DCR_RASTER_FRAG_ENTRY_HI, static_cast<uint32_t>(s.frag_entry >> 32));
  w(VX_DCR_RASTER_FRAG_PARAM_LO, static_cast<uint32_t>(s.frag_param));
  w(VX_DCR_RASTER_FRAG_PARAM_HI, static_cast<uint32_t>(s.frag_param >> 32));
}

template <class W>
void emit_om(const om_state_t& s, W&& w) {
  w(VX_DCR_OM_CBUF_ADDR,      block_addr(s.cbuf_addr));
  w(VX_DCR_OM_CBUF_PITCH,     s.cbuf_pitch);
  w(VX_DCR_OM_CBUF_WRITEMASK, s.colormask);
  if (s.zbuf_addr != 0) {
    w(VX_DCR_OM_ZBUF_ADDR,  block_addr(s.zbuf_addr));
    w(VX_DCR_OM_ZBUF_PITCH, s.zbuf_pitch);
  }
  w(VX_DCR_OM_DEPTH_FUNC,        s.depth_func);
  w(VX_DCR_OM_DEPTH_WRITEMASK,   s.depth_writemask);
  w(VX_DCR_OM_EARLYZ_SAFE,       s.earlyz_safe);
  w(VX_DCR_OM_STENCIL_FUNC,      s.stencil_func);
  w(VX_DCR_OM_STENCIL_ZPASS,     s.stencil_zpass);
  w(VX_DCR_OM_STENCIL_ZFAIL,     s.stencil_zfail);
  w(VX_DCR_OM_STENCIL_FAIL,      s.stencil_fail);
  w(VX_DCR_OM_STENCIL_REF,       s.stencil_ref);
  w(VX_DCR_OM_STENCIL_MASK,      s.stencil_mask);
  w(VX_DCR_OM_STENCIL_WRITEMASK, s.stencil_writemask);
  w(VX_DCR_OM_BLEND_MODE,        s.blend_mode);
  w(VX_DCR_OM_BLEND_FUNC,        s.blend_func);
  w(VX_DCR_OM_BLEND_CONST,       s.blend_const);
  w(VX_DCR_OM_LOGIC_OP,          s.logic_op);
}

template <class W>
void emit_tex(const tex_state_t& s, W&& w) {
  w(VX_DCR_TEX_STAGE,  s.stage);
  w(VX_DCR_TEX_LOGDIM, (s.logheight << 16) | s.logwidth);
  w(VX_DCR_TEX_FORMAT, s.format);
  w(VX_DCR_TEX_FILTER, s.filter);
  w(VX_DCR_TEX_WRAP,   (s.wrap_v << 16) | s.wrap_u);
  w(VX_DCR_TEX_ADDR,   block_addr(s.addr));
  if (s.mip_offsets && s.num_mips) {
    for (uint32_t i = 0; i < s.num_mips && i < VX_TEX_LOD_MAX; ++i)
      w(VX_DCR_TEX_MIPOFF(i), s.mip_offsets[i]);
  } else {
    w(VX_DCR_TEX_MIPOFF(0), 0u);   // mip 0 at the texture base
  }
}

// Sink that fires vx_enqueue_dcr_write, latching the first failure status.
struct QueueSink {
  vx_queue_h  q;
  vx_result_t status = VX_SUCCESS;
  void operator()(uint32_t addr, uint32_t value) {
    if (status == VX_SUCCESS)
      status = vx_enqueue_dcr_write(q, addr, value, 0, nullptr, nullptr);
  }
};

// Sink that appends to a CP command batch.
struct BatchSink {
  DrawCommands& dc;
  void operator()(uint32_t addr, uint32_t value) { dc.dcr_write(addr, value); }
};

} // anonymous namespace

vx_result_t program_raster(vx_queue_h q, const raster_state_t& s) {
  QueueSink sink{q}; emit_raster(s, sink); return sink.status;
}
vx_result_t program_om(vx_queue_h q, const om_state_t& s) {
  QueueSink sink{q}; emit_om(s, sink); return sink.status;
}
vx_result_t program_tex(vx_queue_h q, const tex_state_t& s) {
  QueueSink sink{q}; emit_tex(s, sink); return sink.status;
}

void program_raster(DrawCommands& dc, const raster_state_t& s) { emit_raster(s, BatchSink{dc}); }
void program_om    (DrawCommands& dc, const om_state_t& s)     { emit_om(s, BatchSink{dc}); }
void program_tex   (DrawCommands& dc, const tex_state_t& s)    { emit_tex(s, BatchSink{dc}); }

vx_result_t draw(vx_queue_h q, vx_kernel_h fs_kernel,
                 const raster_state_t& raster,
                 const om_state_t& om,
                 const tex_state_t* tex,
                 const void* args, size_t args_size,
                 uint32_t ndim, const uint32_t grid[3], const uint32_t block[3]) {
  vx_result_t r;
  if ((r = program_raster(q, raster)) != VX_SUCCESS) return r;
  if ((r = program_om(q, om)) != VX_SUCCESS) return r;
  if (tex && (r = program_tex(q, *tex)) != VX_SUCCESS) return r;

  vx_launch_info_t li{};
  li.struct_size = sizeof(li);
  li.kernel      = fs_kernel;
  li.args_host   = args;
  li.args_size   = args_size;
  li.ndim        = ndim;
  for (uint32_t i = 0; i < 3; ++i) {
    li.grid_dim [i] = (grid  && i < ndim) ? grid[i]  : 1;
    li.block_dim[i] = (block && i < ndim) ? block[i] : 1;
  }
  return vx_enqueue_launch(q, &li, 0, nullptr, nullptr);
}

} // namespace graphics
} // namespace vortex
