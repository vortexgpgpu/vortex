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

#include "raster_core.h"
#include <array>
#include <cstring>
#include <deque>
#include <queue>
#include <unordered_map>
#include <vector>
#include "gfx_render.h"
#include "cluster.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

namespace {

// rcache config
constexpr uint32_t kRcacheNumReqs  = VX_CFG_RCACHE_NUM_BANKS;
constexpr uint32_t kRcacheMemPorts = 1;
constexpr uint32_t kRcacheLineSize = VX_CFG_MEM_BLOCK_SIZE;
constexpr uint64_t kRcacheLineMask = ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);

// Single producer lane per cluster.
constexpr uint32_t kNumRasterLanes = 1;

// Tile-header layout in RAM: { uint16 tile_x, uint16 tile_y,
// uint16 pids_offset, uint16 pids_count } = 8 bytes.
constexpr uint32_t kTileHeaderBytes = sizeof(graphics::rast_tile_header_t);

// Stamp encoding for the kernel's vx_rast() result word:
//   bits[ 3:0]  = mask
//   bits[17:4]  = pos_x  (VX_RASTER_DIM_BITS-1 = 14 bits)
//   bits[31:18] = pos_y  (VX_RASTER_DIM_BITS-1 = 14 bits)
constexpr uint32_t kPosBits = VX_RASTER_DIM_BITS - 1;
inline uint32_t encode_pos_mask(uint32_t pos_x, uint32_t pos_y, uint32_t mask) {
  return (mask & 0xfu)
       | ((pos_x & ((1u << kPosBits) - 1u)) << 4)
       | ((pos_y & ((1u << kPosBits) - 1u)) << (4 + kPosBits));
}

} // namespace

// ════════════════════════════════════════════════════════════════════
// RasterCore::Impl
// ════════════════════════════════════════════════════════════════════
//
// Producer FSM: LOAD_TILES → LOAD_PIDS → LOAD_PRIMS → RASTERIZE → READY.
// Memory traffic flows as MemReq/MemRsp through the rcache.
// Once READY, RasterReqs from raster_req_in[0] are served from quad_queue_
// (VX_CFG_NUM_THREADS stamps per response). When the queue drains, responses
// carry stamps=0 (the "done" sentinel the kernel polls for).

class RasterCore::Impl {
public:
  enum class State : uint8_t {
    IDLE,         // pre-DCR / tile_count==0
    LOAD_TILES,
    LOAD_PIDS,
    LOAD_PRIMS,
    RASTERIZE,
    READY,        // serving pops from quad_queue_
  };

  // One pending cache-line read with metadata to deposit bytes on response.
  struct PendingRead {
    uint8_t* dst_ptr;       // destination buffer pointer
    uint32_t cl_offset;     // start byte within the cache line
    uint32_t length;        // bytes to copy
  };

  // Pre-built (cache-line, dst, span) tuple. Streamed through
  // issue_pending_loads() to drive MemReq emission.
  struct LineFetch {
    uint64_t cl_addr;
    uint8_t* dst_ptr;
    uint32_t cl_offset;
    uint32_t length;
  };

  explicit Impl(RasterCore* simobject)
    : simobject_(simobject)
    , state_(State::IDLE)
    , cycle_(0)
  {}

  void reset() {
    cycle_ = 0;
    perf_stats_ = RasterCore::PerfStats();
    reset_load_state();
    state_ = State::IDLE;
    has_begun_ = false;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    dcrs_.write(addr, value);
    // DCR reconfigure invalidates the cached queue + load state AND
    // the per-frame begin trigger — the next frame must re-arm via
    // vx_rast_begin.
    reset_load_state();
    state_ = State::IDLE;
    has_begun_ = false;
    return 0;
  }

  // Called by sfu_unit when a participating warp executes vx_rast_begin.
  // Idempotent — only acts on the first (0→1) transition per frame.
  void on_begin() {
    has_begun_ = true;
  }

  const RasterCore::PerfStats& perf_stats() const { return perf_stats_; }

  void tick() {
    ++cycle_;

    // 1) Drain rcache responses → deposit bytes per pending_reads_ map.
    drain_mem_rsp();

    // 2) Advance producer FSM.
    advance_producer();

    // 3) Serve consumers when ready.
    if (state_ == State::READY) {
      serve_consumers();
    }

    // perf
    if (state_ == State::LOAD_TILES || state_ == State::LOAD_PIDS ||
        state_ == State::LOAD_PRIMS) {
      ++perf_stats_.stall_cycles;
    }
    perf_stats_.mem_latency += pending_reads_.size();
  }

private:
  // ── Reset producer-side state (preserves DCRS) ─────────────────────
  void reset_load_state() {
    tile_headers_.clear();
    pid_table_buf_.clear();
    pid_table_offset_.clear();
    prim_data_.clear();
    primary_pids_.clear();
    pending_reads_.clear();
    next_mem_tag_ = 0;
    pending_count_ = 0;
    issue_idx_ = 0;
    issue_total_ = 0;
    line_fetches_.clear();
    std::queue<RasterStamp> empty;
    std::swap(quad_queue_, empty);
    have_drained_signal_ = false;
  }

  // ── Generic helper: enqueue cache-line fetches to fill `length` bytes
  //    starting at byte_addr into dst_ptr. Coalesces within a line if
  //    the destination is contiguous.
  void enqueue_byte_range(uint64_t byte_addr, uint32_t length, uint8_t* dst_ptr) {
    while (length > 0) {
      uint64_t cl_addr  = byte_addr & kRcacheLineMask;
      uint32_t cl_off   = uint32_t(byte_addr - cl_addr);
      uint32_t cl_room  = kRcacheLineSize - cl_off;
      uint32_t span     = std::min(cl_room, length);
      LineFetch lf;
      lf.cl_addr   = cl_addr;
      lf.dst_ptr   = dst_ptr;
      lf.cl_offset = cl_off;
      lf.length    = span;
      line_fetches_.push_back(lf);
      byte_addr += span;
      dst_ptr   += span;
      length    -= span;
    }
  }

  // ── Issue MemReqs from line_fetches_ until budget exhausted or backpressure
  //    Returns true when issue is complete (issue_idx_ == issue_total_).
  bool issue_pending_loads() {
    // One request per port per cycle: issue at most one rcache load, then
    // return. A while-loop here issues N loads to a single port in one cycle.
    if (issue_idx_ >= issue_total_)
      return true;

    auto& req_ch = simobject_->rcache_req_out.at(0);
    if (req_ch.full()) return false;

    const LineFetch& lf = line_fetches_[issue_idx_];

    MemReq mreq;
    mreq.addr  = lf.cl_addr;
    mreq.op    = MemOp::LD;
    mreq.tag   = next_mem_tag_++;
    mreq.hart_id   = 0;
    mreq.uuid  = 0;

    PendingRead pr;
    pr.dst_ptr   = lf.dst_ptr;
    pr.cl_offset = lf.cl_offset;
    pr.length    = lf.length;
    pending_reads_[mreq.tag] = pr;

    req_ch.send(mreq);
    ++pending_count_;
    ++perf_stats_.mem_reads;
    ++issue_idx_;
    return (issue_idx_ >= issue_total_);
  }

  // ── Drain rcache responses ─────────────────────────────────────────
  void drain_mem_rsp() {
    // One response per port per cycle (each channel in rcache_rsp_in is a port).
    for (auto& ch : simobject_->rcache_rsp_in) {
      if (ch.empty()) continue;
      auto& rsp = ch.peek();
      auto it = pending_reads_.find(uint32_t(rsp.tag));
      if (it == pending_reads_.end()) {
        ch.pop();
        continue;
      }
      const PendingRead pr = it->second;
      pending_reads_.erase(it);

      if (rsp.data) {
        std::memcpy(pr.dst_ptr, rsp.data->data() + pr.cl_offset, pr.length);
      }
      if (pending_count_ > 0) --pending_count_;
      ch.pop();
    }
  }

  // ── Producer FSM advancement ───────────────────────────────────────
  void advance_producer() {
    switch (state_) {
    case State::IDLE: {
      // Wait for both (a) vx_rast_begin from a participating warp AND
      // (b) at least one RasterReq queued (first kernel poll), so the
      // kernel's first vx_rast() returns a real quad rather than a
      // drained sentinel.
      if (has_begun_ && !simobject_->raster_req_in.at(0).empty()) {
        kick_off_load();
      }
      break;
    }
    case State::LOAD_TILES: {
      if (!issue_pending_loads()) return;
      if (pending_count_ == 0) start_load_pids();
      break;
    }
    case State::LOAD_PIDS: {
      if (!issue_pending_loads()) return;
      if (pending_count_ == 0) start_load_prims();
      break;
    }
    case State::LOAD_PRIMS: {
      if (!issue_pending_loads()) return;
      if (pending_count_ == 0) start_rasterize();
      break;
    }
    case State::RASTERIZE: {
      run_rasterizer();
      state_ = State::READY;
      DT(3, simobject_->name() << " rasterize done: queue_size="
         << quad_queue_.size());
      break;
    }
    case State::READY:
      break;
    }
  }

  // ── Kick off LOAD_TILES (lazy, on first RasterReq) ─────────────────
  void kick_off_load() {
    uint32_t tile_count = dcrs_.read(VX_DCR_RASTER_TILE_COUNT);
    if (tile_count == 0) {
      // No work — go straight to READY (drained). Pops will get stamps=0.
      state_ = State::READY;
      have_drained_signal_ = true;
      return;
    }
    tile_headers_.resize(tile_count);

    uint64_t tbuf_addr = uint64_t(dcrs_.read(VX_DCR_RASTER_TBUF_ADDR)) << 6;
    line_fetches_.clear();
    for (uint32_t i = 0; i < tile_count; ++i) {
      enqueue_byte_range(tbuf_addr + uint64_t(i) * kTileHeaderBytes,
                         kTileHeaderBytes,
                         reinterpret_cast<uint8_t*>(&tile_headers_[i]));
    }
    issue_idx_   = 0;
    issue_total_ = uint32_t(line_fetches_.size());
    state_       = State::LOAD_TILES;
  }

  // ── After tile headers loaded → load all pid tables ────────────────
  void start_load_pids() {
    uint64_t tbuf_addr = uint64_t(dcrs_.read(VX_DCR_RASTER_TBUF_ADDR)) << 6;
    line_fetches_.clear();

    // PIDs are stored as uint32_t (4 bytes); VX_RASTER_PID_BITS=16 is the
    // consumer-side width.
    constexpr uint32_t kPidStride = sizeof(uint32_t);

    // Total PID bytes = sum across tiles of pids_count * kPidStride.
    uint32_t total_bytes = 0;
    pid_table_offset_.assign(tile_headers_.size(), 0);
    for (uint32_t i = 0; i < tile_headers_.size(); ++i) {
      pid_table_offset_[i] = total_bytes;
      total_bytes += uint32_t(tile_headers_[i].pids_count) * kPidStride;
    }
    pid_table_buf_.assign(total_bytes, 0);

    if (total_bytes == 0) {
      // No PIDs anywhere — skip to RASTERIZE (will produce no quads).
      issue_idx_   = 0;
      issue_total_ = 0;
      state_       = State::RASTERIZE;
      return;
    }

    for (uint32_t i = 0; i < tile_headers_.size(); ++i) {
      const auto& hdr = tile_headers_[i];
      if (hdr.pids_count == 0) continue;
      // hdr.pids_offset is in uint32_t-word units, measured from the end
      // of this tile's header.
      uint64_t this_header_addr = tbuf_addr + uint64_t(i) * sizeof(graphics::rast_tile_header_t);
      uint64_t pid_table_addr   = this_header_addr
                                + sizeof(graphics::rast_tile_header_t)
                                + uint64_t(hdr.pids_offset) * kPidStride;
      enqueue_byte_range(pid_table_addr,
                         uint32_t(hdr.pids_count) * kPidStride,
                         &pid_table_buf_[pid_table_offset_[i]]);
    }
    issue_idx_   = 0;
    issue_total_ = uint32_t(line_fetches_.size());
    state_       = State::LOAD_PIDS;
  }

  // ── After PIDs loaded → load prim data for each unique pid ─────────
  void start_load_prims() {
    uint64_t pbuf_addr   = uint64_t(dcrs_.read(VX_DCR_RASTER_PBUF_ADDR)) << 6;
    uint32_t pbuf_stride = dcrs_.read(VX_DCR_RASTER_PBUF_STRIDE);
    if (pbuf_stride == 0) pbuf_stride = sizeof(graphics::rast_prim_t);

    // Collect unique pids referenced by any tile; truncate 32-bit storage
    // to the 16-bit PID width.
    constexpr uint32_t kPidStride = sizeof(uint32_t);
    primary_pids_.clear();
    {
      std::unordered_map<uint16_t, bool> seen;
      for (uint32_t i = 0; i < tile_headers_.size(); ++i) {
        const auto& hdr = tile_headers_[i];
        for (uint32_t j = 0; j < hdr.pids_count; ++j) {
          uint32_t pid_word;
          std::memcpy(&pid_word, &pid_table_buf_[pid_table_offset_[i] + j * kPidStride], kPidStride);
          uint16_t pid = uint16_t(pid_word);
          if (!seen[pid]) {
            seen[pid] = true;
            primary_pids_.push_back(pid);
          }
        }
      }
    }

    line_fetches_.clear();
    if (primary_pids_.empty()) {
      issue_idx_   = 0;
      issue_total_ = 0;
      state_       = State::RASTERIZE;
      return;
    }

    // Allocate one rast_prim_t per unique pid; enqueue line fetches to fill.
    prim_data_.clear();
    prim_data_.reserve(primary_pids_.size());
    for (uint16_t pid : primary_pids_) {
      auto [it, _ins] = prim_data_.emplace(pid, graphics::rast_prim_t{});
      uint64_t prim_addr = pbuf_addr + uint64_t(pid) * pbuf_stride;
      enqueue_byte_range(prim_addr,
                         sizeof(graphics::rast_prim_t),
                         reinterpret_cast<uint8_t*>(&it->second));
    }
    issue_idx_   = 0;
    issue_total_ = uint32_t(line_fetches_.size());
    state_       = State::LOAD_PRIMS;
  }

  // ── Run the rasterizer synchronously over loaded buffers. ──────────
  void start_rasterize() {
    state_ = State::RASTERIZE;
  }

  // TE/BE walker — produces quads in the same order as the TE/BE pipeline.
  // The recursive Morton-DFS walker in Rasterizer produces a different order
  // and is not used here.
  //
  // TE: per (tile,prim), 4 priority FIFOs (one per subtile index 2*i+j with
  // i=X-bit, j=Y-bit → TL, BL, TR, BR column-major), priority 0>1>2>3, plus a
  // "bypass" path that lets TL of the current subdivision be processed next
  // when all FIFOs are empty (is_fifo_bypass).
  //
  // BE: per emitted block, 4 quads in row-major order
  // (i=jj*NUM_QUADS_DIM+ii, ii=X-bit, jj=Y-bit → TL, TR, BL, BR), grouped
  // into OUTPUT_QUADS-wide batches arbitrated by priority (batch 0 > 1...).
  // For BLOCK_LOGSIZE=2 and OUTPUT_QUADS=2, batch 0 = {TL,TR}, batch 1 =
  // {BL,BR}; a batch fires only if any quad overlaps, emitting all stamps
  // (non-overlapping ones carry mask=0 but valid pos_x/pos_y).

  struct TileWork {
    uint32_t x;
    uint32_t y;
    uint32_t level;
    graphics::vec3e_t edge_eval;  // (e0, e1, e2) values at (x, y)
  };

  // Edge-equation extents per edge, used for early-reject overlap checks at
  // each subdivision level.
  static graphics::vec3e_t compute_extents(const graphics::vec3e_t edges[3]) {
    auto extent = [](const graphics::vec3e_t& e) -> graphics::FloatE {
      graphics::FloatE z(0);
      graphics::FloatE x_part = (e.x >= z) ? e.x : z;
      graphics::FloatE y_part = (e.y >= z) ? e.y : z;
      return x_part + y_part;
    };
    return graphics::vec3e_t{ extent(edges[0]), extent(edges[1]), extent(edges[2]) };
  }

  // Overlap test: tile of size 2^(tile_logsize+1) at (x,y) with edge values
  // `edge_eval` at corner. Non-overlapping iff any edge's max value
  // (corner + extents·tile_size) is negative.
  static bool tile_overlaps(const graphics::vec3e_t& edge_eval,
                            const graphics::vec3e_t& extents,
                            uint32_t tile_logsize) {
    graphics::FloatE z(0);
    uint32_t shift = tile_logsize + 1;  // tile size in log2 pixels
    return  (edge_eval.x + (extents.x << shift)) >= z
         && (edge_eval.y + (extents.y << shift)) >= z
         && (edge_eval.z + (extents.z << shift)) >= z;
  }

  // Per-pixel coverage + scissor test for a quad.
  void compute_quad(uint32_t qx, uint32_t qy,
                    const graphics::vec3e_t& edge_eval_corner,
                    const graphics::vec3e_t edges[3],
                    uint32_t& out_mask,
                    graphics::vec3e_t out_bcoords[4]) {
    graphics::FloatE z(0);
    out_mask = 0;
    for (uint32_t pj = 0; pj < 2; ++pj) {
      for (uint32_t pi = 0; pi < 2; ++pi) {
        auto ee0 = edge_eval_corner.x + edges[0].x * int(pi) + edges[0].y * int(pj);
        auto ee1 = edge_eval_corner.y + edges[1].x * int(pi) + edges[1].y * int(pj);
        auto ee2 = edge_eval_corner.z + edges[2].x * int(pi) + edges[2].y * int(pj);
        uint32_t px = qx + pi;
        uint32_t py = qy + pj;
        bool covered = (ee0 >= z) && (ee1 >= z) && (ee2 >= z)
                    && (px >= scissor_left_)  && (px <  scissor_right_)
                    && (py >= scissor_top_)   && (py <  scissor_bottom_);
        uint32_t p = pj * 2 + pi;
        if (covered) out_mask |= (1u << p);
        out_bcoords[p].x = ee0;
        out_bcoords[p].y = ee1;
        out_bcoords[p].z = ee2;
      }
    }
  }

  // Emit covered quads for a single block in row-major / OUTPUT_QUADS-batched order.
  void emit_block_quads(const TileWork& block, uint16_t pid,
                        const graphics::vec3e_t edges[3]) {
    constexpr uint32_t kNumQuadsDim   = 1u << (VX_CFG_RASTER_BLOCK_LOGSIZE - 1);
    constexpr uint32_t kPerBlockQuads = kNumQuadsDim * kNumQuadsDim;
    constexpr uint32_t kOutputQuads   = VX_CFG_NUM_THREADS;
    constexpr uint32_t kOutputBatches =
        (kPerBlockQuads + kOutputQuads - 1) / kOutputQuads;

    // Per-quad evaluation in row-major: i=0:(0,0)=TL, 1:(1,0)=TR,
    // 2:(0,1)=BL, 3:(1,1)=BR. pos_x/pos_y in the stamp are pixel/2.
    struct QuadResult {
      uint32_t qx_pix;
      uint32_t qy_pix;
      uint32_t mask;
      graphics::vec3e_t bcoords[4];
    };
    QuadResult quads[kPerBlockQuads];
    for (uint32_t i = 0; i < kPerBlockQuads; ++i) {
      uint32_t ii = i % kNumQuadsDim;
      uint32_t jj = i / kNumQuadsDim;
      QuadResult& q = quads[i];
      q.qx_pix = block.x + 2 * ii;
      q.qy_pix = block.y + 2 * jj;
      // Edge values at quad corner = block_corner + 2*ii*ex + 2*jj*ey
      graphics::vec3e_t quad_corner_eval;
      quad_corner_eval.x = block.edge_eval.x
                         + edges[0].x * int(2 * ii) + edges[0].y * int(2 * jj);
      quad_corner_eval.y = block.edge_eval.y
                         + edges[1].x * int(2 * ii) + edges[1].y * int(2 * jj);
      quad_corner_eval.z = block.edge_eval.z
                         + edges[2].x * int(2 * ii) + edges[2].y * int(2 * jj);
      compute_quad(q.qx_pix, q.qy_pix, quad_corner_eval, edges, q.mask, q.bcoords);
    }

    // Walk batches in priority order. A batch fires iff any quad has
    // mask != 0; when fired, all OUTPUT_QUADS stamps are emitted,
    // including those with mask=0.
    for (uint32_t b = 0; b < kOutputBatches; ++b) {
      uint32_t base = b * kOutputQuads;
      bool any_overlap = false;
      for (uint32_t q = 0; q < kOutputQuads; ++q) {
        uint32_t idx = base + q;
        if (idx < kPerBlockQuads && quads[idx].mask != 0) {
          any_overlap = true;
          break;
        }
      }
      if (!any_overlap) continue;
      for (uint32_t q = 0; q < kOutputQuads; ++q) {
        uint32_t idx = base + q;
        if (idx >= kPerBlockQuads) break;
        const QuadResult& qr = quads[idx];
        RasterStamp stamp;
        stamp.pos_mask = encode_pos_mask(qr.qx_pix >> 1, qr.qy_pix >> 1, qr.mask);
        stamp.pid      = pid;
        for (uint32_t c = 0; c < 4; ++c) {
          stamp.bcoords[0][c] = uint32_t(qr.bcoords[c].x.data());
          stamp.bcoords[1][c] = uint32_t(qr.bcoords[c].y.data());
          stamp.bcoords[2][c] = uint32_t(qr.bcoords[c].z.data());
        }
        quad_queue_.push(stamp);
      }
    }
  }

  // PipeEntry: stage-2 contents (tile + precomputed stage-1 outputs).
  struct PipeEntry {
    TileWork tile;
    TileWork subs[4];
    bool     is_block;
    bool     overlap;
  };

  static PipeEntry make_pipe_entry(const TileWork& tile,
                                   const graphics::vec3e_t& extents,
                                   const graphics::vec3e_t edges[3]) {
    constexpr uint32_t kTopLog   = VX_CFG_RASTER_TILE_LOGSIZE - 1;
    constexpr uint32_t kBlockLog = VX_CFG_RASTER_BLOCK_LOGSIZE;
    PipeEntry pe;
    pe.tile = tile;
    uint32_t tile_logsize = kTopLog - tile.level;
    pe.is_block = (tile_logsize < kBlockLog);
    pe.overlap  = tile_overlaps(tile.edge_eval, extents, tile_logsize);
    if (pe.overlap && !pe.is_block) {
      uint32_t sub_size = 1u << tile_logsize;
      for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
          TileWork& s = pe.subs[2*i + j];
          s.x = tile.x + i * sub_size;
          s.y = tile.y + j * sub_size;
          s.level = tile.level + 1;
          s.edge_eval.x = tile.edge_eval.x
                        + (edges[0].x << tile_logsize) * int(i)
                        + (edges[0].y << tile_logsize) * int(j);
          s.edge_eval.y = tile.edge_eval.y
                        + (edges[1].x << tile_logsize) * int(i)
                        + (edges[1].y << tile_logsize) * int(j);
          s.edge_eval.z = tile.edge_eval.z
                        + (edges[2].x << tile_logsize) * int(i)
                        + (edges[2].y << tile_logsize) * int(j);
        }
      }
    }
    return pe;
  }

  // 2-stage tile-traversal pipeline with 4 priority FIFOs (one per subtile
  // position 2*i+j, i=X-bit, j=Y-bit → column-major TL,BL,TR,BR) and a
  // bypass path.
  //
  // Per cycle: stage2 emits (if block) or pushes its subs to FIFOs (skipping
  // F[0] on bypass); arb sees FIFO state BEFORE push and picks the next
  // stage1 (priority 0 > 1 > 2 > 3); pipeline advances (s2 <= s1, s1 <= arb
  // pick / bypass). The one-cycle delay between subdivision and FIFO push is
  // load-bearing — it lets older non-TL subtiles get selected before TL-side
  // descendants push their own F[0] entries.
  void te_walk_tile(uint32_t tile_x, uint32_t tile_y, uint16_t pid,
                    const graphics::vec3e_t edges[3]) {
    graphics::vec3e_t extents = compute_extents(edges);

    TileWork initial;
    initial.x = tile_x;
    initial.y = tile_y;
    initial.level = 0;
    initial.edge_eval.x = edges[0].x * int(tile_x) + edges[0].y * int(tile_y) + edges[0].z;
    initial.edge_eval.y = edges[1].x * int(tile_x) + edges[1].y * int(tile_y) + edges[1].z;
    initial.edge_eval.z = edges[2].x * int(tile_x) + edges[2].y * int(tile_y) + edges[2].z;

    std::array<std::queue<TileWork>, 4> fifos;
    auto fifos_all_empty = [&]() {
      for (const auto& f : fifos) if (!f.empty()) return false;
      return true;
    };

    bool      s1_has = true;
    TileWork  s1     = initial;
    bool      s2_has = false;
    PipeEntry s2{};

    while (s1_has || s2_has || !fifos_all_empty()) {
      // ── Stage 2 work: emit block, or note that subs are pending push ───
      bool emit_active = s2_has && s2.overlap && s2.is_block;
      bool push_active = s2_has && s2.overlap && !s2.is_block;

      if (emit_active) {
        emit_block_quads(s2.tile, pid, edges);
      }

      // ── Arb decision (sees FIFOs BEFORE this cycle's push) ─────────────
      int arb_idx = -1;
      for (uint32_t i = 0; i < 4; ++i) {
        if (!fifos[i].empty()) { arb_idx = i; break; }
      }
      bool arb_valid = (arb_idx >= 0);
      // is_fifo_bypass: stage2 has subs to push AND no FIFO has anything to grant.
      bool bypass = push_active && !arb_valid;

      // ── Pick next stage 1 ──────────────────────────────────────────────
      bool     next_s1_has = false;
      TileWork next_s1{};
      if (arb_valid) {
        next_s1 = fifos[arb_idx].front();
        fifos[arb_idx].pop();
        next_s1_has = true;
      } else if (bypass) {
        next_s1 = s2.subs[0];  // TL bypasses F[0]
        next_s1_has = true;
      }

      // ── Push stage 2's subs (F[0] skipped on bypass) ───────────────────
      if (push_active) {
        for (uint32_t i = (bypass ? 1u : 0u); i < 4; ++i) {
          fifos[i].push(s2.subs[i]);
        }
      }

      // ── Pipeline advance ───────────────────────────────────────────────
      if (s1_has) {
        s2 = make_pipe_entry(s1, extents, edges);
        s2_has = true;
      } else {
        s2_has = false;
      }
      s1 = next_s1;
      s1_has = next_s1_has;
    }
  }

  // Cached scissor config (refreshed in run_rasterizer from dcrs_).
  uint32_t scissor_left_   = 0;
  uint32_t scissor_top_    = 0;
  uint32_t scissor_right_  = 0;
  uint32_t scissor_bottom_ = 0;

  void run_rasterizer() {
    if (tile_headers_.empty() || primary_pids_.empty()) {
      have_drained_signal_ = true;
      return;
    }

    // Refresh scissor config from DCRs (matches Rasterizer::configure).
    scissor_left_   = dcrs_.read(VX_DCR_RASTER_SCISSOR_X) & 0xffff;
    scissor_right_  = dcrs_.read(VX_DCR_RASTER_SCISSOR_X) >> 16;
    scissor_top_    = dcrs_.read(VX_DCR_RASTER_SCISSOR_Y) & 0xffff;
    scissor_bottom_ = dcrs_.read(VX_DCR_RASTER_SCISSOR_Y) >> 16;

    uint32_t tile_size = 1u << VX_CFG_RASTER_TILE_LOGSIZE;
    for (uint32_t t = 0; t < tile_headers_.size(); ++t) {
      const auto& hdr = tile_headers_[t];
      uint32_t tile_x = uint32_t(hdr.tile_x) * tile_size;
      uint32_t tile_y = uint32_t(hdr.tile_y) * tile_size;
      for (uint32_t j = 0; j < hdr.pids_count; ++j) {
        uint32_t pid_word;
        std::memcpy(&pid_word,
                    &pid_table_buf_[pid_table_offset_[t] + j * sizeof(uint32_t)],
                    sizeof(uint32_t));
        uint16_t pid = uint16_t(pid_word);
        auto pit = prim_data_.find(pid);
        if (pit == prim_data_.end()) continue;
        const auto& prim = pit->second;
        graphics::vec3e_t edges[3] = { prim.edges[0], prim.edges[1], prim.edges[2] };
        te_walk_tile(tile_x, tile_y, pid, edges);
      }
    }
    have_drained_signal_ = true;
  }

  // ── Serve per-core pops from quad_queue_ ───────────────────────────
  void serve_consumers() {
    auto& req_ch = simobject_->raster_req_in.at(0);
    auto& rsp_ch = simobject_->raster_rsp_out.at(0);
    // One consumer transaction per cycle: serve at most one request → response.
    if (req_ch.empty() || rsp_ch.full())
      return;
    {
      const auto& req = req_ch.peek();
      RasterRsp rsp(req);

      // One stamp per active lane. When queue empty, leave default-
      // constructed (pos_mask=0 → drain sentinel).
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        if (!(req.tmask_bits & (1u << t))) continue;
        if (quad_queue_.empty()) continue;
        rsp.stamps[t] = quad_queue_.front();
        quad_queue_.pop();
      }
      rsp_ch.send(rsp);
      req_ch.pop();
    }
  }

  // ── Members ─────────────────────────────────────────────────────────
  RasterCore*                simobject_;
  RasterDCRS       dcrs_;

  State                      state_;

  // Loaded buffers.
  std::vector<graphics::rast_tile_header_t>             tile_headers_;
  std::vector<uint8_t>                                  pid_table_buf_;
  std::vector<uint32_t>                                 pid_table_offset_;  // per-tile offset into pid_table_buf_
  std::unordered_map<uint16_t, graphics::rast_prim_t>   prim_data_;
  std::vector<uint16_t>                                 primary_pids_;

  // Active-phase load issuance.
  std::vector<LineFetch>                                line_fetches_;
  uint32_t                                              issue_idx_   = 0;
  uint32_t                                              issue_total_ = 0;
  uint32_t                                              pending_count_ = 0;
  std::unordered_map<uint32_t, PendingRead>             pending_reads_;
  uint32_t                                              next_mem_tag_ = 0;

  // Quad queue (consumer-facing).
  std::queue<RasterStamp>                               quad_queue_;
  bool                                                  have_drained_signal_ = false;

  // Per-frame begin trigger — gates kick_off_load until a participating
  // warp has executed vx_rast_begin. Cleared on DCR write so each frame
  // must re-arm.
  bool                                                  has_begun_ = false;

  uint64_t                                              cycle_;
  RasterCore::PerfStats                                 perf_stats_;
};

// ════════════════════════════════════════════════════════════════════
// RasterCore — wrappers
// ════════════════════════════════════════════════════════════════════

RasterCore::RasterCore(const SimContext& ctx, const char* name, Cluster* cluster)
  : SimObject<RasterCore>(ctx, name)
  , raster_req_in(kNumRasterLanes, this)
  , raster_rsp_out(kNumRasterLanes, this)
  , rcache_req_out(kRcacheNumReqs, this)
  , rcache_rsp_in(kRcacheNumReqs, this)
{
  __unused(cluster);
  impl_ = new Impl(this);
}

RasterCore::~RasterCore() {
  delete impl_;
}

void RasterCore::on_reset() { impl_->reset(); }
void RasterCore::on_tick()  { impl_->tick(); }

int RasterCore::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

void RasterCore::begin() {
  impl_->on_begin();
}

const RasterCore::PerfStats& RasterCore::perf_stats() const {
  return impl_->perf_stats();
}
