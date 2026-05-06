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
#include <graphics.h>
#include "cluster.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

namespace {

// rcache config — mirrors VX_gpu_pkg.sv:1075-1085 + VX_config.toml [rcache].
constexpr uint32_t kRcacheNumReqs  = RCACHE_NUM_BANKS;
constexpr uint32_t kRcacheMemPorts = 1;
constexpr uint32_t kRcacheLineSize = MEM_BLOCK_SIZE;
constexpr uint64_t kRcacheLineMask = ~uint64_t(MEM_BLOCK_SIZE - 1);

// One RasterCore per cluster after the RTL-style raster_arb collapses
// NUM_SLICES → 1. SimX models a single producer lane.
constexpr uint32_t kNumRasterLanes = 1;

// Tile-header layout in RAM: { uint16 tile_x, uint16 tile_y,
// uint16 pids_offset, uint16 pids_count } = 8 bytes.
constexpr uint32_t kTileHeaderBytes = sizeof(graphics::rast_tile_header_t);

// Stamp encoding for the kernel's vx_rast() result word. Mirrors the RTL
// concat in VX_raster_unit.sv:59-62 and the kernel decode in
// raster_smoke/kernel.cpp:
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
// The producer FSM walks tile/prim buffers in three phases:
//   LOAD_TILES → LOAD_PIDS → LOAD_PRIMS → RASTERIZE → READY.
// Memory traffic flows as MemReq/MemRsp through the rcache; phase data
// is deposited via per-tag (target_ptr, byte_offset, length) bookkeeping.
//
// Once READY, per-core RasterReqs are drained from raster_req_in[0] and
// served from quad_queue_, encoding NUM_THREADS stamps per response. When
// queue drains, subsequent responses carry stamps=0 (the "done" sentinel
// the kernel polls for).

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
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    dcrs_.write(addr, value);
    // DCR reconfigure invalidates the cached queue + load state. Lazy:
    // the next RasterReq will kick off a fresh load.
    reset_load_state();
    state_ = State::IDLE;
    return 0;
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
    while (issue_idx_ < issue_total_) {
      auto& req_ch = simobject_->rcache_req_out.at(0);
      if (req_ch.full()) return false;

      const LineFetch& lf = line_fetches_[issue_idx_];

      MemReq mreq;
      mreq.addr  = lf.cl_addr;
      mreq.write = false;
      mreq.op    = MemOp::READ;
      mreq.type  = AddrType::Global;
      mreq.tag   = next_mem_tag_++;
      mreq.cid   = 0;
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
    }
    return true;
  }

  // ── Drain rcache responses ─────────────────────────────────────────
  void drain_mem_rsp() {
    for (auto& ch : simobject_->rcache_rsp_in) {
      while (!ch.empty()) {
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
  }

  // ── Producer FSM advancement ───────────────────────────────────────
  void advance_producer() {
    switch (state_) {
    case State::IDLE: {
      // Wait until a RasterReq arrives — lazy kick-off.
      if (!simobject_->raster_req_in.at(0).empty()) {
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

    // Total PID bytes = sum across tiles of pids_count * 2.
    uint32_t total_bytes = 0;
    pid_table_offset_.assign(tile_headers_.size(), 0);
    for (uint32_t i = 0; i < tile_headers_.size(); ++i) {
      pid_table_offset_[i] = total_bytes;
      total_bytes += uint32_t(tile_headers_[i].pids_count) * 2;
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
      uint64_t pid_table_addr = tbuf_addr + uint64_t(hdr.pids_offset);
      enqueue_byte_range(pid_table_addr,
                         uint32_t(hdr.pids_count) * 2,
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

    // Collect unique pids referenced by any tile.
    primary_pids_.clear();
    {
      std::unordered_map<uint16_t, bool> seen;
      for (uint32_t i = 0; i < tile_headers_.size(); ++i) {
        const auto& hdr = tile_headers_[i];
        for (uint32_t j = 0; j < hdr.pids_count; ++j) {
          uint16_t pid;
          std::memcpy(&pid, &pid_table_buf_[pid_table_offset_[i] + j * 2], 2);
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

  static void shader_trampoline(uint32_t pos_mask,
                                graphics::vec3e_t bcoords[4],
                                uint32_t pid,
                                void* cb_arg) {
    auto self = static_cast<Impl*>(cb_arg);
    RasterStamp stamp;
    stamp.pos_mask = pos_mask;
    stamp.pid      = pid;
    // Pack vec3e_t bcoords[4] (4 corners × {x, y, z}) into bcoords[axis][corner]
    // as raw Q15.16 fixed-point bit patterns (matches RTL VX_raster_edge's
    // signed-integer multiplier output; the kernel reinterprets via
    // fixed16_t::make + conversion to float).
    for (uint32_t c = 0; c < 4; ++c) {
      stamp.bcoords[0][c] = uint32_t(bcoords[c].x.data());
      stamp.bcoords[1][c] = uint32_t(bcoords[c].y.data());
      stamp.bcoords[2][c] = uint32_t(bcoords[c].z.data());
    }
    self->quad_queue_.push(stamp);
  }

  void run_rasterizer() {
    if (tile_headers_.empty() || primary_pids_.empty()) {
      have_drained_signal_ = true;
      return;
    }
    graphics::Rasterizer rasterizer(&shader_trampoline, this,
                                    RASTER_TILE_LOGSIZE,
                                    RASTER_BLOCK_LOGSIZE);
    rasterizer.configure(dcrs_);

    uint32_t tile_size = 1u << RASTER_TILE_LOGSIZE;
    for (uint32_t t = 0; t < tile_headers_.size(); ++t) {
      const auto& hdr = tile_headers_[t];
      uint32_t tile_x = uint32_t(hdr.tile_x) * tile_size;
      uint32_t tile_y = uint32_t(hdr.tile_y) * tile_size;
      for (uint32_t j = 0; j < hdr.pids_count; ++j) {
        uint16_t pid;
        std::memcpy(&pid, &pid_table_buf_[pid_table_offset_[t] + j * 2], 2);
        auto pit = prim_data_.find(pid);
        if (pit == prim_data_.end()) continue;
        const auto& prim = pit->second;
        graphics::vec3e_t edges[3] = { prim.edges[0], prim.edges[1], prim.edges[2] };
        rasterizer.renderPrimitive(tile_x, tile_y, pid, edges);
      }
    }
    have_drained_signal_ = true;
  }

  // ── Serve per-core pops from quad_queue_ ───────────────────────────
  void serve_consumers() {
    auto& req_ch = simobject_->raster_req_in.at(0);
    auto& rsp_ch = simobject_->raster_rsp_out.at(0);
    while (!req_ch.empty() && !rsp_ch.full()) {
      const auto& req = req_ch.peek();
      RasterRsp rsp(req);

      // One stamp per active lane. When queue empty, leave default-
      // constructed (pos_mask=0 → drain sentinel).
      for (uint32_t t = 0; t < NUM_THREADS; ++t) {
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
  graphics::RasterDCRS       dcrs_;

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

const RasterCore::PerfStats& RasterCore::perf_stats() const {
  return impl_->perf_stats();
}
