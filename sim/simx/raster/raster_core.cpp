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
#include <algorithm>
#include <array>
#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <queue>
#include <utility>
#include <vector>
#include "gfx_ff_model.h"
#include "gfx_frag_rast.h"
#include "cluster.h"
#include "processor_impl.h"   // ProcessorImpl::ram() — early-Z committed-depth peek
#include "core.h"
#include "scheduler.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

namespace {

// rcache config
constexpr uint32_t kRcacheNumReqs  = VX_CFG_RCACHE_NUM_BANKS;
constexpr uint64_t kRcacheLineMask = ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);

// Single producer lane per cluster.
constexpr uint32_t kNumRasterLanes = 1;

constexpr uint32_t kNumSlices = VX_CFG_RASTER_NUM_SLICES;

// The producer reads 32-bit words through the rcache: a bin header
// { uint16 bin_x, uint16 bin_y, uint32 pids_offset, uint32 pids_count }, one
// pid word per listed primitive, and the leading words of its record — the
// scissor rectangle and the edges, plus the depth plane for early-Z.
constexpr uint32_t kWordSize         = 4;
constexpr uint32_t kTileHeaderWords  = sizeof(graphics::rast_bin_header_t) / kWordSize;
constexpr uint32_t kPrimScissorWords = 2;
constexpr uint32_t kPrimEdgeWords    = 9;
#if VX_CFG_RASTER_EARLYZ_ENABLED
constexpr uint32_t kPrimZPlaneWords  = 3;
#else
constexpr uint32_t kPrimZPlaneWords  = 0;
#endif
constexpr uint32_t kPrimWords = kPrimScissorWords + kPrimEdgeWords + kPrimZPlaneWords;
constexpr uint32_t kFetchWords = std::max(kPrimWords, kTileHeaderWords);
static_assert(kPrimWords * kWordSize <= sizeof(graphics::rast_prim_t), "record fetch exceeds the record");

// Fetch requests the memory scheduler queues and tracks at once.
constexpr uint32_t kFetchQueueSize = VX_CFG_RASTER_MEM_QUEUE_SIZE;
// Records in flight from their pid request to the edge stage; the record
// buffer holds one more, so a returning record always has room.
constexpr uint32_t kRecordBufSize   = VX_CFG_RASTER_MEM_FIFO_DEPTH;
constexpr uint32_t kMaxPendingPrims = kRecordBufSize - 1;
// Integer-multiplier pipeline depth: the pid -> record address product, and
// the edge functions evaluated at the bin origin.
constexpr uint32_t kMulLatency = 3;

// Walk geometry: a bin descends to blocks of 2^kBlockLogSize pixels, each
// evaluated as kNumQuadsDim^2 quads and emitted in kOutputQuads-wide batches.
constexpr uint32_t kTileLogSize   = VX_CFG_RASTER_BIN_LOG_SIZE;
constexpr uint32_t kBlockLogSize  = VX_CFG_RASTER_BLOCK_LOG_SIZE;
constexpr uint32_t kNumQuadsDim   = 1u << (kBlockLogSize - 1);
constexpr uint32_t kPerBlockQuads = kNumQuadsDim * kNumQuadsDim;
constexpr uint32_t kOutputQuads   = VX_CFG_NUM_THREADS;
constexpr uint32_t kOutputBatches = (kPerBlockQuads + kOutputQuads - 1) / kOutputQuads;
static_assert(kOutputBatches <= 32, "batch mask is 32 bits");
constexpr uint32_t kBlockBufSize  = 2 * kOutputBatches;
constexpr uint32_t kQuadFifoDepth = VX_CFG_RASTER_QUAD_FIFO_DEPTH;

#if VX_CFG_RASTER_EARLYZ_ENABLED
// Early-Z reads committed depth through the ocache: one word per covered pixel
// per sample, a scheduler request per sample, in batches of one word per ocache
// port.
constexpr uint32_t kEzPixels      = kOutputQuads * 4;
using EzLanes = std::bitset<kEzPixels>;
constexpr uint32_t kEzQueueSize   = VX_CFG_RASTER_MEM_QUEUE_SIZE;
constexpr uint32_t kOcacheNumReqs = VX_CFG_OCACHE_NUM_BANKS;
constexpr uint64_t kOcacheLineMask = ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
// The request leaves through the scheduler's output register and the slice
// merge's; a completed response through the merge's and the scheduler's.
constexpr uint32_t kEzReqDelay    = 2;
constexpr uint32_t kEzRspLatency  = 2;
#endif

// A pass addresses at most GFX_RASTER_MAX_PRIMS primitives: the hardware carries
// a pid, and counts a bin's pids, in VX_RASTER_PID_BITS bits. A pass beyond that
// is one the hardware cannot render, so the model stops on it rather than
// render it.
[[noreturn]] void raster_pass_overflow(const char* what, uint32_t value) {
  std::fprintf(stderr, "error: RASTER pass %s %u exceeds the %u-primitive pass limit\n",
               what, value, GFX_RASTER_MAX_PRIMS);
  std::abort();
}

// Stamp encoding for the staged frag_payload pos_mask word (window slot 0):
//   bits[ 3:0]  = mask
//   bits[17:4]  = pos_x  (VX_RASTER_DIM_BITS-1 = 14 bits)
//   bits[31:18] = pos_y  (VX_RASTER_DIM_BITS-1 = 14 bits)
constexpr uint32_t kPosBits = VX_RASTER_DIM_BITS - 1;
inline uint32_t encode_pos_mask(uint32_t pos_x, uint32_t pos_y, uint32_t mask) {
  return (mask & 0xfu)
       | ((pos_x & ((1u << kPosBits) - 1u)) << 4)
       | ((pos_y & ((1u << kPosBits) - 1u)) << (4 + kPosBits));
}

#if VX_CFG_RASTER_EARLYZ_ENABLED

// The plane's move from a pixel's centre to sample k of the 4x pattern, in
// 64 bits as the FS late-Z computes it: (A'*DX16[k] + B'*DY16[k]) >> 4.
inline int32_t earlyz_sample_offset(const graphics::rast_attrib_t& zp, uint32_t k) {
  static constexpr int dx16[gfx_rast::RAST_MSAA_SAMPLES] = GFX_MSAA4_SAMPLE_DX16;
  static constexpr int dy16[gfx_rast::RAST_MSAA_SAMPLES] = GFX_MSAA4_SAMPLE_DY16;
  return int32_t(((int64_t)zp.x.data() * dx16[k] + (int64_t)zp.y.data() * dy16[k]) >> 4);
}

// Early-Z: evaluate the screen-space depth plane at pixel center (X,Y), moved by
// `zoff` to a sample, and quantize to the 24-bit zbuf value. Bit-identical to the
// FS late-Z path (kernel PLANE_Z Q7.24 plane MAC written SATURATED to
// [0, OM_DEPTH_MASK]), so early-Z culls exactly the fragments late-Z would reject.
inline uint32_t earlyz_plane_depth(const graphics::rast_attrib_t& zp, int X, int Y,
                                   int32_t zoff = 0) {
  int32_t zbits = (int32_t)( (int64_t)zp.x.data() * X
                           + (int64_t)zp.y.data() * Y
                           + (int64_t)zp.z.data()
                           + zoff );
  if (zbits < 0) {
    return 0;
  }
  uint32_t d = (uint32_t)zbits;
  return d > OM_DEPTH_MASK ? (uint32_t)OM_DEPTH_MASK : d;
}

// Early-Z cull decision on 24-bit depth (cand = incoming candidate, stored =
// committed depth read from the buffer). Returns true iff the fragment is
// PROVABLY occluded and safe to drop.
//
// Read-only early-Z never writes depth; the ROP remains the authoritative late-Z.
// The committed depth we read is not causally pinned to this fragment: it may
// already contain this fragment's own eventual write, a co-planar (equal-depth)
// fragment's write, or a causally-later nearer write (fragment processing is not
// globally in submission order). So early-Z must use the REFLEXIVE RELAXATION of
// the depth func — it may cull only a fragment that is STRICTLY behind stored.
// A visible fragment has cand == final-buffer depth <= stored (the buffer only
// moves toward the winner), so a strict-behind test can never cull it: image
// identity holds regardless of read freshness or pipeline ordering. Culling on
// equality (the exact func) is what wrongly drops own/co-planar/final writes.
// Non-monotonic funcs (EQUAL/NOTEQUAL/NEVER/ALWAYS) are never early-culled; the
// driver only arms earlyz_safe for the monotone LESS/LEQUAL/GREATER/GEQUAL case.
inline bool earlyz_occluded(uint32_t func, uint32_t cand, uint32_t stored) {
  switch (func) {
  case VX_OM_DEPTH_FUNC_LESS:
  case VX_OM_DEPTH_FUNC_LEQUAL:  return cand >  stored;   // keep on cand <= stored
  case VX_OM_DEPTH_FUNC_GREATER:
  case VX_OM_DEPTH_FUNC_GEQUAL:  return cand <  stored;   // keep on cand >= stored
  default:                       return false;            // never early-cull
  }
}

#endif

// A registered queue stage: an entry pushed in cycle t can leave from cycle
// t + latency, one push and one pop per cycle. Readiness is the registered
// not-full flag sampled at the start of the cycle, so a pop does not make room
// for a push in the same cycle.
template <typename T>
class StageQueue {
public:
  StageQueue(uint32_t capacity, uint32_t latency)
    : capacity_(capacity)
    , latency_(latency)
  {}

  void begin_cycle() {
    size_at_start_ = uint32_t(entries_.size());
  }

  bool ready() const {
    return size_at_start_ < capacity_;
  }

  bool valid(uint64_t cycle) const {
    return !entries_.empty() && entries_.front().second <= cycle;
  }

  const T& front() const {
    return entries_.front().first;
  }

  void push(T value, uint64_t cycle) {
    entries_.emplace_back(std::move(value), cycle + latency_);
  }

  void pop() {
    entries_.pop_front();
  }

  bool empty() const {
    return entries_.empty();
  }

  void clear() {
    entries_.clear();
    size_at_start_ = 0;
  }

private:
  std::deque<std::pair<T, uint64_t>> entries_;
  uint32_t capacity_;
  uint32_t latency_;
  uint32_t size_at_start_ = 0;
};

} // namespace

// ════════════════════════════════════════════════════════════════════
// RasterCore::Impl
// ════════════════════════════════════════════════════════════════════
//
// The producer streams like the hardware unit it models: a fetch FSM walks the
// owned bins, requesting each bin header, then one pid word per listed
// primitive and that primitive's record words, all through a word-granular
// memory scheduler on the rcache. Each returned record is handed to the edge
// stage and the tile walker as soon as it arrives, while later pid and record
// fetches are in flight. The walker (TE) and block evaluator (BE) are stepped
// one pipeline stage per cycle, and every emitted quad batch reaches the
// consumer-facing output buffer in the cycle the BE produces it.
//
// RasterReqs from raster_req_in[0] wait in per-core request queues until the
// output buffer holds a batch owned by the requesting core (VX_CFG_NUM_THREADS
// stamps per response). Once the producer has fully drained, waiting requests
// are answered with stamps=0, the "done" sentinel the distributor polls for.

class RasterCore::Impl {
public:
  Impl(RasterCore* simobject, Cluster* cluster)
    : simobject_(simobject)
    , cluster_(cluster)
    , cores_per_cluster_(NUM_SOCKETS * VX_CFG_SOCKET_SIZE)
    , base_core_(cluster->id() * (NUM_SOCKETS * VX_CFG_SOCKET_SIZE))
    , total_cores_(VX_CFG_NUM_CLUSTERS * NUM_SOCKETS * VX_CFG_SOCKET_SIZE)
    , fetch_queue_(kFetchQueueSize, 1)
    , fetch_rsp_buf_(2, 1)
    , record_buf_(kRecordBufSize, 1)
    , out_buf_(2, 1)
    , parked_reqs_(NUM_SOCKETS * VX_CFG_SOCKET_SIZE)
    , cycle_(0)
  {
    for (uint32_t s = 0; s < kNumSlices; ++s) {
      slices_.emplace_back();
    }
  }

  void reset() {
    cycle_ = 0;
    perf_stats_ = RasterCore::PerfStats();
    reset_producer();
    // The per-launch reset precedes the delegated launch's frame kick, so both
    // arm states start clean; the frag descriptor persists (DCR state, re-
    // supplied by each draw's config sequence).
    frag_armed_  = false;
    arm_pending_ = false;
    fwd_live_    = 0;
  }

  void set_frag_descriptor(uint64_t frag_entry, uint64_t frag_param) {
    frag_entry_ = frag_entry;
    frag_param_ = frag_param;
  }

  void frame_kick() {
    // Delegated draw launch from the device KMU: the launch follows the draw's
    // whole DCR sequence, so the producer config and FS descriptor are
    // complete. Prime the one-shot kick; tick() starts the producer and arms
    // the owned distributors exactly once per kick (see tick step 0). Kicks
    // come only from the draw's own grid-less launch, so a stale request can
    // never re-arm a later draw's compute front end.
    arm_pending_ = true;
  }

  void on_fwd_done(const FwdDone& msg) {
    __unused(msg);
    assert(fwd_live_ > 0);
    if (--fwd_live_ == 0) {
      frag_armed_ = false;
    }
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    dcrs_.write(addr, value);
    // A DCR reconfigure discards the producer state; the next frame kick
    // starts the walk over the new configuration.
    reset_producer();
    return 0;
  }

  // Snoop the OM depth DCRs (shared depth-buffer config) so the early-Z stage
  // can test against committed depth. Called by Cluster::dcr_write for OM-range
  // writes; does NOT re-arm the producer (unlike a raster DCR write).
  void om_dcr_snoop(uint32_t addr, uint32_t value) {
  #if VX_CFG_RASTER_EARLYZ_ENABLED
    switch (addr) {
    case VX_DCR_OM_ZBUF_ADDR:    zbuf_base_  = uint64_t(value) << 6; break;
    case VX_DCR_OM_ZBUF_PITCH:   zbuf_pitch_ = value; break;
    case VX_DCR_OM_ZBUF_SAMPLE_STRIDE: zbuf_sample_stride_ = value; break;
    case VX_DCR_OM_DEPTH_FUNC:   depth_func_ = value; break;
    case VX_DCR_OM_EARLYZ_SAFE:  earlyz_safe_ = value; break;
    case VX_DCR_OM_DEPTH_FORMAT:
      depth_d24_ = (value & 0x7) == VX_OM_DEPTH_FORMAT_D24S8;
      break;
    default: break;
    }
  #else
    __unused(addr, value);
  #endif
  }

  const RasterCore::PerfStats& perf_stats() const { return perf_stats_; }

  void tick() {
    ++cycle_;

    // 0) A frame kick starts the producer on the next cycle and arms the owned
    //    cores' fragment work distributors once per draw. Decoupled from the
    //    TILE_COUNT poll: a zero-tile / fully-culled draw arms too and drains
    //    to zero quads (the producer serves the drained sentinel), so the
    //    request cannot persist and re-arm the next draw's compute front end.
    //    frag_entry_ == 0 (no FS bound) consumes the kick without starting or
    //    arming.
    bool start = start_pending_;
    start_pending_ = false;
    if (arm_pending_ && !frag_armed_) {
      arm_pending_ = false;
      if (frag_entry_ != 0) {
        start_pending_ = true;
        frag_armed_ = true;
        fwd_live_ = cores_per_cluster_;
        for (uint32_t c = 0; c < cores_per_cluster_; ++c) {
          simobject_->fwd_arm_out.at(c).send({Word(frag_entry_), Word(frag_param_)});
        }
      }
    }

    // Every stage reads its neighbours' registered state, so each queue's
    // readiness is sampled before any stage moves this cycle.
    begin_cycle();

    // 1) Consumers, then the pipeline from its output back to the fetch FSM.
    serve_consumers();
    drain_slices();
  #if VX_CFG_RASTER_EARLYZ_ENABLED
    ez_mem_step();
  #endif
    for (auto& slice : slices_) {
    #if VX_CFG_RASTER_EARLYZ_ENABLED
      ez_step(slice);
    #endif
      be_step(slice);
      te_step(slice);
    }
    edge_step();
    fetch_step(start);

    // The draw's fragment phase closes in on_fwd_done: each distributor
    // reports its drain on the done link, and the last report clears
    // frag_armed_, letting the NEXT draw's FS-descriptor edge (arm_pending_)
    // re-arm without any re-arm during the next draw's front end.

    perf_stats_.mem_latency += words_in_flight_;
    for (const auto& slice : slices_) {
      if (!slice.te_idle()) {
        ++perf_stats_.raster_cycles;
        break;
      }
    }
  }

private:
  // ── Primitive and walk records ─────────────────────────────────────

  // One fetched primitive, as the walker sees it for the bin it was listed in.
  // Tiles of consecutive primitives can coexist in the walk pipeline, so each
  // tile carries its primitive.
  struct PrimCtx {
    uint32_t          pid;
    uint32_t          owner;      // owner core of the bin, local to this cluster
    uint32_t          xloc, yloc; // bin origin in pixels
    graphics::vec3e_t edges[3];
    graphics::vec3e_t extents;
    // Edge value at a pixel's centre minus that at its top-left corner (zero at
    // one sample): a 4x tile is bounded from its first pixel's corner.
    graphics::vec3e_t cbias;
    // The record's rectangle intersected with the DCR one: [left, right) x [top, bottom).
    uint32_t          scissor_left, scissor_right, scissor_top, scissor_bottom;
    graphics::rast_attrib_t zplane;
  };
  using PrimRef = std::shared_ptr<const PrimCtx>;

  struct TileWork {
    uint32_t          x;
    uint32_t          y;
    uint32_t          level;
    graphics::vec3e_t edge_eval;  // (e0, e1, e2) values at (x, y)
    PrimRef           prim;
  };

  // One emitted batch of quads; all of its stamps come from one block.
  struct QuadBatch {
    std::array<RasterStamp, kOutputQuads> stamps;
    PrimRef prim;
  };

  // ── Fetch: requests, responses, scheduler slots ───────────────────

  enum class FetchKind : uint8_t { TILE, PID, PRIM };

  struct FetchReq {
    FetchKind kind = FetchKind::TILE;
    uint32_t  mask = 0;                    // words requested
    uint32_t  pid  = 0;                    // PRIM: the record's primitive
    std::array<uint64_t, kFetchWords> addr{};   // byte address per word
  };

  struct FetchRsp {
    FetchKind kind = FetchKind::TILE;
    uint32_t  pid  = 0;
    std::array<uint32_t, kFetchWords> data{};
  };

  struct FetchSlot {
    bool     busy    = false;
    uint32_t pending = 0;   // words still awaiting their response
    FetchReq req;
    FetchRsp rsp;
  };

  // ── Walk pipeline state, one per slice ─────────────────────────────

  // Stage-2 contents (tile + precomputed stage-1 outputs).
  struct PipeEntry {
    TileWork tile;
    TileWork subs[4];
    bool     is_block;
  };

  // Tile FIFO with a registered head: a push becomes grantable two cycles
  // later (registered occupancy flag + output stage), while sustained pops
  // stream one per cycle. The arb only sees `out`.
  struct TeFifo {
    std::queue<std::pair<TileWork, uint64_t>> ram; // (tile, push_cycle)
    bool     out_valid = false;
    TileWork out{};

    bool empty() const { return !out_valid && ram.empty(); }
    void push(const TileWork& t, uint64_t cycle) { ram.push({t, cycle}); }
    void refill(uint64_t cycle) {
      if (!out_valid && !ram.empty() && cycle > ram.front().second) {
        out = ram.front().first;
        ram.pop();
        out_valid = true;
      }
    }
  };

  struct QuadEval {
    uint32_t qx_pix[kPerBlockQuads];
    uint32_t qy_pix[kPerBlockQuads];
    uint32_t mask[kPerBlockQuads];
    uint32_t smask[kPerBlockQuads];   // pixel p's sample coverage in bits [4p, 4p+4)
    PrimRef  prim;
  };

#if VX_CFG_RASTER_EARLYZ_ENABLED
  // ── Early-Z stage, one per slice, between its quad FIFO and the bus ───
  enum class EzState : uint8_t { IDLE, EVAL1, EVAL1B, EVAL2, RUN, SEND };

  // A scheduler request: one sample's reads for the wave's covering pixels.
  struct EzSlot {
    bool     busy    = false;
    uint32_t sample  = 0;
    EzLanes  lanes;         // pixels read
    EzLanes  waiting;       // pixels whose word is still to arrive
    std::array<uint64_t, kEzPixels> addr{};
    std::array<uint32_t, kEzPixels> data{};
  };

  struct EarlyZ {
    EzState   state = EzState::IDLE;
    QuadBatch wave;
    std::array<uint32_t, kEzPixels> mask{};   // each pixel's sample coverage
    std::array<int32_t, gfx_rast::RAST_MSAA_SAMPLES> zoff{};
    uint32_t  issue_k    = 0;      // next sample to read
    bool      issue_done = false;
    uint32_t  pending    = 0;      // samples read, not yet compared
    uint32_t  cand_k     = 0;      // the sample whose candidates are registered
    bool      sent       = false;  // SEND: the wave left this cycle
    // Memory scheduler.
    StageQueue<uint32_t> req_queue{kEzQueueSize, 1};
    std::array<EzSlot, kEzQueueSize> slots;
    uint32_t  slots_busy = 0;
    bool      slots_full = false;
    uint32_t  batch_idx  = 0;
    StageQueue<uint32_t> rsp_buf{2, kEzRspLatency};

    bool idle() const {
      return state == EzState::IDLE && slots_busy == 0;
    }

    void clear() {
      state = EzState::IDLE;
      issue_k = 0;
      issue_done = false;
      pending = 0;
      cand_k = 0;
      sent = false;
      req_queue.clear();
      slots = {};
      slots_busy = 0;
      slots_full = false;
      batch_idx = 0;
      rsp_buf.clear();
    }

    void begin_cycle() {
      req_queue.begin_cycle();
      rsp_buf.begin_cycle();
      slots_full = (slots_busy == kEzQueueSize);
    }
  };
#endif

  struct Slice {
    // Primitive input to the walker.
    StageQueue<PrimRef>   prim_in{2, 1};
    // TE: stage 1 (tile under test), stage 2 (tested, overlapping), 4 FIFOs.
    std::optional<TileWork>  s1;
    std::optional<PipeEntry> s2;
    std::array<TeFifo, 4>    fifos;
    // TE -> BE block buffer.
    StageQueue<TileWork>  blocks{kBlockBufSize, 1};
    // BE: registered block, registered quad evaluation, sent batches.
    std::optional<TileWork> be_block;
    std::optional<QuadEval> be_quads;
    uint32_t              batch_sent = 0;
    uint32_t              be_phase   = 0;   // 4x: cycles the input block has held
    // BE output.
    StageQueue<QuadBatch> quads{kQuadFifoDepth, 1};
#if VX_CFG_RASTER_EARLYZ_ENABLED
    EarlyZ                ez;
#endif

    bool te_idle() const {
      if (s1 || s2) {
        return false;
      }
      for (const auto& f : fifos) {
        if (!f.empty()) {
          return false;
        }
      }
      return true;
    }

    bool idle() const {
    #if VX_CFG_RASTER_EARLYZ_ENABLED
      if (!ez.idle()) {
        return false;
      }
    #endif
      return prim_in.empty() && te_idle() && blocks.empty()
          && !be_block && !be_quads && quads.empty();
    }

    void clear() {
      prim_in.clear();
      s1.reset();
      s2.reset();
      fifos = {};
      blocks.clear();
      be_block.reset();
      be_quads.reset();
      batch_sent = 0;
      be_phase = 0;
      quads.clear();
    #if VX_CFG_RASTER_EARLYZ_ENABLED
      ez.clear();
    #endif
    }

    void begin_cycle() {
      prim_in.begin_cycle();
      blocks.begin_cycle();
      quads.begin_cycle();
    #if VX_CFG_RASTER_EARLYZ_ENABLED
      ez.begin_cycle();
    #endif
    }
  };

  enum class FetchState : uint8_t { IDLE, TILE, PRIM };

  // ── Producer state reset (preserves DCRS) ──────────────────────────
  void reset_producer() {
    started_ = false;
    start_pending_ = false;
    fetch_state_ = FetchState::IDLE;
    req_valid_ = false;
    req_ = FetchReq();
    pending_prims_ = 0;
    addr_pipe_ = {};
    fetch_queue_.clear();
    fetch_slots_ = {};
    slots_busy_ = 0;
    slots_full_ = false;
    batch_idx_ = 0;
    fetch_rsp_buf_.clear();
    words_in_flight_ = 0;
    record_buf_.clear();
    edge_pipe_ = {};
    for (auto& slice : slices_) {
      slice.clear();
    }
    out_slice_rr_ = 0;
    out_buf_.clear();
    for (auto& q : parked_reqs_) {
      std::queue<RasterReq> empty;
      std::swap(q, empty);
    }
    drained_rr_ = 0;
  }

  void begin_cycle() {
    fetch_queue_.begin_cycle();
    fetch_rsp_buf_.begin_cycle();
    record_buf_.begin_cycle();
    for (auto& slice : slices_) {
      slice.begin_cycle();
    }
    out_buf_.begin_cycle();
    slots_full_ = (slots_busy_ == kFetchQueueSize);
    prims_full_ = (pending_prims_ >= kMaxPendingPrims);
  }

  bool producer_idle() const {
    if (fetch_state_ != FetchState::IDLE || req_valid_ || slots_busy_ != 0
     || !fetch_queue_.empty() || !fetch_rsp_buf_.empty() || !record_buf_.empty()) {
      return false;
    }
    for (const auto& a : addr_pipe_) {
      if (a) {
        return false;
      }
    }
    for (const auto& e : edge_pipe_) {
      if (e) {
        return false;
      }
    }
    for (const auto& slice : slices_) {
      if (!slice.idle()) {
        return false;
      }
    }
    return out_buf_.empty();
  }

  // ── Bin ownership ──────────────────────────────────────────────────
  // Static tile→core ownership: every cluster sees the identical global bin
  // list (same DCR config), so a bin index is device-consistent and each bin is
  // walked by exactly one cluster, into its owner core's queue.
  bool owns_bin(uint32_t bin) const {
    uint32_t owner = bin % total_cores_;
    return owner >= base_core_ && owner < base_core_ + cores_per_cluster_;
  }

  // The first bin at or after `bin` this cluster walks; tile_count_ if none.
  uint32_t next_owned_bin(uint32_t bin) const {
    while (bin < tile_count_ && !owns_bin(bin)) {
      ++bin;
    }
    return bin;
  }

  // ══════════════════════════════════════════════════════════════════
  // Fetch FSM + word-granular memory scheduler
  // ══════════════════════════════════════════════════════════════════

  void request_bin_header(uint32_t bin) {
    uint64_t hdr_addr = tbuf_addr_ + uint64_t(bin) * sizeof(graphics::rast_bin_header_t);
    req_ = FetchReq();
    req_.kind = FetchKind::TILE;
    req_.mask = (1u << kTileHeaderWords) - 1;
    for (uint32_t i = 0; i < kTileHeaderWords; ++i) {
      req_.addr[i] = hdr_addr + i * kWordSize;
    }
    req_valid_ = true;
    curr_bin_ = bin;
  }

  void request_pid(uint64_t pid_addr) {
    req_ = FetchReq();
    req_.kind = FetchKind::PID;
    req_.mask = 1;
    req_.addr[0] = pid_addr;
    req_valid_ = true;
  }

  void request_record(uint32_t pid) {
    uint64_t rec_addr = pbuf_addr_ + uint64_t(pid) * pbuf_stride_;
    req_ = FetchReq();
    req_.kind = FetchKind::PRIM;
    req_.mask = (1u << kPrimWords) - 1;
    req_.pid  = pid;
    for (uint32_t i = 0; i < kPrimWords; ++i) {
      req_.addr[i] = rec_addr + i * kWordSize;
    }
    req_valid_ = true;
  }

  // Leave the current bin: fetch the next owned header, or finish the walk.
  void advance_bin() {
    uint32_t next = next_owned_bin(curr_bin_ + 1);
    if (next < tile_count_) {
      request_bin_header(next);
      fetch_state_ = FetchState::TILE;
    } else {
      fetch_state_ = FetchState::IDLE;
    }
  }

  void start_walk() {
    started_     = true;
    tile_count_  = dcrs_.read(VX_DCR_RASTER_TILE_COUNT);
    tbuf_addr_   = uint64_t(dcrs_.read(VX_DCR_RASTER_TBUF_ADDR)) << 6;
    pbuf_addr_   = uint64_t(dcrs_.read(VX_DCR_RASTER_PBUF_ADDR)) << 6;
    pbuf_stride_ = dcrs_.read(VX_DCR_RASTER_PBUF_STRIDE);
    if (pbuf_stride_ == 0) {
      pbuf_stride_ = sizeof(graphics::rast_prim_t);
    }
    // sorted_pids follows the dense bin-header block; pids_offset is an
    // absolute index into it, so every bin's pid run is one base + off.
    pids_base_ = tbuf_addr_ + uint64_t(tile_count_) * sizeof(graphics::rast_bin_header_t);
    scissor_left_   = dcrs_.read(VX_DCR_RASTER_SCISSOR_X) & 0xffff;
    scissor_right_  = dcrs_.read(VX_DCR_RASTER_SCISSOR_X) >> 16;
    scissor_top_    = dcrs_.read(VX_DCR_RASTER_SCISSOR_Y) & 0xffff;
    scissor_bottom_ = dcrs_.read(VX_DCR_RASTER_SCISSOR_Y) >> 16;
    msaa_ = dcrs_.read(VX_DCR_RASTER_SAMPLES) == gfx_rast::RAST_MSAA_SAMPLES;

    uint32_t first = next_owned_bin(0);
    if (first < tile_count_) {
      request_bin_header(first);
      fetch_state_ = FetchState::TILE;
    }
  }

  // A bin header arrived: an empty bin is skipped, otherwise its pid list is
  // walked one word per request.
  void on_bin_header(const FetchRsp& rsp) {
    graphics::rast_bin_header_t hdr;
    std::memcpy(&hdr, rsp.data.data(), sizeof(hdr));
    uint32_t bin_size = 1u << kTileLogSize;
    curr_xloc_ = uint32_t(hdr.bin_x) * bin_size;
    curr_yloc_ = uint32_t(hdr.bin_y) * bin_size;
    if (hdr.pids_count > GFX_RASTER_MAX_PRIMS) {
      raster_pass_overflow("bin count", hdr.pids_count);
    }
    if (hdr.pids_count == 0) {
      advance_bin();
      return;
    }
    uint64_t pid_addr = pids_base_ + uint64_t(hdr.pids_offset) * sizeof(uint32_t);
    request_pid(pid_addr);
    pid_addr_  = pid_addr;
    pid_reqs_  = hdr.pids_count;
    pid_rsps_  = hdr.pids_count;
    fetch_state_ = FetchState::PRIM;
  }

  // A primitive's record words arrived — { scissor_x, scissor_y, edges[3][3]
  // [, depth plane] }, the head of rast_prim_t: build its walk context.
  PrimRef make_prim(const FetchRsp& rsp) const {
    const auto& w = rsp.data;
    auto ctx = std::make_shared<PrimCtx>();
    ctx->pid   = rsp.pid;
    ctx->owner = (curr_bin_ % total_cores_) - base_core_;
    ctx->xloc  = curr_xloc_;
    ctx->yloc  = curr_yloc_;
    for (uint32_t k = 0; k < 3; ++k) {
      uint32_t base = kPrimScissorWords + 3 * k;
      ctx->edges[k].x = graphics::FloatE::make(int32_t(w[base + 0]));
      ctx->edges[k].y = graphics::FloatE::make(int32_t(w[base + 1]));
      ctx->edges[k].z = graphics::FloatE::make(int32_t(w[base + 2]));
    }
    ctx->extents = compute_extents(ctx->edges);
    ctx->cbias   = graphics::vec3e_t{};
    if (msaa_) {
      ctx->cbias = graphics::vec3e_t{ (ctx->edges[0].x + ctx->edges[0].y) >> 1,
                                      (ctx->edges[1].x + ctx->edges[1].y) >> 1,
                                      (ctx->edges[2].x + ctx->edges[2].y) >> 1 };
    }
    ctx->scissor_left   = std::max(scissor_left_,   RAST_SCISSOR_LO(w[0]));
    ctx->scissor_right  = std::min(scissor_right_,  RAST_SCISSOR_HI(w[0]));
    ctx->scissor_top    = std::max(scissor_top_,    RAST_SCISSOR_LO(w[1]));
    ctx->scissor_bottom = std::min(scissor_bottom_, RAST_SCISSOR_HI(w[1]));
  #if VX_CFG_RASTER_EARLYZ_ENABLED
    uint32_t zbase = kPrimScissorWords + kPrimEdgeWords;
    ctx->zplane.x = graphics::FloatA::make(int32_t(w[zbase + 0]));
    ctx->zplane.y = graphics::FloatA::make(int32_t(w[zbase + 1]));
    ctx->zplane.z = graphics::FloatA::make(int32_t(w[zbase + 2]));
  #endif
    return ctx;
  }

  void fetch_step(bool start) {
    // ── Memory responses: gather words; a completed request leaves through
    //    the response buffer and frees its slot. The buffer takes one
    //    completed request per cycle.
    bool completed = false;
    for (uint32_t ch = 0; ch < kRcacheNumReqs; ++ch) {
      auto& rsp_ch = simobject_->rcache_rsp_in.at(ch);
      if (rsp_ch.empty()) {
        continue;
      }
      const auto& mrsp = rsp_ch.peek();
      uint32_t slot_id = uint32_t(mrsp.tag) / kFetchWords;
      uint32_t word    = uint32_t(mrsp.tag) % kFetchWords;
      auto& slot = fetch_slots_.at(slot_id);
      assert(slot.busy && (slot.pending & (1u << word)));
      bool completes = (slot.pending == (1u << word));
      if (completes && (completed || !fetch_rsp_buf_.ready())) {
        continue;
      }
      uint32_t value = 0;
      if (mrsp.data) {
        uint32_t offset = uint32_t(slot.req.addr[word] & ~kRcacheLineMask);
        std::memcpy(&value, mrsp.data->data() + offset, sizeof(value));
      }
      slot.rsp.data[word] = value;
      slot.pending &= ~(1u << word);
      --words_in_flight_;
      rsp_ch.pop();
      if (completes) {
        slot.rsp.kind = slot.req.kind;
        slot.rsp.pid  = slot.req.pid;
        fetch_rsp_buf_.push(slot.rsp, cycle_);
        completed = true;
        slot.busy = false;
        --slots_busy_;
      }
    }

    // ── Batch issue: the head request sends one batch of up to one word per
    //    rcache port each cycle; a batch with no requested word still takes
    //    its cycle unless it lies past the request's last word.
    if (fetch_queue_.valid(cycle_)) {
      uint32_t slot_id = fetch_queue_.front();
      const auto& req = fetch_slots_.at(slot_id).req;
      uint32_t base = batch_idx_ * kRcacheNumReqs;
      uint32_t batch_mask = (req.mask >> base) & ((1u << kRcacheNumReqs) - 1);
      bool can_send = true;
      for (uint32_t ch = 0; ch < kRcacheNumReqs; ++ch) {
        if ((batch_mask >> ch) & 1) {
          can_send &= !simobject_->rcache_req_out.at(ch).full();
        }
      }
      if (can_send) {
        for (uint32_t ch = 0; ch < kRcacheNumReqs; ++ch) {
          if (!((batch_mask >> ch) & 1)) {
            continue;
          }
          uint32_t word = base + ch;
          MemReq mreq;
          mreq.addr    = req.addr[word] & kRcacheLineMask;
          mreq.op      = MemOp::LD;
          mreq.tag     = slot_id * kFetchWords + word;
          mreq.hart_id = 0;
          mreq.uuid    = 0;
          simobject_->rcache_req_out.at(ch).send(mreq);
          ++perf_stats_.mem_reads;
          ++words_in_flight_;
        }
        uint32_t last_batch = (32 - __builtin_clz(req.mask) - 1) / kRcacheNumReqs;
        if (batch_idx_ == last_batch) {
          batch_idx_ = 0;
          fetch_queue_.pop();
        } else {
          ++batch_idx_;
        }
      }
    }

    // ── Fetch FSM ───────────────────────────────────────────────────
    const FetchRsp* rsp = fetch_rsp_buf_.valid(cycle_) ? &fetch_rsp_buf_.front() : nullptr;
    bool pid_rsp  = rsp && (rsp->kind == FetchKind::PID);
    bool prim_rsp = rsp && (rsp->kind == FetchKind::PRIM);

    bool req_ready = fetch_queue_.ready() && !slots_full_;
    bool pid_req   = req_valid_ && (req_.kind == FetchKind::PID);
    // A pid is requested only while its record has a guaranteed place downstream.
    bool req_fire  = req_valid_ && (!pid_req || !prims_full_) && req_ready;

    // The record request for a pid waits on the address multiply; a record
    // request the scheduler cannot take stalls the multiplier behind it.
    const auto& addr_out = addr_pipe_.back();
    bool prim_req_stall = req_valid_ && (req_.kind == FetchKind::PRIM) && !req_ready;
    bool addr_ready = !prim_req_stall || !addr_out;
    bool addr_fire  = addr_out && addr_ready;
    uint32_t addr_pid = addr_out ? *addr_out : 0;

    bool rsp_ready = (!pid_rsp || addr_ready) && (!prim_rsp || record_buf_.ready());
    bool prim_rsp_fire = prim_rsp && rsp_ready;

    if (req_fire) {
      push_fetch(req_);
      if (pid_req) {
        ++pending_prims_;
      }
    }

    if (addr_ready) {
      for (uint32_t i = kMulLatency - 1; i > 0; --i) {
        addr_pipe_[i] = addr_pipe_[i - 1];
      }
      addr_pipe_[0].reset();
      if (pid_rsp) {
        uint32_t pid = rsp->data[0];
        if (pid >= GFX_RASTER_MAX_PRIMS) {
          raster_pass_overflow("pid", pid);
        }
        addr_pipe_[0] = pid;
      }
    }

    if (prim_rsp_fire) {
      auto prim = make_prim(*rsp);
      DT(4, simobject_->name() << " mem-out: x=" << prim->xloc << ", y=" << prim->yloc
         << ", pid=" << prim->pid);
      record_buf_.push(std::move(prim), cycle_);
    }

    FetchState state = fetch_state_;
    bool tile_rsp = rsp && (state == FetchState::TILE);
    FetchRsp tile_hdr;
    if (tile_rsp) {
      tile_hdr = *rsp;
    }
    if (rsp && rsp_ready) {
      fetch_rsp_buf_.pop();
    }

    // Registered FSM update: a fired request clears unless a new one replaces it.
    if (req_fire) {
      req_valid_ = false;
    }
    switch (state) {
    case FetchState::IDLE:
      if (start) {
        start_walk();
      }
      break;
    case FetchState::TILE:
      if (tile_rsp) {
        on_bin_header(tile_hdr);
      }
      break;
    case FetchState::PRIM: {
      uint64_t pid_addr = pid_addr_;
      uint32_t pid_reqs = pid_reqs_;
      if (req_fire) {
        if (pid_req) {
          pid_addr_ += sizeof(uint32_t);
          --pid_reqs_;
        }
        if (pid_reqs > 1 || (pid_reqs == 1 && !pid_req)) {
          request_pid(pid_addr + (pid_req ? sizeof(uint32_t) : 0));
        }
      }
      if (addr_fire) {
        request_record(addr_pid);
      }
      if (prim_rsp_fire) {
        if (pid_rsps_ == 1) {
          advance_bin();
        }
        --pid_rsps_;
      }
      break;
    }
    }
  }

  void push_fetch(const FetchReq& req) {
    uint32_t slot_id = 0;
    while (fetch_slots_.at(slot_id).busy) {
      ++slot_id;
    }
    auto& slot = fetch_slots_.at(slot_id);
    slot.busy    = true;
    slot.pending = req.mask;
    slot.req     = req;
    slot.rsp     = FetchRsp();
    ++slots_busy_;
    fetch_queue_.push(slot_id, cycle_);
  }

  // ══════════════════════════════════════════════════════════════════
  // Edge stage: evaluates the edge functions at the bin origin, then routes
  // the primitive to its bin's slice. Holds when that slice cannot take it.
  // ══════════════════════════════════════════════════════════════════

  static uint32_t slice_of(const PrimCtx& prim) {
    return ((prim.xloc >> kTileLogSize) + (prim.yloc >> kTileLogSize)) % kNumSlices;
  }

  void edge_step() {
    auto& out = edge_pipe_.back();
    bool stall = out && !slices_.at(slice_of(**out)).prim_in.ready();
    if (stall) {
      return;
    }
    if (out) {
      slices_.at(slice_of(**out)).prim_in.push(*out, cycle_);
    }
    for (uint32_t i = kMulLatency - 1; i > 0; --i) {
      edge_pipe_[i] = std::move(edge_pipe_[i - 1]);
    }
    edge_pipe_[0].reset();
    if (record_buf_.valid(cycle_)) {
      edge_pipe_[0] = record_buf_.front();
      record_buf_.pop();
      --pending_prims_;
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // Tile evaluator (TE)
  // ══════════════════════════════════════════════════════════════════
  //
  // Per (bin, prim), 4 priority FIFOs (one per subtile index 2*i+j with
  // i=X-bit, j=Y-bit → TL, BL, TR, BR column-major), priority 0>1>2>3, plus a
  // "bypass" path that lets TL of the current subdivision be processed next
  // when no FIFO head is grantable.
  //
  // Per cycle: stage 2 emits (if block) or pushes its subs to FIFOs (skipping
  // F[0] on bypass); the arb sees FIFO heads BEFORE the push and picks the next
  // stage 1 (priority 0 > 1 > 2 > 3); the pipeline advances (s2 <= s1, s1 <=
  // arb pick / bypass / new primitive). The one-cycle delay between
  // subdivision and FIFO push is load-bearing — it lets older non-TL subtiles
  // get selected before TL-side descendants push their own F[0] entries. A new
  // primitive enters only on a cycle where stage 1 is empty and neither the
  // arb nor the bypass supplies it. A block that the block buffer cannot take
  // holds the whole walk.

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
                            const graphics::vec3e_t& cbias,
                            uint32_t tile_logsize) {
    graphics::FloatE z(0);
    uint32_t shift = tile_logsize + 1;  // tile size in log2 pixels
    return  (edge_eval.x + (extents.x << shift) - cbias.x) >= z
         && (edge_eval.y + (extents.y << shift) - cbias.y) >= z
         && (edge_eval.z + (extents.z << shift) - cbias.z) >= z;
  }

  // Stage-1 evaluation: nothing when the primitive misses the tile, the block
  // itself at block size, else the tile with its four subtiles.
  static std::optional<PipeEntry> make_pipe_entry(const TileWork& tile) {
    constexpr uint32_t kTopLog = kTileLogSize - 1;
    const graphics::vec3e_t* edges = tile.prim->edges;
    uint32_t tile_logsize = kTopLog - tile.level;
    if (!tile_overlaps(tile.edge_eval, tile.prim->extents, tile.prim->cbias, tile_logsize)) {
      return std::nullopt;
    }
    PipeEntry pe;
    pe.tile = tile;
    pe.is_block = (tile_logsize < kBlockLogSize);
    if (!pe.is_block) {
      uint32_t sub_size = 1u << tile_logsize;
      for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
          TileWork& s = pe.subs[2*i + j];
          s.x = tile.x + i * sub_size;
          s.y = tile.y + j * sub_size;
          s.level = tile.level + 1;
          s.prim = tile.prim;
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

  // The primitive's top tile: its bin, with the edges evaluated at the origin.
  static TileWork make_top_tile(const PrimRef& prim) {
    TileWork init;
    init.x = prim->xloc;
    init.y = prim->yloc;
    init.level = 0;
    const auto* e = prim->edges;
    init.edge_eval.x = e[0].x * int(prim->xloc) + e[0].y * int(prim->yloc) + e[0].z;
    init.edge_eval.y = e[1].x * int(prim->xloc) + e[1].y * int(prim->yloc) + e[1].z;
    init.edge_eval.z = e[2].x * int(prim->xloc) + e[2].y * int(prim->yloc) + e[2].z;
    init.prim = prim;
    return init;
  }

  void te_step(Slice& slice) {
    bool emit = slice.s2 && slice.s2->is_block;
    bool output_stall = emit && !slice.blocks.ready();
    if (!output_stall) {
      bool push_active = slice.s2 && !slice.s2->is_block;
      int arb_idx = -1;
      for (uint32_t i = 0; i < 4; ++i) {
        if (slice.fifos[i].out_valid) {
          arb_idx = int(i);
          break;
        }
      }
      // Stage 2 has subs to push AND no FIFO has anything to grant.
      bool bypass = push_active && (arb_idx < 0);

      std::optional<TileWork> next_s1;
      if (arb_idx >= 0) {
        next_s1 = std::move(slice.fifos[arb_idx].out);
        slice.fifos[arb_idx].out_valid = false;
      } else if (bypass) {
        next_s1 = slice.s2->subs[0];
      } else if (!slice.s1 && slice.prim_in.valid(cycle_)) {
        const auto& prim = slice.prim_in.front();
        DT(4, simobject_->name() << " te-in: x=" << prim->xloc << ", y=" << prim->yloc
           << ", pid=" << prim->pid);
        next_s1 = make_top_tile(prim);
        slice.prim_in.pop();
      }

      if (emit) {
        DT(4, simobject_->name() << " be-in: x=" << slice.s2->tile.x << ", y=" << slice.s2->tile.y
           << ", pid=" << slice.s2->tile.prim->pid);
        slice.blocks.push(slice.s2->tile, cycle_);
      }
      if (push_active) {
        for (uint32_t i = (bypass ? 1u : 0u); i < 4; ++i) {
          slice.fifos[i].push(slice.s2->subs[i], cycle_);
        }
      }

      if (slice.s1) {
        slice.s2 = make_pipe_entry(*slice.s1);
      } else {
        slice.s2.reset();
      }
      slice.s1 = std::move(next_s1);
    }

    // FIFO head refill (grantable from the next cycle on).
    for (auto& f : slice.fifos) {
      f.refill(cycle_);
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // Block evaluator (BE)
  // ══════════════════════════════════════════════════════════════════
  //
  // A block's kNumQuadsDim^2 quads are evaluated in row-major order
  // (i=jj*kNumQuadsDim+ii, ii=X-bit, jj=Y-bit → TL, TR, BL, BR) and grouped
  // into kOutputQuads-wide batches arbitrated by priority (batch 0 > 1...),
  // one batch per cycle. A batch is emitted only if any of its quads overlaps,
  // carrying all of its stamps (non-overlapping ones carry mask=0 but valid
  // pos_x/pos_y). The block holds the evaluator until its last batch leaves.

  // Per-pixel coverage + scissor test for a quad: a bit per pixel, and in
  // `smask` the pixel's sample coverage (bit 0 alone at one sample). Under 4x
  // multisampling a pixel is covered when any of its samples is.
  static uint32_t compute_quad(uint32_t qx, uint32_t qy,
                               const graphics::vec3e_t& edge_eval_corner,
                               const PrimCtx& prim, bool msaa, uint32_t& smask) {
    const graphics::vec3e_t* edges = prim.edges;
    graphics::FloatE z(0);
    uint32_t mask = 0;
    smask = 0;
    // Vulkan top-left fill rule: a sample exactly on an edge (value == 0) is
    // covered only if that edge is a top or left edge (gradient a > 0, or
    // a == 0 && b > 0); otherwise the boundary sample belongs to the abutting
    // triangle. Gradients (a, b) = (edges[k].x, edges[k].y) are constant across
    // the quad, so classify once. Keeps shared edges covered exactly once.
    bool tl0 = (edges[0].x > z) || (edges[0].x == z && edges[0].y > z);
    bool tl1 = (edges[1].x > z) || (edges[1].x == z && edges[1].y > z);
    bool tl2 = (edges[2].x > z) || (edges[2].x == z && edges[2].y > z);
    for (uint32_t pj = 0; pj < 2; ++pj) {
      for (uint32_t pi = 0; pi < 2; ++pi) {
        auto ee0 = edge_eval_corner.x + edges[0].x * int(pi) + edges[0].y * int(pj);
        auto ee1 = edge_eval_corner.y + edges[1].x * int(pi) + edges[1].y * int(pj);
        auto ee2 = edge_eval_corner.z + edges[2].x * int(pi) + edges[2].y * int(pj);
        uint32_t px = qx + pi;
        uint32_t py = qy + pj;
        bool in0 = tl0 ? (ee0 >= z) : (ee0 > z);
        bool in1 = tl1 ? (ee1 >= z) : (ee1 > z);
        bool in2 = tl2 ? (ee2 >= z) : (ee2 > z);
        bool in_scissor = (px >= prim.scissor_left) && (px <  prim.scissor_right)
                       && (py >= prim.scissor_top)  && (py <  prim.scissor_bottom);
        uint32_t samples = 0;
        if (in_scissor) {
          samples = msaa ? gfx_rast::rast_sample_mask(edges, graphics::vec3e_t{ ee0, ee1, ee2 })
                         : uint32_t(in0 && in1 && in2);
        }
        if (samples) {
          mask |= 1u << (pj * 2 + pi);
        }
        smask |= samples << (4 * (pj * 2 + pi));
      }
    }
    return mask;
  }

  static QuadEval evaluate_block(const TileWork& block, bool msaa) {
    const graphics::vec3e_t* edges = block.prim->edges;
    QuadEval qe;
    qe.prim = block.prim;
    for (uint32_t i = 0; i < kPerBlockQuads; ++i) {
      uint32_t ii = i % kNumQuadsDim;
      uint32_t jj = i / kNumQuadsDim;
      qe.qx_pix[i] = block.x + 2 * ii;
      qe.qy_pix[i] = block.y + 2 * jj;
      // Edge values at quad corner = block_corner + 2*ii*ex + 2*jj*ey
      graphics::vec3e_t corner;
      corner.x = block.edge_eval.x + edges[0].x * int(2 * ii) + edges[0].y * int(2 * jj);
      corner.y = block.edge_eval.y + edges[1].x * int(2 * ii) + edges[1].y * int(2 * jj);
      corner.z = block.edge_eval.z + edges[2].x * int(2 * ii) + edges[2].y * int(2 * jj);
      qe.mask[i] = compute_quad(qe.qx_pix[i], qe.qy_pix[i], corner, *block.prim, msaa, qe.smask[i]);
    }
    return qe;
  }

  static uint32_t overlapping_batches(const QuadEval& qe) {
    uint32_t valid = 0;
    for (uint32_t i = 0; i < kPerBlockQuads; ++i) {
      if (qe.mask[i] != 0) {
        valid |= 1u << (i / kOutputQuads);
      }
    }
    return valid;
  }

  void be_step(Slice& slice) {
    bool stall = false;
    if (slice.be_quads) {
      const QuadEval& qe = *slice.be_quads;
      uint32_t batch_valid = overlapping_batches(qe);
      uint32_t pending = batch_valid & ~slice.batch_sent;
      if (pending != 0) {
        uint32_t b = __builtin_ctz(pending);
        bool fire = slice.quads.ready();
        uint32_t sent_n = slice.batch_sent | (fire ? (1u << b) : 0u);
        bool sent_all = (sent_n == batch_valid);
        stall = !fire || !sent_all;
        if (fire) {
          QuadBatch batch;
          batch.prim = qe.prim;
          for (uint32_t q = 0; q < kOutputQuads; ++q) {
            uint32_t idx = b * kOutputQuads + q;
            if (idx >= kPerBlockQuads) {
              break;
            }
            batch.stamps[q].pos_mask = encode_pos_mask(qe.qx_pix[idx] >> 1, qe.qy_pix[idx] >> 1, qe.mask[idx]);
            batch.stamps[q].pid      = qe.prim->pid;
            batch.stamps[q].smask    = qe.smask[idx];
          }
          DT(4, simobject_->name() << " be-out: x=" << (qe.qx_pix[b * kOutputQuads] >> 1)
             << ", y=" << (qe.qy_pix[b * kOutputQuads] >> 1) << ", pid=" << qe.prim->pid);
          slice.quads.push(std::move(batch), cycle_);
          slice.batch_sent = sent_all ? 0 : sent_n;
        }
      }
    }
    if (stall) {
      return;
    }
    if (slice.be_block) {
      slice.be_quads = evaluate_block(*slice.be_block, msaa_);
    } else {
      slice.be_quads.reset();
    }
    slice.be_block.reset();
    // Under 4x multisampling a block holds the evaluator's input for a cycle
    // that computes sample 0's edge offsets and then a cycle per sample; it
    // leaves the block buffer with its last sample.
    if (slice.blocks.valid(cycle_)) {
      if (msaa_ && slice.be_phase < gfx_rast::RAST_MSAA_SAMPLES) {
        ++slice.be_phase;
      } else {
        slice.be_block = slice.blocks.front();
        slice.blocks.pop();
        slice.be_phase = 0;
      }
    }
  }

  // Round-robin the slices' quad batches onto the shared output buffer, one
  // batch per cycle.
  void drain_slices() {
    if (!out_buf_.ready()) {
      return;
    }
    for (uint32_t n = 0; n < kNumSlices; ++n) {
      uint32_t s = (out_slice_rr_ + n) % kNumSlices;
    #if VX_CFG_RASTER_EARLYZ_ENABLED
      if (earlyz_active()) {
        auto& ez = slices_.at(s).ez;
        if (ez.state == EzState::SEND && !ez_wave_empty(ez)) {
          out_buf_.push(ez_narrowed_wave(ez), cycle_);
          ez.sent = true;
          out_slice_rr_ = (s + 1) % kNumSlices;
          break;
        }
        continue;
      }
    #endif
      auto& quads = slices_.at(s).quads;
      if (quads.valid(cycle_)) {
        out_buf_.push(quads.front(), cycle_);
        quads.pop();
        out_slice_rr_ = (s + 1) % kNumSlices;
        break;
      }
    }
  }

#if VX_CFG_RASTER_EARLYZ_ENABLED

  // ── Early-Z: narrow each wave's coverage against committed depth ──
  // Tests each covered sample's plane depth against the depth buffer and drops
  // a pixel only when every sample it covers is STRICTLY behind
  // (earlyz_occluded). The plane depth is bit-identical to the FS late-Z, and
  // the strict-behind (reflexive-relaxed) test can never drop a visible
  // fragment, so enabling early-Z is image-identical to the ROP-only path. A
  // pixel that keeps any sample keeps its whole mask: the shader sees the
  // rasterized coverage (gl_SampleMaskIn, the centroid), before the depth test.
  //
  // The stage holds one wave: three cycles evaluate its plane, then
  // a request per sample goes to the memory scheduler while earlier samples'
  // responses are compared, and the narrowed wave leaves (a fully culled one is
  // dropped). A response is compared once its sample's candidates are
  // registered: at once when they already are, else a cycle later. The committed
  // depth is the ocache's word when the response arrives; it is not causally
  // pinned to this fragment, which the strict-behind rule makes harmless.

  bool earlyz_active() const {
    return earlyz_safe_ && depth_d24_;
  }

  static bool ez_wave_empty(const EarlyZ& ez) {
    for (uint32_t m : ez.mask) {
      if (m != 0) {
        return false;
      }
    }
    return true;
  }

  // A pixel keeps its rasterized mask while any of its samples survives.
  static QuadBatch ez_narrowed_wave(const EarlyZ& ez) {
    QuadBatch out = ez.wave;
    for (uint32_t q = 0; q < kOutputQuads; ++q) {
      auto& st = out.stamps[q];
      for (uint32_t i = 0; i < 4; ++i) {
        if (ez.mask[q * 4 + i] == 0) {
          st.pos_mask &= ~(1u << i);
          st.smask    &= ~(0xfu << (4 * i));
        }
      }
    }
    return out;
  }

  // Pixel p's position: X = pos_x*2 + (i&1), Y = pos_y*2 + (i>>1).
  static void ez_pixel_pos(const EarlyZ& ez, uint32_t p, int& X, int& Y) {
    const auto& st = ez.wave.stamps[p / 4];
    uint32_t qx = (st.pos_mask >> 4) & ((1u << kPosBits) - 1u);
    uint32_t qy = (st.pos_mask >> (4 + kPosBits)) & ((1u << kPosBits) - 1u);
    X = int(qx) * 2 + int(p & 1u);
    Y = int(qy) * 2 + int((p >> 1) & 1u);
  }

  uint64_t ez_tag(uint32_t slice, uint32_t slot, uint32_t lane) const {
    return (uint64_t(slice) * kEzQueueSize + slot) * kEzPixels + lane;
  }

  // The scheduler: responses fill their request, a completed request leaves
  // through its response buffer (one per slice per cycle); the head request of
  // each slice issues a batch of a word per ocache port, slices taking turns
  // for a port.
  void ez_mem_step() {
    std::array<bool, kNumSlices> completed{};
    for (uint32_t ch = 0; ch < kOcacheNumReqs; ++ch) {
      auto& rsp_ch = simobject_->ocache_rsp_in.at(ch);
      if (rsp_ch.empty()) {
        continue;
      }
      const auto& mrsp = rsp_ch.peek();
      uint32_t lane  = uint32_t(mrsp.tag % kEzPixels);
      uint32_t slot  = uint32_t((mrsp.tag / kEzPixels) % kEzQueueSize);
      uint32_t s     = uint32_t(mrsp.tag / (uint64_t(kEzPixels) * kEzQueueSize));
      auto& ez = slices_.at(s).ez;
      auto& sl = ez.slots.at(slot);
      if (!sl.busy || !sl.waiting.test(lane)) {
        rsp_ch.pop();   // a read of a walk the producer has since reset
        continue;
      }
      bool completes = (sl.waiting.count() == 1);
      if (completes && (completed.at(s) || !ez.rsp_buf.ready())) {
        continue;
      }
      uint32_t value = 0;
      if (mrsp.data) {
        uint32_t offset = uint32_t(sl.addr[lane] & ~kOcacheLineMask);
        std::memcpy(&value, mrsp.data->data() + offset, sizeof(value));
      } else {
        cluster_->processor()->ram()->read(&value, sl.addr[lane], sizeof(value));
      }
      sl.data[lane] = value;
      sl.waiting.reset(lane);
      rsp_ch.pop();
      if (completes) {
        ez.rsp_buf.push(slot, cycle_);
        completed.at(s) = true;
      }
    }

    uint32_t ports_used = 0;
    for (uint32_t n = 0; n < kNumSlices; ++n) {
      uint32_t s = (ez_issue_rr_ + n) % kNumSlices;
      auto& ez = slices_.at(s).ez;
      if (!ez.req_queue.valid(cycle_)) {
        continue;
      }
      uint32_t slot = ez.req_queue.front();
      const auto& sl = ez.slots.at(slot);
      uint32_t base = ez.batch_idx * kOcacheNumReqs;
      uint32_t batch_mask = 0;
      for (uint32_t ch = 0; ch < kOcacheNumReqs && base + ch < kEzPixels; ++ch) {
        batch_mask |= uint32_t(sl.lanes.test(base + ch)) << ch;
      }
      bool can_send = true;
      for (uint32_t ch = 0; ch < kOcacheNumReqs; ++ch) {
        if ((batch_mask >> ch) & 1) {
          can_send &= !((ports_used >> ch) & 1) && !simobject_->ocache_req_out.at(ch).full();
        }
      }
      if (!can_send) {
        continue;
      }
      for (uint32_t ch = 0; ch < kOcacheNumReqs; ++ch) {
        if (!((batch_mask >> ch) & 1)) {
          continue;
        }
        uint32_t lane = base + ch;
        MemReq mreq;
        mreq.addr    = sl.addr[lane] & kOcacheLineMask;
        mreq.op      = MemOp::LD;
        mreq.tag     = uint32_t(ez_tag(s, slot, lane));
        mreq.hart_id = 0;
        mreq.uuid    = 0;
        simobject_->ocache_req_out.at(ch).send(mreq, kEzReqDelay);
        ++perf_stats_.mem_reads;
      }
      ports_used |= batch_mask;
      uint32_t last_lane = kEzPixels - 1;
      while (!sl.lanes.test(last_lane)) {
        --last_lane;
      }
      uint32_t last_batch = last_lane / kOcacheNumReqs;
      if (ez.batch_idx == last_batch) {
        ez.batch_idx = 0;
        ez.req_queue.pop();
      } else {
        ++ez.batch_idx;
      }
    }
    ez_issue_rr_ = (ez_issue_rr_ + 1) % kNumSlices;
  }

  void ez_step(Slice& slice) {
    auto& ez = slice.ez;
    switch (ez.state) {
    case EzState::IDLE:
      if (earlyz_active() && slice.quads.valid(cycle_)) {
        ez.wave = slice.quads.front();
        slice.quads.pop();
        for (uint32_t p = 0; p < kEzPixels; ++p) {
          ez.mask[p] = (ez.wave.stamps[p / 4].smask >> (4 * (p % 4))) & 0xfu;
        }
        ez.state = EzState::EVAL1;
      }
      break;
    case EzState::EVAL1: {
      const auto& zp = ez.wave.prim->zplane;
      for (uint32_t k = 0; k < gfx_rast::RAST_MSAA_SAMPLES; ++k) {
        ez.zoff[k] = msaa_ ? earlyz_sample_offset(zp, k) : 0;
      }
      ez.issue_k = 0;
      ez.state = EzState::EVAL1B;
      break;
    }
    case EzState::EVAL1B:
      ez.state = EzState::EVAL2;
      break;
    case EzState::EVAL2:
      ez.cand_k     = 0;
      ez.issue_done = false;
      ez.state      = EzState::RUN;
      break;
    case EzState::RUN:
      ez_run(slice);
      break;
    case EzState::SEND:
      if (ez.sent || ez_wave_empty(ez)) {
        ez.sent  = false;
        ez.state = EzState::IDLE;
      }
      break;
    }
  }

  void ez_run(Slice& slice) {
    auto& ez = slice.ez;
    const uint32_t samples = msaa_ ? gfx_rast::RAST_MSAA_SAMPLES : 1u;
    const bool issue_last = (ez.issue_k == samples - 1);

    // Issue: the pixels covering the next sample, one request; a sample no
    // pixel covers is skipped without one.
    EzLanes issue_cov;
    for (uint32_t p = 0; p < kEzPixels; ++p) {
      issue_cov[p] = (ez.mask[p] >> ez.issue_k) & 1;
    }
    bool req_fire = false;
    if (!ez.issue_done && issue_cov.any() && ez.req_queue.ready() && !ez.slots_full) {
      uint32_t slot = 0;
      while (ez.slots.at(slot).busy) {
        ++slot;
      }
      auto& sl = ez.slots.at(slot);
      sl.busy    = true;
      sl.sample  = ez.issue_k;
      sl.lanes   = issue_cov;
      sl.waiting = issue_cov;
      for (uint32_t p = 0; p < kEzPixels; ++p) {
        int X, Y;
        ez_pixel_pos(ez, p, X, Y);
        sl.addr[p] = zbuf_base_ + uint64_t(ez.issue_k) * zbuf_sample_stride_
                   + uint64_t(Y) * zbuf_pitch_ + uint64_t(X) * 4;
      }
      ++ez.slots_busy;
      ez.req_queue.push(slot, cycle_);
      req_fire = true;
    }
    bool issue_step = !ez.issue_done && (req_fire || issue_cov.none());

    // Compare: a completed request whose sample's candidates are registered.
    bool rsp_valid = ez.rsp_buf.valid(cycle_);
    uint32_t rsp_k = 0;
    bool rsp_fire = false;
    if (rsp_valid) {
      uint32_t slot = ez.rsp_buf.front();
      auto& sl = ez.slots.at(slot);
      rsp_k = sl.sample;
      if (ez.cand_k == rsp_k) {
        const auto& zp = ez.wave.prim->zplane;
        for (uint32_t p = 0; p < kEzPixels; ++p) {
          if (!sl.lanes.test(p)) {
            continue;
          }
          int X, Y;
          ez_pixel_pos(ez, p, X, Y);
          uint32_t cand   = earlyz_plane_depth(zp, X, Y, ez.zoff[rsp_k]);
          uint32_t stored = sl.data[p] & OM_DEPTH_MASK;
          ++perf_stats_.earlyz_tested;
          if (earlyz_occluded(depth_func_, cand, stored)) {
            ez.mask[p] &= ~(1u << rsp_k);
            ++perf_stats_.earlyz_culled;
          }
        }
        sl.busy = false;
        --ez.slots_busy;
        ez.rsp_buf.pop();
        rsp_fire = true;
      }
    }

    // The candidates registered next: a waiting response's sample, else the
    // oldest still in flight, else the one being read.
    uint32_t in_flight = ez.pending & ~(rsp_fire ? (1u << rsp_k) : 0u);
    uint32_t pending_n = in_flight | (req_fire ? (1u << ez.issue_k) : 0u);
    uint32_t next_k = in_flight ? uint32_t(__builtin_ctz(in_flight)) : ez.issue_k;
    ez.cand_k  = (rsp_valid && !rsp_fire) ? rsp_k : next_k;
    ez.pending = pending_n;
    if (issue_step && !issue_last) {
      ++ez.issue_k;
    }
    ez.issue_done = ez.issue_done || (issue_step && issue_last);
    if (ez.issue_done && pending_n == 0) {
      ez.state = EzState::SEND;
    }
  }

#endif

  // ── Serve consumers ────────────────────────────────────────────────
  // A request waits in its core's queue until the output buffer's head batch
  // belongs to that core; batches leave strictly in emission order. Once the
  // producer has drained, waiting requests get the default pos_mask=0
  // sentinel, one per cycle, and their cores exit the raster loop; total
  // coverage across all cores == the full scene, once.
  void serve_consumers() {
    auto& req_ch = simobject_->raster_req_in.at(0);
    auto& rsp_ch = simobject_->raster_rsp_out.at(0);
    if (!req_ch.empty()) {
      const auto& req = req_ch.peek();
      parked_reqs_.at(req.core_id - base_core_).push(req);
      req_ch.pop();
    }

    bool head_valid = out_buf_.valid(cycle_);
    if (rsp_ch.full()) {
      if (head_valid) {
        ++perf_stats_.stall_cycles;
      }
      return;
    }

    if (head_valid) {
      const QuadBatch& batch = out_buf_.front();
      auto& parked = parked_reqs_.at(batch.prim->owner);
      if (parked.empty()) {
        ++perf_stats_.stall_cycles;
        return;
      }
      const auto& req = parked.front();
      RasterRsp rsp(req);
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        if (!(req.tmask_bits & (1u << t))) {
          continue;
        }
        rsp.stamps[t] = batch.stamps[t];
      }
      DT(4, simobject_->name() << " out: core=" << req.core_id << ", pid=" << batch.prim->pid);
      rsp_ch.send(rsp);
      parked.pop();
      out_buf_.pop();
      return;
    }

    if (!started_ || !producer_idle()) {
      return;
    }
    for (uint32_t n = 0; n < cores_per_cluster_; ++n) {
      uint32_t c = (drained_rr_ + n) % cores_per_cluster_;
      auto& parked = parked_reqs_.at(c);
      if (!parked.empty()) {
        rsp_ch.send(RasterRsp(parked.front()));
        parked.pop();
        drained_rr_ = (c + 1) % cores_per_cluster_;
        break;
      }
    }
  }

  // ── Members ─────────────────────────────────────────────────────────
  RasterCore*                simobject_;
  Cluster*                   cluster_;
  RasterDCRS       dcrs_;

#if VX_CFG_RASTER_EARLYZ_ENABLED
  // Early-Z config snooped from the OM depth DCRs (the depth buffer is
  // shared with the ROP). Gated by VX_DCR_OM_EARLYZ_SAFE, which the driver sets
  // only when the FS has no depth-export and the func is monotonic (LESS/LEQUAL).
  // The whole stage follows the same knob the datapath does, so a configuration
  // without it does not model a cull it would never perform.
  uint32_t                   earlyz_safe_ = 0;
  // The candidate is the 24-bit unorm plane, so any other depth target never culls.
  bool                       depth_d24_ = true;
  uint64_t                   zbuf_base_   = 0;   // byte address (ZBUF_ADDR << 6)
  uint32_t                   zbuf_pitch_  = 0;
  uint32_t                   zbuf_sample_stride_ = 0;   // bytes between sample images
  uint32_t                   depth_func_  = VX_OM_DEPTH_FUNC_ALWAYS;
  uint32_t                   ez_issue_rr_ = 0;
#endif

  // Fragment-shader dispatch descriptor (RASTER_FRAG_* DCRs). Persists across
  // the per-launch reset (re-supplied by each draw's DCR sequence); used to arm
  // each owned core's scheduler for the draw.
  uint64_t                   frag_entry_ = 0;
  uint64_t                   frag_param_ = 0;
  // Per-draw fragment-phase arm, triggered by the KMU's delegated launch:
  //   arm_pending_ : one-shot request set by frame_kick (the draw's grid-less
  //                  launch, delivered after the per-launch reset and the whole
  //                  DCR sequence); consumed on the next tick.
  //   frag_armed_  : the owned distributors are armed for this draw; cleared by
  //                  the last done-link report, releasing the next draw's kick.
  // Triggered by the real per-draw launch — NOT level-triggered on persistent
  // DCR state and NOT gated on the TILE_COUNT poll — so a fully-culled draw arms,
  // drains to zero quads, and cannot leave a stale arm that re-fires during a
  // later draw's compute front end (which would rasterize the reused pool's stale
  // tiles). The arm cannot fire during the next draw's binning launches either,
  // because those launches carry non-empty grids and produce no kick.
  bool                       arm_pending_ = false;
  bool                       frag_armed_  = false;
  // Distributors still armed this draw (counted down by done-link reports).
  uint32_t                   fwd_live_    = 0;

  // Static screen-space tile→core ownership (FWD). Each global bin index is owned
  // by exactly one core (bin % total_cores_); this cluster's RasterCore walks
  // only bins whose owner lies in [base_core_, base_core_ + cores_per_cluster_),
  // emitting their quads to that owner core. This makes every covered pixel
  // rasterized once across the whole device, with no multi-cluster
  // double-blend and no within-cluster shared-queue work-stealing.
  uint32_t                   cores_per_cluster_;
  uint32_t                   base_core_;
  uint32_t                   total_cores_;

  // Walk configuration, latched from the DCRs when the walk starts.
  uint32_t                   tile_count_  = 0;
  uint64_t                   tbuf_addr_   = 0;
  uint64_t                   pbuf_addr_   = 0;
  uint32_t                   pbuf_stride_ = 0;
  uint64_t                   pids_base_   = 0;
  // 4x multisampling (VX_DCR_RASTER_SAMPLES).
  bool                       msaa_ = false;
  // Draw-wide scissor; every primitive narrows it to its record's own rectangle.
  uint32_t                   scissor_left_   = 0;
  uint32_t                   scissor_top_    = 0;
  uint32_t                   scissor_right_  = 0;
  uint32_t                   scissor_bottom_ = 0;

  // Fetch FSM.
  bool                       started_       = false;  // the walk has begun this frame
  bool                       start_pending_ = false;  // start the walk next cycle
  FetchState                 fetch_state_   = FetchState::IDLE;
  bool                       req_valid_     = false;
  FetchReq                   req_;
  uint32_t                   curr_bin_  = 0;
  uint32_t                   curr_xloc_ = 0;
  uint32_t                   curr_yloc_ = 0;
  uint64_t                   pid_addr_  = 0;   // next pid word of the bin
  uint32_t                   pid_reqs_  = 0;   // pids of the bin still to request
  uint32_t                   pid_rsps_  = 0;   // records of the bin still to arrive
  uint32_t                   pending_prims_ = 0;  // pids requested, records not yet in the edge stage
  bool                       prims_full_    = false;
  std::array<std::optional<uint32_t>, kMulLatency> addr_pipe_;  // pid -> record address

  // Memory scheduler.
  StageQueue<uint32_t>                         fetch_queue_;   // requests (slot ids) awaiting issue
  std::array<FetchSlot, kFetchQueueSize>       fetch_slots_;
  uint32_t                                     slots_busy_ = 0;
  bool                                         slots_full_ = false;
  uint32_t                                     batch_idx_  = 0;  // head request's next batch
  StageQueue<FetchRsp>                         fetch_rsp_buf_;
  uint32_t                                     words_in_flight_ = 0;

  // Fetched records, then the edge stage.
  StageQueue<PrimRef>                          record_buf_;
  std::array<std::optional<PrimRef>, kMulLatency> edge_pipe_;

  // Walk pipelines and the shared output.
  std::vector<Slice>                           slices_;
  uint32_t                                     out_slice_rr_ = 0;
  StageQueue<QuadBatch>                        out_buf_;

  // Requests waiting for their core's next batch, indexed by local core id
  // (global core id - base_core_).
  std::vector<std::queue<RasterReq>>           parked_reqs_;
  uint32_t                                     drained_rr_ = 0;

  uint64_t                                     cycle_;
  RasterCore::PerfStats                        perf_stats_;
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
#if VX_CFG_RASTER_EARLYZ_ENABLED
  , ocache_req_out(kOcacheNumReqs, this)
  , ocache_rsp_in(kOcacheNumReqs, this)
#endif
  , fwd_arm_out(NUM_SOCKETS * VX_CFG_SOCKET_SIZE, this)
  , fwd_done_in(this)
{
  impl_ = new Impl(this, cluster);
  fwd_done_in.bind(this, &RasterCore::on_fwd_done);
}

RasterCore::~RasterCore() {
  delete impl_;
}

void RasterCore::on_reset() { impl_->reset(); }
void RasterCore::on_tick()  { impl_->tick(); }
void RasterCore::on_fwd_done(const FwdDone& msg) { impl_->on_fwd_done(msg); }

int RasterCore::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

void RasterCore::om_dcr_snoop(uint32_t addr, uint32_t value) {
  impl_->om_dcr_snoop(addr, value);
}

void RasterCore::set_frag_descriptor(uint64_t frag_entry, uint64_t frag_param) {
  impl_->set_frag_descriptor(frag_entry, frag_param);
}

void RasterCore::frame_kick() {
  impl_->frame_kick();
}

const RasterCore::PerfStats& RasterCore::perf_stats() const {
  return impl_->perf_stats();
}
