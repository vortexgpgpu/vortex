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

#include "local_mem.h"
#include "mem_block_pool.h"
#include <mem.h>
#include <bitmanip.h>
#include <vector>
#include <deque>
#include <cstring>
#if VX_CFG_EXT_A_ENABLED
#include "amo_ops.h"
#include "amo_unit.h"
#endif

using namespace vortex;

// A request's timing, in cycles from the one its engine takes it in:
//   0  the engine takes the request into its input queue of two; it takes one
//      only in a cycle that starts with at most one queued
//   1  the lanes' words are compared and the request moves on to be issued
//   2  its first pass is issued to the bank registers
//   3  the banks are accessed
//   4  each served lane takes its bank's word
//   5  the response reaches its block's response register
//   6  the response register goes to the LSU block
// With several LSU blocks, the block an engine takes from is chosen the cycle
// before (see arbitrate()), and a second, read-only engine works the banks'
// second port, so two warps' reads go through each cycle. Stores and atomics
// take the first engine alone: one enters it only once the read engine has
// drained, and the read engine takes no read while a store or atomic sits in
// the first engine, while one waited at an input the cycle before, or in a
// cycle the first engine may be taking one, so every write keeps its order
// against every read.
// The model does a pass's bank accesses when it is issued. Every access keeps
// its order relative to the others, so this is the order the banks see.
class LocalMem::Impl {
protected:
  // a pass issued at cycle t has its response at the response queue's head
  // at t + 3, from where the block's response register takes it
  static constexpr uint32_t RSP_LATENCY = 3;
  // an engine's response slots, held from the pass until the block takes the
  // response from its response register
  static constexpr uint32_t RSP_QUEUE_SIZE = 8;
#if VX_CFG_EXT_A_ENABLED
  // An atomic reads its word, registers it, then writes the new value back:
  // the two cycles after its pass the banks take nothing else.
  static constexpr uint32_t AMO_STALL_CYCLES = 2;
#endif

  struct Request {
    LsuReq   req;
    uint32_t block;
    Request() : req(0), block(0) {}
  };

  struct RspEntry {
    uint64_t ready;
    uint32_t block;
    LsuRsp   rsp;
  };

  static constexpr uint32_t S0_SIZE = 2;

  struct Engine {
    std::deque<Request> s0;
    bool        s1_valid = false;
    Request     s1;
    BitVector<> pending;
    // the next pass's bank winners are known: a pass that leaves lanes
    // pending is followed by a cycle that finds the next ones
    bool        ready = false;
    uint32_t    phase = 0; // a wide load's word-phase in service
    std::deque<RspEntry> rsp_queue;
  };

  LocalMem* simobject_;
  Config    config_;
  RAM       ram_;
  uint32_t  addr_bits_;
  uint32_t  lg2_word_;
#if VX_CFG_EXT_A_ENABLED
  AmoUnit   amo_unit_;
#endif
  uint32_t  stall_cycles_;
  std::vector<Engine> engines_;
  struct RspReg {
    bool     valid = false;
    uint32_t engine = 0;
    LsuRsp   rsp;
    RspReg() : rsp(0) {}
  };
  std::vector<RspReg> rsp_regs_; // per block
  std::vector<uint32_t> arb_last_; // per engine, its round-robin arbiter's last choice
  struct Selection {
    bool     valid = false;
    uint32_t block = 0;
    bool     read = false; // the chosen block's request is known to be a plain read
  };
  Selection a_sel_;
  Selection b_sel_;
  bool      write_waiting_r_;
  bool      dma_slot_;
  uint32_t  dma_last_grant_;
  mutable PerfStats perf_stats_;

  uint64_t to_local_addr(uint64_t addr) const {
    return bit_getw(addr, 0, addr_bits_ - 1);
  }

  // A bank serves one word per cycle; byte addresses within one word are the
  // same access as far as the bank is concerned.
  uint64_t bank_word(uint64_t addr) const {
    return to_local_addr(addr) >> lg2_word_;
  }

  uint32_t bank_of(uint64_t addr) const {
    return (uint32_t)(bank_word(addr) & ((1u << config_.B) - 1));
  }

  static bool is_plain_read(const LsuReq& req) {
    return !req.is_write() && !req.is_amo();
  }

  static bool holds_write(const Engine& e) {
    for (auto& r : e.s0) {
      if (!is_plain_read(r.req)) {
        return true;
      }
    }
    return e.s1_valid && !is_plain_read(e.s1.req);
  }

  // The lanes one pass serves: each bank goes to its lowest pending lane, and
  // a plain read also serves every pending lane reading the same word. A
  // store or an atomic takes its bank alone, so lanes writing one word land
  // in lane order and the highest lane's bytes are the ones left.
  // A wide load's word-phase reads at a fixed byte offset from each lane's
  // aligned group, so its conflicts are those of the group addresses.
  BitVector<> select_pass(const LsuReq& req, const BitVector<>& pending, uint32_t offset) const {
    uint32_t num_banks = 1u << config_.B;
    std::vector<int32_t> winner(num_banks, -1);
    BitVector<> served(pending.size());
    bool join = is_plain_read(req);
    for (uint32_t i = 0; i < pending.size(); ++i) {
      if (!pending.test(i)) {
        continue;
      }
      uint64_t addr = req.addrs.at(i) + offset;
      uint32_t b = this->bank_of(addr);
      if (winner.at(b) == -1) {
        winner.at(b) = i;
        served.set(i);
      } else if (join && this->bank_word(addr) == this->bank_word(req.addrs.at(winner.at(b)) + offset)) {
        served.set(i);
      }
    }
    return served;
  }

  // Do one lane's bank access and return the word the lane gets back, if any.
  std::shared_ptr<mem_block_t> access(const LsuReq& req, uint32_t i) {
    uint64_t la = to_local_addr(req.addrs.at(i));
    uint64_t line_addr = la & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
#if VX_CFG_EXT_A_ENABLED
    uint32_t hart_id = make_hart_id(req.cid, req.wid, req.tids.at(i));
    if (req.is_amo()) {
      // Shared-memory atomics read-modify-write in place and return the old
      // word; LR/SC reservations are kept per hart.
      const uint32_t bmask    = VX_CFG_MEM_BLOCK_SIZE - 1;
      const uint32_t byte_off = (uint32_t)(la & bmask);
      const uint64_t byteen   = req.byteen.at(i);
      const uint32_t wbytes   = byteen ? (uint32_t)__builtin_popcountll(byteen) : 4u;
      const uint8_t  width    = (wbytes >= 8) ? 3 : 2;
      const bool     is_lr    = (req.op == MemOp::AMO_LR);
      const bool     is_sc    = (req.op == MemOp::AMO_SC);
      const bool     sc_fail  = is_sc && !amo_unit_.check(hart_id, la);
      const bool     do_store = !is_lr && !sc_fail;

      uint8_t obuf[8] = {0};
      ram_.read(obuf, la, wbytes);
      const uint64_t old_word = amo_load_word(obuf, 0, width);
      const uint64_t rhs = req.data.at(i)
        ? amo_load_word(req.data.at(i)->data(), byte_off, width) : 0;
      auto rmw = amo_compute(req.op, width, old_word, rhs,
                             req.flags.amo_unsigned, req.amo_cmp.at(i));

      auto rsp_data = make_mem_block();
      std::memset(rsp_data->data(), 0, rsp_data->size());
      const uint64_t ret_word = is_sc ? (sc_fail ? 1ull : 0ull) : rmw.ret_word;
      amo_store_word(rsp_data->data(), byte_off, width, ret_word);

      // An SC gives up its own reservation whether it succeeds or fails; an
      // LR claims one. A committed store breaks every other hart's
      // reservation on the word, or a second hart's stale reservation would
      // survive the first hart's successful SC and one update would be lost.
      if (is_lr) {
        amo_unit_.reserve(hart_id, la);
      } else if (is_sc) {
        amo_unit_.clear(hart_id, la);
      }
      if (do_store) {
        uint8_t sbuf[8];
        amo_store_word(sbuf, 0, width, rmw.new_word);
        ram_.write(sbuf, la, wbytes);
        amo_unit_.invalidate(la, hart_id);
      }
      // an atomic responds, so it counts as a read
      ++perf_stats_.reads;
      return rsp_data;
    }
#endif
    if (req.is_write()) {
      for (uint32_t b = 0; req.data.at(i) && b < VX_CFG_MEM_BLOCK_SIZE; ++b) {
        if (req.byteen.at(i) & (1ull << b)) {
          uint8_t value = (*req.data.at(i))[b];
          ram_.write(&value, line_addr + b, 1);
        }
      }
#if VX_CFG_EXT_A_ENABLED
      // A plain store breaks a reservation on the word it writes, as an
      // atomic's commit does, or a store between a hart's load-reserved and
      // its store-conditional would go unnoticed.
      amo_unit_.invalidate(la, hart_id);
#endif
      ++perf_stats_.writes;
      return nullptr;
    }
    auto rsp_data = make_mem_block();
    ram_.read(rsp_data->data(), line_addr, VX_CFG_MEM_BLOCK_SIZE);
    ++perf_stats_.reads;
    return rsp_data;
  }

  bool dma_pending() const {
    for (auto& in : simobject_->DmaInputs) {
      if (!in.empty()) {
        return true;
      }
    }
    return false;
  }

  // Serve one DMA client's row access. Clients are served round-robin,
  // starting after the last one served.
  void dma_fire() {
    for (uint32_t i = 0; i < config_.dma_clients; ++i) {
      uint32_t c = (dma_last_grant_ + 1 + i) % config_.dma_clients;
      bool pending = false;
      bool rsp_ready = true;
      for (uint32_t p = 0; p < DMA_PORTS; ++p) {
        auto& in = simobject_->DmaInputs.at(c * DMA_PORTS + p);
        if (in.empty()) {
          continue;
        }
        pending = true;
        if (!in.peek().is_write() && simobject_->DmaOutputs.at(c * DMA_PORTS + p).full()) {
          rsp_ready = false;
        }
      }
      if (!pending) {
        continue;
      }
      if (!rsp_ready) {
        return;
      }
      for (uint32_t p = 0; p < DMA_PORTS; ++p) {
        auto& in = simobject_->DmaInputs.at(c * DMA_PORTS + p);
        if (in.empty()) {
          continue;
        }
        auto& req = in.peek();
        uint64_t line_addr = to_local_addr(req.addr) & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
        if (req.is_write()) {
          for (uint32_t b = 0; req.data && b < VX_CFG_MEM_BLOCK_SIZE; ++b) {
            if (req.byteen & (1ull << b)) {
              uint8_t value = (*req.data)[b];
              ram_.write(&value, line_addr + b, 1);
            }
          }
#if VX_CFG_EXT_A_ENABLED
          amo_unit_.invalidate(to_local_addr(req.addr), req.hart_id);
#endif
        } else {
          MemRsp rsp{req.tag, req.hart_id, req.uuid};
          auto rsp_data = make_mem_block();
          ram_.read(rsp_data->data(), line_addr, VX_CFG_MEM_BLOCK_SIZE);
          rsp.data = rsp_data;
          // The request arrived over a registered channel; the read data
          // returns the cycle after it was issued.
          simobject_->DmaOutputs.at(c * DMA_PORTS + p).send(rsp, 0);
        }
        DT(4, simobject_->name() << "-dma" << c << " req : " << req);
        in.pop();
      }
      dma_last_grant_ = c;
      return;
    }
  }

  // Issue an engine's next pass. Returns true when it was the request's last,
  // which frees S1 for a new request in the same cycle. The response queue
  // occupancy is the one from the start of the cycle.
  bool issue_pass(Engine& e, uint32_t rsp_count) {
    auto& req = e.s1.req;
    bool has_rsp = !req.is_write() || req.is_amo();
    if (has_rsp && rsp_count >= RSP_QUEUE_SIZE) {
      return false;
    }
    auto served = this->select_pass(req, e.pending, e.phase * config_.word_size);
    LsuRsp rsp(config_.num_lanes);
    rsp.tag   = req.tag;
    rsp.cid   = req.cid;
    rsp.uuid  = req.uuid;
    rsp.phase = e.phase;
    for (uint32_t i = 0; i < config_.num_lanes; ++i) {
      if (!served.test(i)) {
        continue;
      }
      auto data = this->access(req, i);
      if (has_rsp) {
        rsp.mask.set(i);
        rsp.data.at(i) = data;
      }
      e.pending.reset(i);
    }
    rsp.phase_last = e.pending.none();
    DT(4, simobject_->name() << " pass: phase=" << e.phase << ", served=" << served << ", " << req);
    if (has_rsp) {
      e.rsp_queue.push_back({SimPlatform::instance().cycles() + RSP_LATENCY, e.s1.block, std::move(rsp)});
    }
#if VX_CFG_EXT_A_ENABLED
    if (req.is_amo()) {
      stall_cycles_ = AMO_STALL_CYCLES;
    }
#endif
    if (e.pending.none()) {
      if (e.phase < req.flags.nwords_m1) {
        // the next word-phase serves every lane again, from the next cycle
        ++e.phase;
        e.pending = req.mask;
        return false;
      }
      e.phase = 0;
      e.s1_valid = false;
      return true;
    }
    ++perf_stats_.bank_stalls;
    e.ready = false;
    return false;
  }

  void take(Engine& e, uint32_t b) {
    auto& in = simobject_->ReqIn.at(b);
    Request r;
    r.req = in.peek();
    r.block = b;
    DT(4, simobject_->name() << "-req" << b << ": " << r.req);
    e.s0.push_back(std::move(r));
    in.pop();
  }

  // The lowest eligible block after the last one chosen, round-robin.
  int32_t pick(const std::vector<bool>& eligible, uint32_t& last) {
    for (uint32_t i = 0; i < config_.num_blocks; ++i) {
      uint32_t b = (last + 1 + i) % config_.num_blocks;
      if (eligible.at(b)) {
        last = b;
        return b;
      }
    }
    return -1;
  }

  // Each engine takes the block chosen for it the cycle before, if its
  // request is still there, then the choices for the next cycle are made from
  // the requests waiting at the start of this one. A block served this cycle
  // may present a new request of either kind next cycle, so a choice of it
  // counts as a store: the first engine takes a store only while the read
  // engine is idle, and the read engine takes nothing in that cycle. The read
  // engine takes a read only while no store or atomic waited at an input the
  // cycle before or sits in the first engine.
  void arbitrate(const std::vector<bool>& s0_ready, bool a_write, bool b_busy) {
    uint32_t n = config_.num_blocks;
    std::vector<bool> in_valid(n);
    std::vector<bool> in_read(n);
    std::vector<bool> fired(n, false);
    for (uint32_t b = 0; b < n; ++b) {
      auto& in = simobject_->ReqIn.at(b);
      in_valid.at(b) = !in.empty();
      in_read.at(b)  = in_valid.at(b) && is_plain_read(in.peek());
    }
    bool a_take = a_sel_.valid && s0_ready.at(0) && (a_sel_.read || !b_busy);
    bool b_take = b_sel_.valid && s0_ready.at(1) && !a_write && !write_waiting_r_
               && !(a_take && !a_sel_.read);
    bool b_fired = false;
    if (a_take && in_valid.at(a_sel_.block)) {
      this->take(engines_.at(0), a_sel_.block);
      fired.at(a_sel_.block) = true;
    }
    if (b_take && in_valid.at(b_sel_.block)) {
      this->take(engines_.at(1), b_sel_.block);
      fired.at(b_sel_.block) = true;
      b_fired = true;
    }

    std::vector<bool> known_read(n);
    std::vector<bool> a_elig(n);
    std::vector<bool> b_elig(n);
    bool b_busy_n = b_busy || b_fired;
    bool write_waiting = false;
    for (uint32_t b = 0; b < n; ++b) {
      known_read.at(b) = in_read.at(b) && !fired.at(b);
      a_elig.at(b) = known_read.at(b) || (in_valid.at(b) && !b_busy_n);
      write_waiting = write_waiting || (in_valid.at(b) && !in_read.at(b) && !fired.at(b));
    }
    int32_t a_next = this->pick(a_elig, arb_last_.at(0));
    for (uint32_t b = 0; b < n; ++b) {
      b_elig.at(b) = known_read.at(b) && ((int32_t)b != a_next);
    }
    int32_t b_next = this->pick(b_elig, arb_last_.at(1));
    a_sel_.valid = (a_next != -1);
    a_sel_.block = (a_next != -1) ? a_next : 0;
    a_sel_.read  = (a_next != -1) && known_read.at(a_next);
    b_sel_.valid = (b_next != -1);
    b_sel_.block = (b_next != -1) ? b_next : 0;
    write_waiting_r_ = write_waiting;
  }

public:
  Impl(LocalMem* simobject, const Config& config)
    : simobject_(simobject)
    , config_(config)
    , ram_(config.capacity)
    , addr_bits_(log2ceil(config.capacity))
    , lg2_word_(log2ceil(config.word_size))
#if VX_CFG_EXT_A_ENABLED
    , amo_unit_(VX_CFG_AMO_RS_SIZE < 2 ? 2u : (uint32_t)VX_CFG_AMO_RS_SIZE)
#endif
    , stall_cycles_(0)
    , engines_((config.num_blocks > 1) ? 2 : 1)
    , rsp_regs_(config.num_blocks)
    , arb_last_(engines_.size(), 0)
    , write_waiting_r_(false)
    , dma_slot_(false)
    , dma_last_grant_(0)
  {
    for (auto& e : engines_) {
      e.pending = BitVector<>(config.num_lanes);
    }
  }

  virtual ~Impl() {}

  void reset() {
    // perf_stats_ survives: the MPM counters run from device reset.
#if VX_CFG_EXT_A_ENABLED
    amo_unit_.reset();
#endif
    stall_cycles_ = 0;
    for (auto& e : engines_) {
      e.s0.clear();
      e.s1_valid = false;
      e.pending.reset();
      e.phase = 0;
      e.rsp_queue.clear();
    }
    for (auto& r : rsp_regs_) {
      r.valid = false;
    }
    std::fill(arb_last_.begin(), arb_last_.end(), config_.num_blocks - 1);
    a_sel_ = Selection();
    b_sel_ = Selection();
    write_waiting_r_ = false;
    dma_slot_ = false;
    dma_last_grant_ = config_.dma_clients ? (config_.dma_clients - 1) : 0;
  }

  void tick() {
    uint32_t num_engines = engines_.size();
    uint64_t now = SimPlatform::instance().cycles();

    // Responses. Each block's response register goes to the block; a free
    // one then takes the first engine's queue head ahead of the read engine's.
    // An engine's slots count the responses in its queue and in the registers.
    std::vector<uint32_t> rsp_count(num_engines);
    for (uint32_t k = 0; k < num_engines; ++k) {
      rsp_count.at(k) = engines_.at(k).rsp_queue.size();
    }
    for (auto& r : rsp_regs_) {
      rsp_count.at(r.engine) += r.valid;
    }
    for (uint32_t b = 0; b < config_.num_blocks; ++b) {
      auto& r = rsp_regs_.at(b);
      if (r.valid && simobject_->RspOut.at(b).try_send(r.rsp, 0)) {
        DT(4, simobject_->name() << "-rsp" << b << ": " << r.rsp);
        r.valid = false;
      }
    }
    for (uint32_t k = 0; k < num_engines; ++k) {
      auto& e = engines_.at(k);
      if (e.rsp_queue.empty() || e.rsp_queue.front().ready > now) {
        continue;
      }
      auto& r = rsp_regs_.at(e.rsp_queue.front().block);
      if (r.valid) {
        continue;
      }
      r.valid  = true;
      r.engine = k;
      r.rsp    = std::move(e.rsp_queue.front().rsp);
      e.rsp_queue.pop_front();
    }

    // the arbiter's view of the engines, as registered at the cycle start
    bool a_write = holds_write(engines_.at(0));
    bool b_busy  = (num_engines > 1) && (!engines_.at(1).s0.empty() || engines_.at(1).s1_valid);
    std::vector<bool> s0_ready(num_engines);
    for (uint32_t k = 0; k < num_engines; ++k) {
      s0_ready.at(k) = (engines_.at(k).s0.size() < S0_SIZE);
    }

    // S1: a DMA access, or the engines' passes. A DMA request seen in one
    // cycle takes the banks the next; no pass is issued while one is seen.
    // An engine whose last pass left lanes pending spends this cycle finding
    // the next winners.
    std::vector<bool> s1_ready(num_engines);
    std::vector<bool> can_pass(num_engines);
    for (uint32_t k = 0; k < num_engines; ++k) {
      auto& e = engines_.at(k);
      s1_ready.at(k) = !e.s1_valid;
      can_pass.at(k) = e.s1_valid && e.ready;
      if (e.s1_valid && !e.ready) {
        e.ready = true;
      }
    }
    if (stall_cycles_ != 0) {
      --stall_cycles_;
      dma_slot_ = false;
    } else {
      bool dma_valid = this->dma_pending();
      if (dma_slot_) {
        this->dma_fire();
      }
      dma_slot_ = dma_valid;
      if (!dma_valid) {
        for (uint32_t k = 0; k < num_engines; ++k) {
          if (can_pass.at(k)) {
            s1_ready.at(k) = this->issue_pass(engines_.at(k), rsp_count.at(k));
          }
        }
      }
    }

    // S0 -> S1, with the single-pass check
    for (uint32_t k = 0; k < num_engines; ++k) {
      auto& e = engines_.at(k);
      if (s1_ready.at(k) && !e.s0.empty()) {
        e.s1 = std::move(e.s0.front());
        e.s0.pop_front();
        e.s1_valid = true;
        e.pending = e.s1.req.mask;
        e.ready = true;
      }
    }

    if (num_engines == 1) {
      // one block: its request goes straight to the engine
      auto& in = simobject_->ReqIn.at(0);
      if (s0_ready.at(0) && !in.empty()) {
        this->take(engines_.at(0), 0);
      }
    } else {
      this->arbitrate(s0_ready, a_write, b_busy);
    }

    bool idle = (stall_cycles_ == 0) && !dma_slot_ && !a_sel_.valid && !b_sel_.valid;
    for (auto& e : engines_) {
      idle = idle && e.s0.empty() && !e.s1_valid && e.rsp_queue.empty();
    }
    for (auto& r : rsp_regs_) {
      idle = idle && !r.valid;
    }
    if (idle && !this->dma_pending()) {
      simobject_->tick_sleep();
    }
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }
};

///////////////////////////////////////////////////////////////////////////////

LocalMem::LocalMem(const SimContext& ctx, const char* name, const Config& config)
  : SimObject<LocalMem>(ctx, name)
  , ReqIn(config.num_blocks, this)
  , RspOut(config.num_blocks, this)
  , DmaInputs(config.dma_clients * DMA_PORTS, this)
  , DmaOutputs(config.dma_clients * DMA_PORTS, this)
  , impl_(new Impl(this, config))
{}

LocalMem::~LocalMem() {
  delete impl_;
}

void LocalMem::on_reset() {
  impl_->reset();
}

void LocalMem::on_tick() {
  impl_->tick();
}

const LocalMem::PerfStats& LocalMem::perf_stats() const {
  return impl_->perf_stats();
}
