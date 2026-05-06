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

#include "cache.h"
#include "mem_block_pool.h"
#include "debug.h"
#include "types.h"
#include "amo_unit.h"
#include <cstring>
#include <list>
#include <queue>
#include <unordered_map>
#include <util.h>
#include <vector>

using namespace vortex;

struct params_t {
  uint32_t sets_per_bank;
  uint32_t lines_per_set;
  uint32_t words_per_line;
  uint32_t log2_num_inputs;

  int32_t word_select_addr_start;
  int32_t word_select_addr_end;

  int32_t bank_select_addr_start;
  int32_t bank_select_addr_end;

  int32_t set_select_addr_start;
  int32_t set_select_addr_end;

  int32_t tag_select_addr_start;
  int32_t tag_select_addr_end;

  params_t(const Cache::Config &config) {
    int32_t offset_bits = config.L - config.W;
    int32_t index_bits = config.C - (config.L + config.A + config.B);
    assert(offset_bits >= 0);
    assert(index_bits >= 0);

    this->log2_num_inputs = log2ceil(config.num_inputs);

    this->sets_per_bank = 1 << index_bits;
    this->lines_per_set = 1 << config.A;
    this->words_per_line = 1 << offset_bits;

    // Word select
    this->word_select_addr_start = config.W;
    this->word_select_addr_end = (this->word_select_addr_start + offset_bits - 1);

    // Bank select
    this->bank_select_addr_start = (1 + this->word_select_addr_end);
    this->bank_select_addr_end = (this->bank_select_addr_start + config.B - 1);

    // Set select
    this->set_select_addr_start = (1 + this->bank_select_addr_end);
    this->set_select_addr_end = (this->set_select_addr_start + index_bits - 1);

    // Tag select
    this->tag_select_addr_start = (1 + this->set_select_addr_end);
    this->tag_select_addr_end = (config.addr_width - 1);
  }

  uint32_t addr_bank_id(uint64_t addr) const {
    if (bank_select_addr_end >= bank_select_addr_start)
      return (uint32_t)bit_getw(addr, bank_select_addr_start, bank_select_addr_end);
    else
      return 0;
  }

  uint32_t addr_set_id(uint64_t addr) const {
    if (set_select_addr_end >= set_select_addr_start)
      return (uint32_t)bit_getw(addr, set_select_addr_start, set_select_addr_end);
    else
      return 0;
  }

  uint64_t addr_tag(uint64_t addr) const {
    if (tag_select_addr_end >= tag_select_addr_start)
      return bit_getw(addr, tag_select_addr_start, tag_select_addr_end);
    else
      return 0;
  }

  uint64_t mem_addr(uint32_t bank_id, uint32_t set_id, uint64_t tag) const {
    uint64_t addr(0);
    if (bank_select_addr_end >= bank_select_addr_start)
      addr = bit_setw(addr, bank_select_addr_start, bank_select_addr_end, bank_id);
    if (set_select_addr_end >= set_select_addr_start)
      addr = bit_setw(addr, set_select_addr_start, set_select_addr_end, set_id);
    if (tag_select_addr_end >= tag_select_addr_start)
      addr = bit_setw(addr, tag_select_addr_start, tag_select_addr_end, tag);
    return addr;
  }
};

struct line_t {
  uint64_t tag;
  uint32_t lru_ctr;
  bool valid;
  bool dirty;
  std::shared_ptr<mem_block_t> data;  // line bytes

  void reset() {
    valid = false;
    dirty = false;
    lru_ctr = 0;
    data.reset();
  }
};

static inline void line_merge(line_t& line, const std::shared_ptr<mem_block_t>& src, uint64_t byteen) {
  // Copy-on-write only when shared. line.data may be aliased with in-flight
  // responses, fill payloads, or writeback messages; mutating in place would
  // corrupt them. When this cache line is the sole owner, mutate in place to
  // avoid a heap allocation on the hot path.
  if (line.data) {
    if (line.data.use_count() > 1) {
      line.data = make_mem_block_copy(*line.data);
    }
  } else {
    line.data = make_mem_block();
    std::memset(line.data->data(), 0, line.data->size());
  }
  if (src) {
    for (uint32_t b = 0; b < MEM_BLOCK_SIZE; ++b) {
      if (byteen & (1ull << b)) {
        (*line.data)[b] = (*src)[b];
      }
    }
  }
}

struct set_t {
  std::vector<line_t> lines;
  uint32_t fifo_ptr;    // next victim for FIFO policy

  set_t(uint32_t num_ways)
      : lines(num_ways), fifo_ptr(0) {}

  void reset() {
    for (auto &line : lines) {
      line.reset();
    }
    fifo_ptr = 0;
  }

  // Pure tag lookup: returns hit id (or -1), fills free/repl line ids. No mutation.
  // Callers must invoke update_lru() *after* all stall checks pass, otherwise
  // PLRU counters drift on retry.
  int tag_match(uint64_t tag, uint8_t policy, uint32_t rand_idx,
                int *free_line_id, int *repl_line_id) const {
    int hit_line_id = -1;
    *free_line_id = -1;
    *repl_line_id = 0;

    uint32_t max_cnt = 0;
    bool any_valid = false;
    bool plru_chosen = false;

    for (uint32_t i = 0, n = lines.size(); i < n; ++i) {
      const auto &line = lines.at(i);

      if (!line.valid) {
        if (*free_line_id == -1)
          *free_line_id = i;
        continue;
      }
      any_valid = true;

      if (line.tag == tag)
        hit_line_id = i;

      if (policy == Cache::PLRU) {
        if (!plru_chosen || line.lru_ctr >= max_cnt) {
          max_cnt = line.lru_ctr;
          *repl_line_id = i;
          plru_chosen = true;
        }
      }
    }

    // Select victim per policy (for miss path).
    switch (policy) {
    case Cache::FIFO:
      *repl_line_id = fifo_ptr % lines.size();
      break;
    case Cache::RANDOM:
      *repl_line_id = rand_idx % lines.size();
      break;
    case Cache::PLRU:
    default:
      if (!any_valid)
        *repl_line_id = (*free_line_id != -1) ? *free_line_id : 0;
      break;
    }

    return hit_line_id;
  }

  // Apply PLRU age update for a tag access. Pass hit_line_id == -1 for a miss
  // (all valid lines age, no reset). Call once per *committed* access.
  void update_lru(int hit_line_id) {
    for (uint32_t i = 0, n = lines.size(); i < n; ++i) {
      auto &line = lines.at(i);
      if (!line.valid)
        continue;
      if ((int)i == hit_line_id) {
        line.lru_ctr = 0;
      } else {
        ++line.lru_ctr;
      }
    }
  }

  // Choose a victim line for installing a fill. Does NOT mutate state.
  int select_victim(uint8_t policy, uint32_t rand_idx,
                    int *free_line_id, int *repl_line_id) const {
    *free_line_id = -1;
    *repl_line_id = 0;

    uint32_t max_cnt = 0;
    bool any_valid = false;

    for (uint32_t i = 0, n = lines.size(); i < n; ++i) {
      const auto &line = lines.at(i);
      if (!line.valid) {
        if (*free_line_id == -1)
          *free_line_id = i;
        continue;
      }
      any_valid = true;
      if (policy == Cache::PLRU) {
        if (line.lru_ctr >= max_cnt) {
          max_cnt = line.lru_ctr;
          *repl_line_id = i;
        }
      }
    }

    switch (policy) {
    case Cache::FIFO:
      *repl_line_id = fifo_ptr % lines.size();
      break;
    case Cache::RANDOM:
      *repl_line_id = rand_idx % lines.size();
      break;
    case Cache::PLRU:
    default:
      if (!any_valid)
        *repl_line_id = (*free_line_id != -1) ? *free_line_id : 0;
      break;
    }

    return (*free_line_id != -1) ? *free_line_id : *repl_line_id;
  }
};

struct bank_req_t {

  using Ptr = std::shared_ptr<bank_req_t>;

  enum ReqType {
    None = 0,
    Fill = 1,
    Replay = 2,
    Core = 3,
    AmoProbe = 4  // non-LLC AMO passthrough: probe-and-invalidate then forward
  };

  uint64_t addr;
  uint32_t cid;
  uint64_t req_tag;
  uint64_t uuid;
  uint32_t mshr_id;
  ReqType type;
  bool write;
  // For write-through write-misses that piggy-back on a pending fill MSHR:
  // the core response was already sent at miss time, so Replay must not
  // emit another response — only run line_merge.
  bool skip_core_rsp;

  // TLM data:
  //   For Core writes: incoming write data + byteen.
  //   For Fill: captured fill data from below (mem_rsp.data).
  std::shared_ptr<mem_block_t> data;
  uint64_t byteen;

  // AMO sideband (LLC bank only). When `amo.valid`, `write` is false
  // and the bank handles RMW commit through the AmoUnit in the same
  // cycle as a write-hit. `data` is null for AMOs — `amo.rhs` carries
  // rs2. The reservation key is `amo.hart_id` (computed upstream as
  // make_hart_id(cid, wid, tid)).
  amo_req_t amo;

  bank_req_t() {
    this->reset();
  }

  void reset() {
    addr = 0;
    cid = 0;
    req_tag = 0;
    uuid = 0;
    mshr_id = 0;
    type = ReqType::None;
    write = false;
    skip_core_rsp = false;
    data.reset();
    byteen = 0;
    amo = amo_req_t{};
  }

  friend std::ostream &operator<<(std::ostream &os, const bank_req_t &req) {
    os << "addr=0x" << std::hex << req.addr;
    os << ", rw=" << std::dec << req.write;
    os << ", type=" << req.type;
    os << ", req_tag=" << req.req_tag;
    os << ", cid=" << req.cid;
    os << " (#" << req.uuid << ")";
    return os;
  }
};

struct mshr_entry_t {
  bank_req_t bank_req;
  uint32_t set_id;
  uint64_t addr_tag;
  uint32_t line_id;

  mshr_entry_t() {
    this->reset();
  }

  void reset() {
    bank_req.reset();
    set_id = 0;
    addr_tag = 0;
    line_id = 0;
  }
};

class MSHR {
public:
  MSHR(uint32_t size)
      : entries_(size), ready_reqs_(0), size_(0) {}

  uint32_t capacity() const {
    return entries_.size();
  }

  uint32_t size() const {
    return size_;
  }

  bool empty() const {
    return (0 == size_);
  }

  bool full() const {
    assert(size_ <= entries_.size());
    return (size_ == entries_.size());
  }

  bool has_ready_reqs() const {
    return (ready_reqs_ != 0);
  }

  const mshr_entry_t &peek(uint32_t id) const {
    return entries_.at(id);
  }

  // Returns true if there is an active pending request for the given set/tag.
  // If true, optionally returns the root entry id.
  bool lookup(uint32_t set_id, uint64_t addr_tag, uint32_t *root_id = nullptr) const {
    for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
      const auto &entry = entries_.at(i);
      if (entry.bank_req.type != bank_req_t::None && entry.set_id == set_id && entry.addr_tag == addr_tag) {
        if (root_id)
          *root_id = i;
        return true;
      }
    }
    return false;
  }

  // Enqueue a new core request and return the allocated entry id.
  int enqueue(const bank_req_t &bank_req, uint32_t set_id, uint64_t addr_tag) {
    assert(bank_req.type == bank_req_t::Core);
    for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
      auto &entry = entries_.at(i);
      if (entry.bank_req.type == bank_req_t::None) {
        entry.bank_req = bank_req;
        entry.set_id = set_id;
        entry.addr_tag = addr_tag;
        entry.line_id = 0; // victim is selected at Fill time
        ++size_;
        return i;
      }
    }
    std::abort(); // no free slot found!
    return -1;
  }

  // Mark all pending requests matching the entry's tag for replay.
  mshr_entry_t &replay(uint32_t id) {
    auto &root_entry = entries_.at(id);
    assert(root_entry.bank_req.type == bank_req_t::Core);
    assert(ready_reqs_ == 0);
    for (auto &entry : entries_) {
      if (entry.bank_req.type == bank_req_t::Core && entry.set_id == root_entry.set_id && entry.addr_tag == root_entry.addr_tag) {
        entry.bank_req.type = bank_req_t::Replay;
        ++ready_reqs_;
      }
    }
    return root_entry;
  }

  // Dequeue the next ready replay request. Reads are dequeued before writes
  // so that read responses capture the pre-write cached line state. A
  // write-through wt-merge entry must not modify the line until any reads
  // pending on the same fill have completed and captured their response data.
  void dequeue(bank_req_t *out) {
    assert(ready_reqs_ > 0);
    mshr_entry_t *picked = nullptr;
    for (auto &entry : entries_) {
      if (entry.bank_req.type != bank_req_t::Replay)
        continue;
      if (!entry.bank_req.write) {
        picked = &entry;
        break;
      }
      if (picked == nullptr)
        picked = &entry;
    }
    *out = picked->bank_req;
    picked->bank_req.type = bank_req_t::None;
    --ready_reqs_;
    --size_;
  }

  void reset() {
    for (auto &entry : entries_) {
      entry.reset();
    }
    ready_reqs_ = 0;
    size_ = 0;
  }

private:
  std::vector<mshr_entry_t> entries_;
  uint32_t ready_reqs_;
  uint32_t size_;
};

class CacheBank : public SimObject<CacheBank> {
public:
  SimChannel<MemReq> core_req_in;
  SimChannel<MemRsp> core_rsp_out;

  SimChannel<MemReq> mem_req_out;
  SimChannel<MemRsp> mem_rsp_in;

  CacheBank(const SimContext &ctx,
            const char *name,
            const Cache::Config &config,
            const params_t &params,
            uint32_t bank_id)
      : SimObject<CacheBank>(ctx, name), core_req_in(this), core_rsp_out(this), mem_req_out(this), mem_rsp_in(this), config_(config), params_(params), bank_id_(bank_id), sets_(params.sets_per_bank, params.lines_per_set), mshr_(config.mshr_size), pipe_req_(TFifo<bank_req_t>::Create("", config.latency)), rand_ctr_(0)
#if EXT_A_ENABLED
      , amo_unit_(__MAX(2u, (uint32_t)AMO_RS_SIZE))
#endif
  {
    this->on_reset();
  }

  const Cache::PerfStats &perf_stats() const {
    return perf_stats_;
  }

  // Flush API.
  // flush_begin() arms the bank; subsequent ticks scan all sets/ways and emit
  // a writeback request for every dirty line via mem_req_out (write-back only;
  // write-through caches have nothing to evict). flush_done() reports when the
  // walk finishes AND the cache is otherwise idle.
  void flush_begin() {
    if (!config_.write_back) {
      flushing_ = false;
      flush_set_idx_ = 0;
      flush_way_idx_ = 0;
      return;
    }
    flushing_ = true;
    flush_set_idx_ = 0;
    flush_way_idx_ = 0;
  }

  bool flush_done() const {
    return !flushing_;
  }

protected:
  void on_reset() {
    perf_stats_ = Cache::PerfStats();
    pending_mshr_size_ = 0;
    pending_read_reqs_ = 0;
    pending_write_reqs_ = 0;
    pending_fill_reqs_ = 0;
    rand_ctr_ = 0;
    for (auto &set : sets_) {
      set.reset();
    }
    mshr_.reset();
    flushing_ = false;
    flush_set_idx_ = 0;
    flush_way_idx_ = 0;
#if EXT_A_ENABLED
    amo_unit_.reset();
    for (auto &e : amo_passthru_) {
      e = amo_passthru_entry_t{};
    }
#endif
  }

  void on_tick() {
    // Process next request at the head of the pipeline.
    if (!pipe_req_->empty()) {
        this->processRequests();
    }

    // Accept one new input if there's room.
    if (!pipe_req_->full()) {
        this->processInputs();
    }

    // flush walk: emit writebacks for dirty lines.
    if (flushing_) {
      this->processFlush();
    }

    // calculate memory latency
    perf_stats_.mem_latency += pending_fill_reqs_;
  }

private:
  // Pipeline front: per-tick input arbitration.
  //
  // Priority (highest → lowest):
  //   1) replay   — drains an already-marked Replay entry from the MSHR
  //                 (guaranteed hit; no miss path)
  //   2) fill     — memory response; installs the line and marks pending
  //                 MSHR entries for replay (gated on no pending replay so
  //                 a fill never preempts an in-flight replay's line)
  //   3) flush    — handled by processFlush() (separate state machine)
  //   4) core_req — new core request (may miss and allocate an MSHR slot)
  //
  // At most one input fires per tick. All inputs flow through pipe_req_.
  void processInputs() {
    // 1) replay
    if (mshr_.has_ready_reqs()) {
      bank_req_t bank_req;
      mshr_.dequeue(&bank_req);
      pipe_req_->push(bank_req);
      DT(3, this->name() << " replay-deq: " << bank_req);
      return;
    }

    // 2) fill only when no replay is pending
    if (!this->mem_rsp_in.empty()) {
      auto &mem_rsp = this->mem_rsp_in.peek();
#if EXT_A_ENABLED
      // Non-LLC AMO passthrough response: forward straight to core
      // without filling. The original tag was rewritten to
      // (mshr_capacity + pid) when the bank emitted the request from
      // the AmoProbe handler — that namespace partition survives
      // arbiter tag mangling.
      if (mem_rsp.tag >= amo_passthru_tag_base()
       && mem_rsp.tag <  amo_passthru_tag_base() + AMO_PASSTHRU_CAP) {
        uint32_t pid = mem_rsp.tag - amo_passthru_tag_base();
        auto &e = amo_passthru_.at(pid);
        assert(e.valid && "AMO passthru response without entry");
        if (this->core_rsp_out.full()) {
          return; // stall
        }
        MemRsp core_rsp{e.req_tag, e.cid, e.uuid};
        core_rsp.data = mem_rsp.data;
        this->core_rsp_out.send(core_rsp);
        DT(3, this->name() << " amo-passthru-rsp: " << core_rsp);
        e.valid = false;
        this->mem_rsp_in.pop();
        --pending_fill_reqs_;
        return;
      }
#endif
      uint32_t mshr_id = mem_rsp.tag;
      const auto &root_peek = mshr_.peek(mshr_id);
      bank_req_t bank_req;
      bank_req.reset();
      bank_req.type    = bank_req_t::Fill;
      bank_req.addr    = params_.mem_addr(bank_id_, root_peek.set_id, root_peek.addr_tag);
      bank_req.cid     = root_peek.bank_req.cid;
      bank_req.uuid    = root_peek.bank_req.uuid;
      bank_req.mshr_id = mshr_id;
      bank_req.data    = mem_rsp.data;
      pipe_req_->push(bank_req);
      DT(3, this->name() << " fill-rsp: " << mem_rsp);
      this->mem_rsp_in.pop();
      --pending_fill_reqs_;
      return;
    }

    // 3) core request
    if (!this->core_req_in.empty()) {
      auto &core_req = this->core_req_in.peek();
      // Conservative MSHR occupancy check: any request that may miss must
      // reserve a slot. Counts both currently-allocated entries and
      // in-flight pipe requests that haven't reached MSHR allocation yet.
      // AMO requests always need a return; at the LLC they reserve like
      // a load (proposal §3.7), at non-LLC they don't fill so no MSHR
      // slot is needed but a passthru side-table slot is.
      const bool is_amo = memop_is_amo(core_req.op);
#if EXT_A_ENABLED
      const bool is_amo_passthru = is_amo && !config_.is_llc;
      if (is_amo_passthru) {
        // Need a free passthru-table slot before accepting.
        bool any_free = false;
        for (const auto &e : amo_passthru_) { if (!e.valid) { any_free = true; break; } }
        if (!any_free) {
          ++perf_stats_.mshr_stalls;
          return;
        }
      }
#else
      const bool is_amo_passthru = false;
#endif
      bool needs_mshr = (!core_req.write || config_.write_back || is_amo) && !is_amo_passthru;
      if (needs_mshr && (mshr_.size() + pending_mshr_size_) >= mshr_.capacity()) {
        ++perf_stats_.mshr_stalls;
        return;
      }
      bank_req_t bank_req;
      bank_req.reset();
      bank_req.type    = is_amo_passthru ? bank_req_t::AmoProbe : bank_req_t::Core;
      bank_req.addr    = core_req.addr;
      bank_req.cid     = core_req.cid;
      bank_req.uuid    = core_req.uuid;
      bank_req.req_tag = core_req.tag;
      bank_req.write   = core_req.write;
      bank_req.data    = core_req.data;
      bank_req.byteen  = core_req.byteen;
      if (is_amo) {
        bank_req.amo = core_req.amo;
      }
      pipe_req_->push(bank_req);
      DT(3, this->name() << " core-req: " << core_req);
      // pending_mshr_size_ tracks Core-typed in-flight requests so the
      // MSHR pre-reservation in processInputs is conservative. AmoProbe
      // doesn't allocate an MSHR slot (no fill on the response), so it
      // stays out of this counter.
      if (!is_amo_passthru) ++pending_mshr_size_;
      if (core_req.write) ++perf_stats_.writes;
      else                ++perf_stats_.reads;
      this->core_req_in.pop();
      return;
    }
  }

#if EXT_A_ENABLED
  // AMO commit at the LLC bank. Returns false if the cycle stalls
  // (caller leaves bank_req at the head of pipe_req_); returns true
  // when the commit completes and the caller should pop pipe_req_.
  // Mirrors the write-hit pattern: collect all stall conditions
  // before any mutation.
  bool commitAmo(const bank_req_t &bank_req, set_t &set, int hit_id, uint32_t set_id) {
    auto &hit_line = set.lines.at(hit_id);
    const uint64_t line_addr = (bank_req.addr >> config_.L) << config_.L;
    const uint32_t byte_off  = (uint32_t)(bank_req.addr & (MEM_BLOCK_SIZE - 1));
    const AmoType  op        = bank_req.amo.op;
    const uint8_t  width     = bank_req.amo.width;
    const uint32_t hid       = bank_req.amo.hart_id;

    const bool sc_fail  = (op == AmoType::SC) && !amo_unit_.check(hid, line_addr);
    const bool do_store = (op != AmoType::LR) && !sc_fail;

    // Stall checks (collect all before mutating).
    if (this->core_rsp_out.full())
      return false;
    if (do_store && !config_.write_back && this->mem_req_out.full())
      return false;

    // Pure compute: read old word, derive new and ret.
    uint64_t old_word = 0;
    if (hit_line.data) {
      old_word = amo_load_word(hit_line.data->data(), byte_off, width);
    }
    auto rmw = amo_unit_.compute(op, width, old_word, bank_req.amo.rhs);

    // Build response payload: a fresh block with the relevant word at byte_off.
    // For LR/AMO* the word carries old_word (LSU sext at width gives rd).
    // For SC it carries 0 (success) or 1 (failure).
    auto rsp_block = make_mem_block();
    std::memset(rsp_block->data(), 0, rsp_block->size());
    uint64_t rsp_word = (op == AmoType::SC) ? (sc_fail ? 1ull : 0ull)
                                            : rmw.ret_word;
    amo_store_word(rsp_block->data(), byte_off, width, rsp_word);

    // Reservation update before commit so a same-cycle invalidate
    // from the store path doesn't kill our just-installed entry.
    if (op == AmoType::LR) {
      amo_unit_.reserve(hid, line_addr);
    } else if (op == AmoType::SC) {
      // RVA: SC always invalidates the reservation, success or fail.
      amo_unit_.clear(hid, line_addr);
    }

    if (do_store) {
      // Merge the new word at byte_off into the line.
      auto store_block = make_mem_block();
      std::memset(store_block->data(), 0, store_block->size());
      amo_store_word(store_block->data(), byte_off, width, rmw.new_word);
      uint64_t byteen = amo_byteen(byte_off, width);
      line_merge(hit_line, store_block, byteen);
      if (config_.write_back) {
        hit_line.dirty = true;
      } else {
        // Write-through: emit a write of the merged word downstream.
        MemReq w;
        w.addr   = params_.mem_addr(bank_id_, set_id, params_.addr_tag(bank_req.addr));
        w.write  = true;
        w.cid    = bank_req.cid;
        w.uuid   = bank_req.uuid;
        w.data   = store_block;
        w.byteen = byteen;
        this->mem_req_out.send(w);
        DT(3, this->name() << " amo-writethrough: " << w);
      }
      // Break other harts' reservations on this line (proposal §3.9).
      amo_unit_.invalidate(line_addr, /*except=*/hid);
    }

    // Send AMO response.
    MemRsp rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
    rsp.data = rsp_block;
    this->core_rsp_out.send(rsp);
    DT(3, this->name() << " amo-rsp op=" << op << " sc_fail=" << sc_fail << ": " << rsp);
    return true;
  }
#endif

  // Pipeline tail: process the head of pipe_req_ — at most one bank_req_t per tick.
  void processRequests() {
    const bank_req_t &bank_req = pipe_req_->peek();

    auto need_core_rsp = [&](const bank_req_t &req) {
      return (!req.write || config_.write_reponse);
    };

    switch (bank_req.type) {
    case bank_req_t::None:
      pipe_req_->pop();
      return;

#if EXT_A_ENABLED
    case bank_req_t::AmoProbe: {
      // Non-LLC AMO passthrough (proposal §3.8). Probe the local line
      // first so any cached copy doesn't shadow the LLC's view: dirty
      // line → writeback (so the LLC's bytes are fresh before the AMO
      // RMW commits there); any hit → invalidate (so the next normal
      // load takes a fresh miss after the AMO completes). Then forward
      // the original AMO MemReq downstream tagged with
      // AMO_PASSTHRU_TAG_FLAG so the response routes back to
      // core_rsp_out without installing a fill.
      assert(!config_.is_llc && "AmoProbe at LLC is a wiring bug");
      uint32_t set_id   = params_.addr_set_id(bank_req.addr);
      uint64_t addr_tag = params_.addr_tag(bank_req.addr);
      auto &set         = sets_.at(set_id);
      int32_t free_id = -1, repl_id = 0;
      int hit_id = set.tag_match(addr_tag, config_.repl_policy, rand_ctr_, &free_id, &repl_id);
      const bool hit   = (hit_id != -1);
      const bool dirty = hit && set.lines.at(hit_id).valid && set.lines.at(hit_id).dirty;

      // Stall checks: writeback (if dirty) AND the AMO forward both
      // need an mem_req_out slot. They serialize across two cycles
      // when the FIFO can't hold both, mirroring the proposal's
      // hit-dirty stall note (§3.8).
      const uint32_t out_slots_needed = (dirty ? 1u : 0u) + 1u;
      if (this->mem_req_out.size() + out_slots_needed > this->mem_req_out.capacity()) {
        return; // stall
      }
      // Pre-allocate a passthru-table slot before mutating state.
      uint32_t pid = AMO_PASSTHRU_CAP;
      for (uint32_t i = 0; i < AMO_PASSTHRU_CAP; ++i) {
        if (!amo_passthru_.at(i).valid) { pid = i; break; }
      }
      if (pid == AMO_PASSTHRU_CAP) {
        return; // stall — table is full (input gate should normally prevent this)
      }

      if (dirty) {
        auto &line = set.lines.at(hit_id);
        MemReq wb;
        wb.addr   = params_.mem_addr(bank_id_, set_id, line.tag);
        wb.write  = true;
        wb.cid    = bank_req.cid;
        wb.uuid   = bank_req.uuid;
        wb.data   = line.data;
        wb.byteen = ~uint64_t(0) >> (64 - MEM_BLOCK_SIZE);
        this->mem_req_out.send(wb);
        DT(3, this->name() << " amo-probe-wb: " << wb);
        ++perf_stats_.evictions;
      }
      if (hit) {
        auto &line = set.lines.at(hit_id);
        line.valid = false;
        line.dirty = false;
      }

      // Forward AMO downstream. Tag is rewritten so the response
      // doesn't collide with the MSHR tag namespace.
      auto &e = amo_passthru_.at(pid);
      e.valid   = true;
      e.req_tag = bank_req.req_tag;
      e.cid     = bank_req.cid;
      e.uuid    = bank_req.uuid;
      MemReq amo_fwd;
      amo_fwd.addr   = bank_req.addr;
      amo_fwd.write  = false;
      amo_fwd.tag    = amo_passthru_tag_base() + pid;
      amo_fwd.cid    = bank_req.cid;
      amo_fwd.uuid   = bank_req.uuid;
      amo_fwd.amo    = bank_req.amo;
      amo_fwd.op     = amo_to_memop(bank_req.amo.op);
      this->mem_req_out.send(amo_fwd);
      DT(3, this->name() << " amo-probe-fwd: " << amo_fwd);
      ++pending_fill_reqs_; // counts as an outstanding mem-roundtrip for perf
      pipe_req_->pop();
      return;
    }
#endif

    case bank_req_t::Fill: {
      // Install the new line. Replay is mutex with fill (priority gating in
      // processInputs), so any line we evict here has no in-flight replay.
      uint32_t set_id   = params_.addr_set_id(bank_req.addr);
      uint64_t addr_tag = params_.addr_tag(bank_req.addr);
      auto &set         = sets_.at(set_id);
      int32_t free_id   = -1, repl_id = 0;
      int32_t victim_id = set.select_victim(config_.repl_policy, rand_ctr_, &free_id, &repl_id);
      auto &victim_line = set.lines.at(victim_id);

      // Stall if a writeback is needed and the egress queue can't accept it.
      const bool need_writeback = config_.write_back && victim_line.valid && victim_line.dirty;
      if (need_writeback && this->mem_req_out.full())
        return; // stall

      if (config_.repl_policy == Cache::FIFO) {
        set.fifo_ptr = (set.fifo_ptr + 1) % set.lines.size();
      } else if (config_.repl_policy == Cache::RANDOM) {
        ++rand_ctr_;
      }
      if (need_writeback) {
        MemReq wb;
        wb.addr   = params_.mem_addr(bank_id_, set_id, victim_line.tag);
        wb.write  = true;
        wb.cid    = bank_req.cid;
        wb.uuid   = bank_req.uuid;
        wb.data   = victim_line.data;
        wb.byteen = ~uint64_t(0) >> (64 - MEM_BLOCK_SIZE);
        this->mem_req_out.send(wb);
        DT(3, this->name() << " writeback: " << wb);
        ++perf_stats_.evictions;
      }
      victim_line.valid   = true;
      victim_line.tag     = addr_tag;
      victim_line.lru_ctr = 0;
      victim_line.dirty   = false;
      victim_line.data    = bank_req.data;
      mshr_.replay(bank_req.mshr_id);
      pipe_req_->pop();
    } break;

    case bank_req_t::Replay: {
      // Replay invariant: the line is present. Fill is mutex with replay in
      // input arbitration (a fill cannot preempt a pending replay's line).
      uint32_t set_id   = params_.addr_set_id(bank_req.addr);
      uint64_t addr_tag = params_.addr_tag(bank_req.addr);
      auto &set         = sets_.at(set_id);
      int32_t free_id = -1, repl_id = 0;
      int hit_id = set.tag_match(addr_tag, config_.repl_policy, rand_ctr_, &free_id, &repl_id);
      assert(hit_id != -1 && "replay miss");

#if EXT_A_ENABLED
      if (bank_req.amo.valid) {
        assert(config_.is_llc && "AMO replay reached non-LLC bank");
        if (!this->commitAmo(bank_req, set, hit_id, set_id))
          return; // stall
        if (config_.repl_policy == Cache::PLRU)
          set.update_lru(hit_id);
        pipe_req_->pop();
        return;
      }
#endif

      if (need_core_rsp(bank_req) && this->core_rsp_out.full())
        return; // stall

      auto &hit_line = set.lines.at(hit_id);
      if (bank_req.write) {
        // Write-through replay only exists for the wt-merge case: the core
        // response and the memory write were already issued at miss time.
        // Guard so a future change doesn't silently drop a store.
        if (!config_.write_back) {
          assert(bank_req.skip_core_rsp && "WT replay without pre-sent store");
        }
        line_merge(hit_line, bank_req.data, bank_req.byteen);
        if (config_.write_back)
          hit_line.dirty = true;
#if EXT_A_ENABLED
        // Write-back write-miss replay: this is a store from above
        // (always WT above the LLC, per §3.1.5) reaching the LLC tag
        // array → break other harts' reservations on the line.
        // For the WT wt-merge replay, invalidation already fired at
        // miss time when the writethrough was emitted.
        if (config_.is_llc && config_.write_back) {
          uint64_t line_addr = (bank_req.addr >> config_.L) << config_.L;
          amo_unit_.invalidate(line_addr, /*except=*/bank_req.amo.hart_id);
        }
#endif
      }
      if (need_core_rsp(bank_req) && !bank_req.skip_core_rsp) {
        MemRsp rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
        if (!bank_req.write)
          rsp.data = hit_line.data;
        this->core_rsp_out.send(rsp);
        DT(3, this->name() << " replay-rsp: " << rsp);
      }
      if (config_.repl_policy == Cache::PLRU)
        set.update_lru(hit_id);

      pipe_req_->pop();
    } break;

    case bank_req_t::Core: {
      uint32_t set_id   = params_.addr_set_id(bank_req.addr);
      uint64_t addr_tag = params_.addr_tag(bank_req.addr);
      auto &set         = sets_.at(set_id);

      int32_t free_id = -1, repl_id = 0;
      // Pure tag match — no LRU mutation here. update_lru() commits below,
      // after all stall checks pass, otherwise PLRU drifts on retry.
      int hit_id = set.tag_match(addr_tag, config_.repl_policy, rand_ctr_, &free_id, &repl_id);

      if (hit_id != -1) {
#if EXT_A_ENABLED
        if (bank_req.amo.valid) {
          assert(config_.is_llc && "AMO Core+hit reached non-LLC bank");
          if (!this->commitAmo(bank_req, set, hit_id, set_id))
            return; // stall
          if (config_.repl_policy == Cache::PLRU)
            set.update_lru(hit_id);
          --pending_mshr_size_;
          pipe_req_->pop();
          return;
        }
#endif
        // Cache hit.
        const bool need_rsp = need_core_rsp(bank_req);
        const bool need_mem = bank_req.write && !config_.write_back;
        if (need_mem && this->mem_req_out.full())
          return;
        if (need_rsp && this->core_rsp_out.full())
          return;

        auto &hit_line = set.lines.at(hit_id);
        if (bank_req.write) {
          line_merge(hit_line, bank_req.data, bank_req.byteen);
          if (config_.write_back) {
            hit_line.dirty = true;
          } else {
            MemReq w;
            w.addr   = params_.mem_addr(bank_id_, set_id, addr_tag);
            w.write  = true;
            w.cid    = bank_req.cid;
            w.uuid   = bank_req.uuid;
            w.data   = bank_req.data;
            w.byteen = bank_req.byteen;
            this->mem_req_out.send(w);
            DT(3, this->name() << " writethrough: " << w);
#if EXT_A_ENABLED
            // Writethrough commit reaches the LLC tag array → break
            // other harts' reservations (proposal §3.9).
            if (config_.is_llc) {
              uint64_t line_addr = (bank_req.addr >> config_.L) << config_.L;
              amo_unit_.invalidate(line_addr, /*except=*/bank_req.amo.hart_id);
            }
#endif
          }
#if EXT_A_ENABLED
          // Write-back write-hit: also visible to LLC tag array since
          // we mutated this LLC line. Break other harts' reservations.
          if (config_.is_llc && config_.write_back) {
            uint64_t line_addr = (bank_req.addr >> config_.L) << config_.L;
            amo_unit_.invalidate(line_addr, /*except=*/bank_req.amo.hart_id);
          }
#endif
        }
        if (need_rsp) {
          MemRsp rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
          if (!bank_req.write)
            rsp.data = hit_line.data;
          this->core_rsp_out.send(rsp);
          DT(3, this->name() << " core-rsp: " << rsp);
        }
      } else {
        // Cache miss.
        if (bank_req.write && !config_.write_back) {
          // Write-through write miss: forward store to memory and respond
          // immediately. No fill / no MSHR allocation (this store doesn't
          // wait for a line). If a fill for this line is already in flight
          // from an earlier miss, also enqueue a wt-merge replay so the
          // bytes get folded into the line once the fill arrives — without
          // this, a subsequent read could see the pre-store fill data.
          uint32_t pending_root_id = 0;
          bool fill_pending = mshr_.lookup(set_id, addr_tag, &pending_root_id);

          const bool need_rsp = need_core_rsp(bank_req);
          if (this->mem_req_out.full())
            return;
          if (need_rsp && this->core_rsp_out.full())
            return;
          // wt-merge needs an MSHR slot; processInputs's MSHR gate doesn't
          // reserve one for write-through writes. Stall rather than abort.
          if (fill_pending && mshr_.full())
            return;

          MemReq w;
          w.addr   = params_.mem_addr(bank_id_, set_id, addr_tag);
          w.write  = true;
          w.cid    = bank_req.cid;
          w.uuid   = bank_req.uuid;
          w.data   = bank_req.data;
          w.byteen = bank_req.byteen;
          this->mem_req_out.send(w);
          DT(3, this->name() << " writethrough: " << w);

#if EXT_A_ENABLED
          // Writethrough write-miss is still a "store from above" reaching
          // the LLC bank → break other harts' reservations (proposal §3.9).
          if (config_.is_llc) {
            uint64_t line_addr = (bank_req.addr >> config_.L) << config_.L;
            amo_unit_.invalidate(line_addr, /*except=*/bank_req.amo.hart_id);
          }
#endif

          if (need_rsp) {
            MemRsp rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
            this->core_rsp_out.send(rsp);
            DT(3, this->name() << " core-rsp: " << rsp);
          }
          if (fill_pending) {
            bank_req_t merge_req = bank_req;
            merge_req.skip_core_rsp = true;
            mshr_.enqueue(merge_req, set_id, addr_tag);
            DT(3, this->name() << " mshr-enqueue (wt-merge): " << bank_req);
          }
        } else {
          // Read miss, or write-back write miss → MSHR-backed.
          // First miss for this line sends the fill request; subsequent
          // misses to the same line just chain into the existing entry.
          uint32_t root_id = 0;
          bool mshr_pending = mshr_.lookup(set_id, addr_tag, &root_id);
          if (!mshr_pending && this->mem_req_out.full())
            return;

          assert(!mshr_.full());
          int mshr_id = mshr_.enqueue(bank_req, set_id, addr_tag);
          DT(3, this->name() << " mshr-enqueue: " << bank_req);
          if (!mshr_pending) {
            MemReq fill;
            fill.addr  = params_.mem_addr(bank_id_, set_id, addr_tag);
            fill.write = false;
            fill.tag   = mshr_id; // routes the fill response back here
            fill.cid   = bank_req.cid;
            fill.uuid  = bank_req.uuid;
            this->mem_req_out.send(fill);
            DT(3, this->name() << " fill-req: " << fill);
            ++pending_fill_reqs_;
          }
        }
        if (bank_req.write) ++perf_stats_.write_misses;
        else                ++perf_stats_.read_misses;
      }

      if (config_.repl_policy == Cache::PLRU)
        set.update_lru(hit_id);
      --pending_mshr_size_;
      pipe_req_->pop();
    } break;

    default:
      std::abort();
    }
  }

  void processFlush() {
    // Wait for in-flight requests to drain before walking lines, otherwise an
    // outstanding fill could install a fresh line behind our scan and leave
    // a dirty victim un-evicted.
    if (pending_fill_reqs_ != 0
     || !pipe_req_->empty()
     || !mshr_.empty()) {
      return;
    }
    while (flush_set_idx_ < params_.sets_per_bank) {
      auto &set = sets_.at(flush_set_idx_);
      while (flush_way_idx_ < set.lines.size()) {
        auto &line = set.lines.at(flush_way_idx_);
        if (line.valid && line.dirty) {
          if (this->mem_req_out.full())
            return; // stall — try again next cycle
          MemReq mem_req;
          mem_req.addr = params_.mem_addr(bank_id_, flush_set_idx_, line.tag);
          mem_req.write = true;
          mem_req.data = line.data;
          mem_req.byteen = ~uint64_t(0) >> (64 - MEM_BLOCK_SIZE);
          this->mem_req_out.send(mem_req);
          DT(3, this->name() << " flush-wb: " << mem_req);
          ++perf_stats_.evictions;
          line.dirty = false;
        }
        ++flush_way_idx_;
      }
      ++flush_set_idx_;
      flush_way_idx_ = 0;
    }
    flushing_ = false;
    DT(3, this->name() << " flush-done");
  }

  Cache::Config config_;
  params_t params_;
  uint32_t bank_id_;

  std::vector<set_t> sets_;
  MSHR mshr_;
  uint32_t pending_mshr_size_;
  TFifo<bank_req_t>::Ptr pipe_req_;

  Cache::PerfStats perf_stats_;

  uint64_t pending_read_reqs_;
  uint64_t pending_write_reqs_;
  uint64_t pending_fill_reqs_;
  uint32_t rand_ctr_;

  // Flush walk state.
  bool     flushing_;
  uint32_t flush_set_idx_;
  uint32_t flush_way_idx_;

#if EXT_A_ENABLED
  AmoUnit  amo_unit_;

  // Non-LLC AMO passthrough table. When config_.is_llc==false, AMOs
  // probe-and-invalidate the local line, then forward the MemReq
  // downstream via mem_req_out. The response comes back via
  // mem_rsp_in but must NOT install a fill — it's a load-style AMO
  // return that gets forwarded straight to core_rsp_out.
  //
  // Identification: arbiters along the path mangle the tag
  // (req.tag = (req.tag << shift) | input_id) and unshift on the
  // return — the bank-side tag is preserved across the round-trip.
  // Bit-flag schemes don't survive (uint32_t can overflow if shifts
  // accumulate past 32), so we partition the tag NAMESPACE: fill
  // responses use tag in [0, mshr_capacity), passthru responses use
  // tag in [mshr_capacity, mshr_capacity + AMO_PASSTHRU_CAP). The
  // arbiter only adds bits at the LSB, so the partition boundary
  // remains intact.
  static constexpr uint32_t AMO_PASSTHRU_CAP = 8;
  struct amo_passthru_entry_t {
    bool     valid   = false;
    uint64_t req_tag = 0;
    uint32_t cid     = 0;
    uint64_t uuid    = 0;
  };
  std::array<amo_passthru_entry_t, AMO_PASSTHRU_CAP> amo_passthru_;
  uint32_t amo_passthru_tag_base() const { return mshr_.capacity(); }
#endif

  friend class SimObject<CacheBank>;
};

///////////////////////////////////////////////////////////////////////////////

class Cache::Impl {
public:
  Impl(Cache *simobject, const Config &config)
      : simobject_(simobject), config_(config), params_(config), banks_(1 << config.B), nc_mem_arbs_(config.mem_ports) {
    char sname[100];

    uint32_t num_banks = (1 << config.B);

    if (config_.bypass) {
      snprintf(sname, 100, "%s-bypass_arb", simobject->name().c_str());
      auto bypass_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin, config_.num_inputs, config_.mem_ports);
      for (uint32_t i = 0; i < config_.num_inputs; ++i) {
        simobject->core_req_in.at(i).bind(&bypass_arb->ReqIn.at(i));
        bypass_arb->RspOut.at(i).bind(&simobject->core_rsp_out.at(i));
      }
      for (uint32_t i = 0; i < config_.mem_ports; ++i) {
        bypass_arb->ReqOut.at(i).bind(&simobject->mem_req_out.at(i));
        simobject->mem_rsp_in.at(i).bind(&bypass_arb->RspIn.at(i));
      }
      return;
    }

    // create non-cacheable arbiter
    for (uint32_t i = 0; i < config_.mem_ports; ++i) {
      snprintf(sname, 100, "%s-nc_arb%d", simobject->name().c_str(), i);
      nc_mem_arbs_.at(i) = MemArbiter::Create(sname, ArbiterType::Priority, 2, 1);
    }

    // Connect non-cacheable arbiter output port 0 to outgoing memory ports
    for (uint32_t i = 0; i < config_.mem_ports; ++i) {
      nc_mem_arbs_.at(i)->ReqOut.at(0).bind(&simobject->mem_req_out.at(i));
      simobject->mem_rsp_in.at(i).bind(&nc_mem_arbs_.at(i)->RspIn.at(0));
    }

    // Create bank's memory arbiter
    snprintf(sname, 100, "%s-mem_arb", simobject->name().c_str());
    auto bank_mem_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin, num_banks, config_.mem_ports);

    // Connect bank's memory arbiter to non-cacheable arbiter's input port 0
    for (uint32_t i = 0; i < config_.mem_ports; ++i) {
      bank_mem_arb->ReqOut.at(i).bind(&nc_mem_arbs_.at(i)->ReqIn.at(0));
      nc_mem_arbs_.at(i)->RspOut.at(0).bind(&bank_mem_arb->RspIn.at(i));
    }

    // Create bank's core crossbar
    snprintf(sname, 100, "%s-core_xbar", simobject->name().c_str());
    bank_core_xbar_ = MemCrossBar::Create(sname, ArbiterType::RoundRobin, config_.num_inputs, num_banks,
                                          [&](const MemCrossBar::ReqType &req) {
                                            return params_.addr_bank_id(req.addr);
                                          });

    // Create cache banks
    for (uint32_t i = 0, n = num_banks; i < n; ++i) {
      snprintf(sname, 100, "%s-bank%d", simobject->name().c_str(), i);
      banks_.at(i) = CacheBank::Create(sname, config, params_, i);

      // bind core ports
      bank_core_xbar_->ReqOut.at(i).bind(&banks_.at(i)->core_req_in);
      banks_.at(i)->core_rsp_out.bind(&bank_core_xbar_->RspIn.at(i));

      // bind memory ports
      banks_.at(i)->mem_req_out.bind(&bank_mem_arb->ReqIn.at(i));
      bank_mem_arb->RspOut.at(i).bind(&banks_.at(i)->mem_rsp_in);
    }
  }

  void reset() {
    if (config_.bypass)
      return;
    // calculate cache initialization cycles
    init_cycles_ = params_.sets_per_bank;
  }

  void tick() {
    if (config_.bypass)
      return;

    // wait on cache initialization cycles
    if (init_cycles_ != 0) {
      --init_cycles_;
      DT(3, simobject_->name() << " init: line=" << init_cycles_);
      return;
    }

    // handle cache bypasss responses
    for (uint32_t i = 0, n = config_.mem_ports; i < n; ++i) {
      // Forward non-cacheable arbiter's output (1) to core response ports
      auto &bypass = nc_mem_arbs_.at(i)->RspOut.at(1);
      if (!bypass.empty()) {
        auto &mem_rsp = bypass.peek();
        if (this->processBypassResponse(mem_rsp)) {
          bypass.pop();
        }
      }
    }

    // schedule core responses
    for (uint32_t req_id = 0, n = config_.num_inputs; req_id < n; ++req_id) {
      auto &bank_rsp = bank_core_xbar_->RspOut.at(req_id);
      if (bank_rsp.empty())
        continue;
      auto &core_rsp = bank_rsp.peek();
      if (simobject_->core_rsp_out.at(req_id).try_send(core_rsp, 0)) {
        DT(3, simobject_->name() << " core-rsp: " << core_rsp);
        bank_rsp.pop();
      }
    }

    // schedule core requests
    for (uint32_t req_id = 0, n = config_.num_inputs; req_id < n; ++req_id) {
      auto &core_req_in = simobject_->core_req_in.at(req_id);
      if (core_req_in.empty())
        continue;
      auto &core_req = core_req_in.peek();
      if (core_req.type == AddrType::IO) {
        if (this->processBypassRequest(core_req, req_id)) {
          core_req_in.pop();
        }
      } else {
        if (bank_core_xbar_->ReqIn.at(req_id).try_send(core_req, 0)) {
          core_req_in.pop();
        }
      }
    }
  }

  PerfStats perf_stats() const {
    PerfStats perf_stats;
    if (!config_.bypass) {
      for (const auto &bank : banks_) {
        perf_stats += bank->perf_stats();
      }
      perf_stats.bank_stalls = bank_core_xbar_->collisions();
    }
    return perf_stats;
  }

  void flush_begin() {
    if (config_.bypass) return;
    for (auto &bank : banks_) {
      bank->flush_begin();
    }
  }

  bool flush_done() const {
    if (config_.bypass) return true;
    for (const auto &bank : banks_) {
      if (!bank->flush_done()) return false;
    }
    return true;
  }

private:

  bool processBypassResponse(const MemRsp &mem_rsp) {
    // core response backpressure check
    uint32_t req_id = mem_rsp.tag & ((1 << params_.log2_num_inputs) - 1);
    if (simobject_->core_rsp_out.at(req_id).full())
      return false; // stall
    uint64_t tag = mem_rsp.tag >> params_.log2_num_inputs;
    MemRsp core_rsp{tag, mem_rsp.cid, mem_rsp.uuid};
    core_rsp.data = mem_rsp.data;  // forward TLM payload through bypass
    simobject_->core_rsp_out.at(req_id).send(core_rsp, 0);
    DT(3, simobject_->name() << " bypass-core-rsp: " << core_rsp);
    return true;
  }

  bool processBypassRequest(const MemReq &core_req, uint32_t req_id) {
    // Push core request to non-cacheable arbiter's input (1)
    uint32_t mem = req_id % config_.mem_ports;
    if (nc_mem_arbs_.at(mem)->ReqIn.at(1).full())
      return false; // stall
    MemReq mem_req(core_req);
    mem_req.tag = (core_req.tag << params_.log2_num_inputs) + req_id;
    nc_mem_arbs_.at(mem)->ReqIn.at(1).send(mem_req, 0);
    DT(3, simobject_->name() << " bypass-dram-req: " << mem_req);
    return true;
  }

  Cache *const simobject_;
  Config config_;
  params_t params_;
  std::vector<CacheBank::Ptr> banks_;
  MemArbiter::Ptr bank_arb_;
  std::vector<MemArbiter::Ptr> nc_mem_arbs_;
  MemCrossBar::Ptr bank_core_xbar_;
  uint32_t init_cycles_;
};

///////////////////////////////////////////////////////////////////////////////

Cache::Cache(const SimContext &ctx, const char *name, const Config &config)
    : SimObject<Cache>(ctx, name), core_req_in(config.num_inputs, this), core_rsp_out(config.num_inputs, this), mem_req_out(config.mem_ports, this), mem_rsp_in(config.mem_ports, this), impl_(new Impl(this, config)) {}

Cache::~Cache() {
  delete impl_;
}

void Cache::on_reset() {
  impl_->reset();
}

void Cache::on_tick() {
  impl_->tick();
}

Cache::PerfStats Cache::perf_stats() const {
  return impl_->perf_stats();
}

void Cache::flush_begin() {
  impl_->flush_begin();
}

bool Cache::flush_done() const {
  return impl_->flush_done();
}