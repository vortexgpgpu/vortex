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
    Core = 3
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

  // Mark all pending requests matching the entry's tag for replay
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
      : SimObject<CacheBank>(ctx, name), core_req_in(this), core_rsp_out(this), mem_req_out(this), mem_rsp_in(this), config_(config), params_(params), bank_id_(bank_id), sets_(params.sets_per_bank, params.lines_per_set), mshr_(config.mshr_size), pipe_req_(TFifo<bank_req_t>::Create("", config.latency)), rand_ctr_(0) {
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
    inflight_replays_per_set_.assign(params_.sets_per_bank, 0);
    flushing_ = false;
    flush_set_idx_ = 0;
    flush_way_idx_ = 0;
  }

  void on_tick() {
    // process input requests
    this->processInputs();

    // process pipeline requests
    this->processRequests();

    // calculate memory latency
    perf_stats_.mem_latency += pending_fill_reqs_;

    // flush walk: emit writebacks for dirty lines.
    if (flushing_) {
      this->processFlush();
    }
  }

private:
  void processInputs() {
    // Step 1: drain mem_rsp_in out-of-band. Fills bypass pipe_req_ — a
    // stalled Replay at pipe_req_'s head must not block fill processing,
    // else MSHR locks and the cache→adapter→coalescer→LSU chain deadlocks
    // under high warp density.
    if (!this->mem_rsp_in.empty()) {
      auto &mem_rsp = this->mem_rsp_in.peek();
      uint32_t mshr_id = mem_rsp.tag;
      const auto &root_peek = mshr_.peek(mshr_id);
      uint32_t fill_set_id = root_peek.set_id;
      bool fill_blocked = (inflight_replays_per_set_.at(fill_set_id) > 0)
                       || (config_.write_back && this->mem_req_out.full());
      if (!fill_blocked) {
        auto &root_entry = mshr_.replay(mshr_id);
        auto &set = sets_.at(fill_set_id);
        int32_t free_line_id = -1;
        int32_t repl_line_id = 0;
        int32_t victim_line_id = set.select_victim(config_.repl_policy, rand_ctr_, &free_line_id, &repl_line_id);
        if (config_.repl_policy == Cache::FIFO) {
          set.fifo_ptr = (set.fifo_ptr + 1) % set.lines.size();
        } else if (config_.repl_policy == Cache::RANDOM) {
          ++rand_ctr_;
        }
        auto &victim_line = set.lines.at(victim_line_id);
        if (config_.write_back && victim_line.valid && victim_line.dirty) {
          MemReq mem_req;
          mem_req.addr = params_.mem_addr(bank_id_, fill_set_id, victim_line.tag);
          mem_req.write = true;
          mem_req.cid = root_entry.bank_req.cid;
          mem_req.uuid = root_entry.bank_req.uuid;
          mem_req.data = victim_line.data;
          mem_req.byteen = ~uint64_t(0) >> (64 - MEM_BLOCK_SIZE);
          this->mem_req_out.send(mem_req);
          DT(3, this->name() << " writeback: " << mem_req);
          ++perf_stats_.evictions;
        }
        victim_line.valid = true;
        victim_line.tag = root_entry.addr_tag;
        victim_line.lru_ctr = 0;
        victim_line.dirty = false;
        victim_line.data = mem_rsp.data;
        DT(3, this->name() << " fill-rsp: " << mem_rsp);
        this->mem_rsp_in.pop();
        --pending_fill_reqs_;
      }
    }

    // Step 2: schedule pipeline inputs (replay > core_req) into pipe_req_.
    if (pipe_req_->full())
      return;

    // schedule MSHR replay
    if (mshr_.has_ready_reqs()) {
      bank_req_t bank_req;
      mshr_.dequeue(&bank_req);
      uint32_t set_id = params_.addr_set_id(bank_req.addr);
      ++inflight_replays_per_set_.at(set_id);
      pipe_req_->push(bank_req);
      return;
    }

    // schedule core request
    if (!this->core_req_in.empty()) {
      auto &core_req = this->core_req_in.peek();
      // check MSHR occupancy (conservative: any request that may miss and use MSHR)
      bool use_mshr = !core_req.write || config_.write_back;
      if (use_mshr && ((mshr_.size() + pending_mshr_size_) >= mshr_.capacity())) {
        ++perf_stats_.mshr_stalls;
        return; // stall
      }
      bank_req_t bank_req;
      bank_req.reset();
      bank_req.type = bank_req_t::Core;
      bank_req.addr = core_req.addr;
      bank_req.cid = core_req.cid;
      bank_req.uuid = core_req.uuid;
      bank_req.req_tag = core_req.tag;
      bank_req.write = core_req.write;
      bank_req.data = core_req.data;
      bank_req.byteen = core_req.byteen;
      pipe_req_->push(bank_req);
      DT(3, this->name() << " core-req: " << core_req);
      ++pending_mshr_size_;
      if (core_req.write)
        ++perf_stats_.writes;
      else
        ++perf_stats_.reads;
      this->core_req_in.pop();
      return;
    }
  }

  void processRequests() {
    if (pipe_req_->empty())
      return;

    const bank_req_t &bank_req = pipe_req_->peek();

    auto need_core_rsp = [&](const bank_req_t &req) {
      return (!req.write || config_.write_reponse);
    };

    switch (bank_req.type) {
    case bank_req_t::None:
      break;

    case bank_req_t::Replay: {
      // Check core output backpressure first — no mutation before all stalls clear.
      if (need_core_rsp(bank_req) && this->core_rsp_out.full())
        return; // stall

      uint32_t set_id = params_.addr_set_id(bank_req.addr);
      uint64_t addr_tag = params_.addr_tag(bank_req.addr);

      auto &set = sets_.at(set_id);
      int32_t free_line_id = -1;
      int32_t repl_line_id = 0;
      int hit_line_id = set.tag_match(addr_tag, config_.repl_policy, rand_ctr_, &free_line_id, &repl_line_id);
      assert(hit_line_id != -1);

      auto &hit_line = set.lines.at(hit_line_id);
      if (bank_req.write) {
        // Write-through Replay is only used for the wt-merge path (the core
        // response and memory write were already issued at miss time). Guard
        // the invariant so a future code change doesn't silently drop a store.
        if (!config_.write_back) {
          assert(bank_req.skip_core_rsp && "WT replay without pre-sent store");
        }
        // Write-miss completed by Fill; Replay completes the store by merging
        // the bytes into the (newly filled) line. Mark dirty only for write-back.
        line_merge(hit_line, bank_req.data, bank_req.byteen);
        if (config_.write_back)
          hit_line.dirty = true;
      }

      if (need_core_rsp(bank_req) && !bank_req.skip_core_rsp) {
        MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
        if (!bank_req.write) {
          core_rsp.data = hit_line.data;
        }
        this->core_rsp_out.send(core_rsp);
        DT(3, this->name() << " replay: " << core_rsp);
      }

      // Commit LRU update last — Replay never restalls past this point.
      if (config_.repl_policy == Cache::PLRU)
        set.update_lru(hit_line_id);
    } break;

    case bank_req_t::Core: {
      uint32_t set_id = params_.addr_set_id(bank_req.addr);
      uint64_t addr_tag = params_.addr_tag(bank_req.addr);

      auto &set = sets_.at(set_id);

      int32_t free_line_id = -1;
      int32_t repl_line_id = 0;
      // Pure tag match — no LRU mutation. update_lru() runs only after all
      // stall checks pass, otherwise PLRU counters drift on every retry.
      int hit_line_id = set.tag_match(addr_tag, config_.repl_policy, rand_ctr_, &free_line_id, &repl_line_id);

      if (hit_line_id != -1) {
        //
        // Hit handling
        //
        // Gather all stall conditions BEFORE any mutation. Otherwise a stall
        // after line_merge or mem_req_out.send leaves the request in the pipe
        // and replays them on the next tick, causing duplicate writes / merges.
        const bool need_rsp = need_core_rsp(bank_req);
        const bool need_mem = bank_req.write && !config_.write_back;
        if (need_mem && this->mem_req_out.full())
          return; // stall
        if (need_rsp && this->core_rsp_out.full())
          return; // stall

        auto &hit_line = set.lines.at(hit_line_id);
        if (bank_req.write) {
          line_merge(hit_line, bank_req.data, bank_req.byteen);
          if (!config_.write_back) {
            MemReq mem_req;
            mem_req.addr = params_.mem_addr(bank_id_, set_id, addr_tag);
            mem_req.write = true;
            mem_req.cid = bank_req.cid;
            mem_req.uuid = bank_req.uuid;
            mem_req.data = bank_req.data;
            mem_req.byteen = bank_req.byteen;
            this->mem_req_out.send(mem_req);
            DT(3, this->name() << " writethrough: " << mem_req);
          } else {
            hit_line.dirty = true;
          }
        }

        if (need_rsp) {
          MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
          if (!bank_req.write) {
            core_rsp.data = hit_line.data;
          }
          this->core_rsp_out.send(core_rsp);
          DT(3, this->name() << " core-rsp: " << core_rsp);
        }
      } else {
        //
        // Miss handling
        //
        // Write-through miss: forward store to memory and respond immediately (no fill/MSHR).
        // Special case: if the line is currently being filled via a pending MSHR
        // entry, also enqueue this store so Replay applies line_merge after fill.
        // Without this, the in-flight fill captures pre-store bytes and any
        // subsequent read of this line returns stale data.
        if (bank_req.write && !config_.write_back) {
          uint32_t pending_root_id = 0;
          bool fill_pending = mshr_.lookup(set_id, addr_tag, &pending_root_id);

          const bool need_rsp = need_core_rsp(bank_req);
          // All stall checks first — anything past this point commits.
          if (this->mem_req_out.full())
            return; // stall
          if (need_rsp && this->core_rsp_out.full())
            return; // stall
          // The wt-merge replay enqueue needs an MSHR slot, but processInputs()
          // does not reserve one for write-through writes. Stall here rather
          // than asserting at enqueue.
          if (fill_pending && mshr_.full())
            return; // stall

          {
            MemReq mem_req;
            mem_req.addr = params_.mem_addr(bank_id_, set_id, addr_tag);
            mem_req.write = true;
            mem_req.cid = bank_req.cid;
            mem_req.uuid = bank_req.uuid;
            mem_req.data = bank_req.data;
            mem_req.byteen = bank_req.byteen;
            this->mem_req_out.send(mem_req);
            DT(3, this->name() << " writethrough: " << mem_req);
          }

          if (need_rsp) {
            MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
            this->core_rsp_out.send(core_rsp);
            DT(3, this->name() << " core-rsp: " << core_rsp);
          }

          if (fill_pending) {
            // Enqueue a no-response replay-write so line_merge runs post-fill.
            // The core response was already sent above; mark this entry to skip
            // it on replay.
            bank_req_t merge_req = bank_req;
            merge_req.skip_core_rsp = true;
            mshr_.enqueue(merge_req, set_id, addr_tag);
            DT(3, this->name() << " mshr-enqueue (wt-merge): " << bank_req);
          }
        } else {
          // MSHR-backed miss (read miss, or write-back write miss).
          uint32_t root_id = 0;
          bool mshr_pending = mshr_.lookup(set_id, addr_tag, &root_id);

          // If we are the first miss for this block, we should send the fill request this cycle.
          if (!mshr_pending && this->mem_req_out.full())
            return; // stall

          // Allocate an MSHR entry for this request.
          assert(!mshr_.full());
          int mshr_id = mshr_.enqueue(bank_req, set_id, addr_tag);
          DT(3, this->name() << " mshr-enqueue: " << bank_req);

          if (!mshr_pending) {
            MemReq mem_req;
            mem_req.addr = params_.mem_addr(bank_id_, set_id, addr_tag);
            mem_req.write = false;
            mem_req.tag = mshr_id; // root id used to route the fill response
            mem_req.cid = bank_req.cid;
            mem_req.uuid = bank_req.uuid;
            this->mem_req_out.send(mem_req);
            DT(3, this->name() << " fill-req: " << mem_req);
            ++pending_fill_reqs_;
          }
        }

        // Update performance stats
        if (bank_req.write)
          ++perf_stats_.write_misses;
        else
          ++perf_stats_.read_misses;
      }

      // Commit LRU update last — request is past all stall points and will pop.
      // hit_line_id == -1 (miss) ages all valid lines without resetting any.
      if (config_.repl_policy == Cache::PLRU)
        set.update_lru(hit_line_id);

      // Update pending MSHR size
      --pending_mshr_size_;
    } break;

    default:
      std::abort();
    }

    // pop processed request; release per-set Replay tracking so a deferred
    // fill into the same set can proceed.
    if (bank_req.type == bank_req_t::Replay) {
      uint32_t set_id = params_.addr_set_id(bank_req.addr);
      assert(inflight_replays_per_set_.at(set_id) > 0);
      --inflight_replays_per_set_.at(set_id);
    }
    pipe_req_->pop();
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
  std::vector<uint32_t> inflight_replays_per_set_;
  uint32_t rand_ctr_;

  // Flush walk state.
  bool     flushing_;
  uint32_t flush_set_idx_;
  uint32_t flush_way_idx_;

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