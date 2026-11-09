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

#pragma once

#include <dram_timing.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace vortex {

// One DRAM channel as a reservation-based bank model: each request is
// scheduled once, by placing its precharge, activate and column commands at
// the first DRAM clocks every JEDEC constraint allows -- before or after
// commands reserved earlier -- and its read data departs tCL + tBL after its
// last column. Banks keep their rows open; a request to a bank waits for the
// bank's earlier requests (first come, first served per bank). The controller
// issues one command per clock; on HBM an ACT holds the row-command bus for
// two.
//
// Requests enter the controller's read or write buffer in order, one per
// request's bursts' worth of clocks, while the buffer has room; a write is
// complete once buffered. The controller serves reads, and turns to writes
// when its write bursts reach the high watermark or no read waits, back once
// they fall below the low one with a read waiting -- dram_sim's write drain. A
// batch of writes is placed after every column reserved before it, and the
// reads after it after the batch, so the data bus turns around once per batch.
// Reservations reach at most `horizon` clocks ahead: a request that would
// reach further waits.
//
// Refresh, when on, is an operation of its own: every `refi` clocks it takes
// the clock's turn to reserve. A per-bank refresh closes the next bank in
// turn -- a precharge once the bank's commands allow it, then REF -- and keeps
// it from activating for `rfc` clocks; an all-bank refresh does the same to
// every bank with one precharge and one REF. A refresh that would reach past
// the horizon waits, and the requests with it.
//
// Foldsim's memory endpoint (its shell's rtl/foldsim_mem_bank.sv) implements
// the same decisions; the normative copy of this model is the shell's
// model/dram_bank_model.h, and this one gives Vortex's simulators the same
// timing (rtlsim's AXI_MEM_MODEL=bank, against which ci/testcases/foldsim.yaml
// holds Foldsim cycle-exact). The faces below put it behind DramSim's
// interface and behind an AXI4 port.
class DramBankModel {
public:
  enum class Refresh { Off, PerBank, AllBank };

  struct Config {
    dram_timing::geometry_t geom;
    dram_timing::clocks_t   t;
    Refresh  refresh     = Refresh::Off;  // dram_sim models none
    uint32_t refi        = 0;    // clocks from one refresh operation to the next
    uint32_t rfc         = 0;    // clocks a refreshed bank may not activate
    uint32_t read_queue  = 32;   // reserved read bursts whose column has not issued
    uint32_t write_queue = 32;   // the same for writes
    // Write bursts buffered or reserved and not issued at which the channel
    // turns to writes, and below which it turns back to waiting reads:
    // dram_sim's 80 % and 20 % of its 32-entry write buffer.
    uint32_t wr_high     = 26;
    uint32_t wr_low      = 7;
    // Clocks a reservation may reach past the current one: foldsim's ring of
    // 512 clocks, less the word of 64 it is retiring.
    uint32_t horizon     = 448;
    uint32_t bursts      = 2;    // 32 B bursts per request

    static Config make(dram_timing::Standard standard, uint32_t data_rate, uint32_t block_size,
                       Refresh refresh = Refresh::Off) {
      Config c;
      c.geom = dram_timing::geometry(standard);
      c.t = dram_timing::clocks(standard, data_rate);
      c.bursts = (block_size > dram_timing::kBurstSize) ? block_size / dram_timing::kBurstSize : 1;
      if (c.bursts > (1u << c.geom.log2_visit)) {
        throw std::invalid_argument("dram_bank_model: a request spans more than one row visit");
      }
      c.set_refresh(refresh, dram_timing::refresh(standard, data_rate));
      return c;
    }

    // Per bank, the banks take turns within tREFI; all-bank, the channel
    // refreshes once per tREFI.
    void set_refresh(Refresh mode, const dram_timing::refresh_t& r) {
      refresh = mode;
      refi = (mode == Refresh::Off) ? 0 : (mode == Refresh::PerBank) ? r.REFI / this->num_banks() : r.REFI;
      rfc = (mode == Refresh::Off) ? 0 : (mode == Refresh::PerBank) ? r.RFCPB : r.RFC;
    }

    static Refresh parse_refresh(const std::string& name) {
      if (name.empty() || name == "off") return Refresh::Off;
      if (name == "bank") return Refresh::PerBank;
      if (name == "all") return Refresh::AllBank;
      throw std::invalid_argument("dram_bank_model: refresh must be off, bank or all (got '" + name + "')");
    }

    // Column-to-column spacing on one pseudo-channel, and within one bank
    // group, by the directions of the earlier and the later command.
    uint32_t rtw() const {
      int64_t v = int64_t(t.CL) + t.BL + geom.rtw_extra - t.CWL;
      return uint32_t(std::max<int64_t>(t.CCDS, v));
    }
    uint32_t wtr_s() const { return t.CWL + t.BL + t.WTRS; }
    uint32_t wtr_l() const { return t.CWL + t.BL + t.WTRL; }
    uint32_t sep_pc(bool from_w, bool to_w) const {
      if (from_w == to_w) return t.CCDS;
      return from_w ? wtr_s() : rtw();
    }
    uint32_t sep_bg(bool from_w, bool to_w) const {
      if (from_w == to_w) return t.CCDL;
      return from_w ? wtr_l() : 0;
    }
    // The furthest clock after a command its constraints on later commands
    // reach, which must lie within the horizon.
    uint32_t col_reach(bool w) const {
      uint32_t r = std::max(t.CCDS, t.CCDL);
      r = std::max(r, w ? std::max(wtr_s(), wtr_l()) : rtw());
      return r - 1;
    }
    uint32_t act_reach() const {
      uint32_t r = std::max({t.RRDS, t.RRDL, t.FAW}) - 1;
      return std::max<uint32_t>(r, geom.two_clock_act ? 1 : 0);
    }
    uint32_t num_banks() const { return 1u << geom.bank_bits(); }
    uint32_t pc_of(uint32_t bank) const { return bank & ((1u << geom.log2_pcs) - 1); }
    uint32_t bg_of(uint32_t bank) const {
      return (bank >> geom.log2_pcs) & ((1u << geom.log2_bgs) - 1);
    }
  };

  struct Stats {
    uint64_t reads = 0, writes = 0, hits = 0, misses = 0, conflicts = 0, refreshes = 0;
    uint64_t deferred = 0;  // clocks a request or refresh waited: it reached past the horizon
  };

  explicit DramBankModel(const Config& cfg) : cfg_(cfg) { this->reset(); }

  const Config& config() const { return cfg_; }
  const Stats& stats() const { return stats_; }

  void reset() {
    d_ = 0;
    ingress_.clear();
    ingress_ready_ = 0;
    uint32_t nb = cfg_.num_banks();
    open_.assign(nb, false);
    row_.assign(nb, 0);
    act_.assign(nb, kNever);
    col_ready_.assign(nb, 0);
    pre_ready_.assign(nb, 0);
    cmd_.clear();
    rowbus_.clear();
    cols_.clear();
    acts_.clear();
    colkind_.clear();
    dep_.clear();
    rq_ = 0;
    wq_ = 0;
    rbuf_.clear();
    wbuf_.clear();
    write_mode_ = false;
    frontier_ = 0;
    batch_floor_ = 0;
    batch_last_ = 0;
    read_floor_ = 0;
    ref_due_ = cfg_.refi;
    ref_bank_ = 0;
    stats_ = Stats();
  }

  // The bank and row of a byte address within the channel: a row visit of
  // four bursts, then pseudo-channel, bank group and bank from the low end,
  // the bank XORed with the row folded to its width (permutation interleave).
  void decode(uint64_t addr, uint32_t* bank, uint32_t* row) const {
    const auto& g = cfg_.geom;
    uint32_t bb = g.bank_bits();
    uint64_t tx = (addr & (dram_timing::kChannelSize - 1)) / dram_timing::kBurstSize;
    tx >>= g.log2_visit;
    uint64_t b = tx & ((1u << bb) - 1);
    tx >>= bb;
    uint64_t r = tx >> (g.log2_row_bursts - g.log2_visit);
    for (uint64_t x = r; x != 0; x >>= bb) {
      b ^= x & ((1u << bb) - 1);
    }
    *bank = uint32_t(b);
    *row = uint32_t(r);
  }

  // A request joins the ingress; `tag` comes back when it completes.
  void push(uint32_t tag, bool write, uint64_t addr) {
    Request r;
    r.tag = tag;
    r.write = write;
    this->decode(addr, &r.bank, &r.row);
    ingress_.push_back(r);
  }

  bool ingress_empty() const { return ingress_.empty(); }
  size_t ingress_size() const { return ingress_.size(); }
  uint64_t clock() const { return d_; }

  // One DRAM clock d: columns issued by d leave the controller queues, the
  // ingress head enters its buffer if it may, the head of the buffer the
  // controller serves is reserved if it can be (its commands no earlier than
  // d+1), and a read whose data departs at d+1 completes.
  template <typename ReadDone, typename WriteDone>
  void tick(ReadDone&& read_done, WriteDone&& write_done) {
    const int64_t d = int64_t(d_);
    for (auto it = colkind_.begin(); it != colkind_.end() && it->first <= d;) {
      (it->second ? wq_ : rq_) -= 1;
      it = colkind_.erase(it);
    }
    const uint32_t nb = cfg_.bursts;
    if (!ingress_.empty() && d >= ingress_ready_) {
      const Request& r = ingress_.front();
      if (r.write ? (wq_ + nb * (wbuf_.size() + 1) <= cfg_.write_queue)
                  : (rq_ + nb * (rbuf_.size() + 1) <= cfg_.read_queue)) {
        (r.write ? wbuf_ : rbuf_).push_back(r);
        if (r.write) {
          write_done(r.tag);
        }
        ingress_.pop_front();
        ingress_ready_ = d + nb;
      }
    }
    const uint64_t wn = wq_ + nb * wbuf_.size();
    const uint64_t rn = rq_ + nb * rbuf_.size();
    if (!write_mode_) {
      if (wn >= cfg_.wr_high || rn == 0) {
        write_mode_ = true;
        batch_floor_ = std::max(d + 1, frontier_ + 1);
      }
    } else if (wn < cfg_.wr_low && rn != 0) {
      write_mode_ = false;
      read_floor_ = batch_last_ + 1;
    }
    auto& buf = write_mode_ ? wbuf_ : rbuf_;
    if (cfg_.refresh != Refresh::Off && cfg_.refi != 0 && d + 1 >= ref_due_) {
      if (this->refresh(d + 1, d + int64_t(cfg_.horizon))) {
        ref_due_ += cfg_.refi;
        ++stats_.refreshes;
      } else {
        ++stats_.deferred;
      }
    } else if (!buf.empty()) {
      if (this->reserve(buf.front(), d + 1, d + int64_t(cfg_.horizon),
                        write_mode_ ? batch_floor_ : read_floor_)) {
        buf.pop_front();
      } else {
        ++stats_.deferred;
      }
    }
    auto it = dep_.find(d + 1);
    if (it != dep_.end()) {
      read_done(it->second);
      dep_.erase(it);
    }
    this->prune(d);
    ++d_;
  }

private:
  static constexpr int64_t kNever = INT64_MIN / 4;
  static constexpr int64_t kNone = INT64_MAX / 4;

  struct Request {
    uint32_t tag;
    bool     write;
    uint32_t bank;
    uint32_t row;
  };

  struct Col {
    uint32_t pc, pg;
    bool w;
  };

  struct Act {
    uint32_t pc, pg;
  };

  bool cmd_free(int64_t t) const { return !cmd_.count(t); }
  bool row_free(int64_t t) const { return !rowbus_.count(t); }

  // The first clock in [lb, end] a column of direction `w` to (pc, pg) may
  // take, or kNone. A clock inside a reserved column's forbidden span skips
  // to its end.
  int64_t place_col(int64_t lb, int64_t end, uint32_t pc, uint32_t pg, bool w) const {
    const int64_t reach = this->max_col_sep();
    int64_t t = lb;
    while (t <= end) {
      int64_t next = t;
      if (!this->cmd_free(t)) {
        next = t + 1;
      }
      for (auto it = cols_.lower_bound(t - reach); it != cols_.end() && it->first <= t + reach; ++it) {
        const int64_t c = it->first;
        const Col& col = it->second;
        if (col.pc != pc || c == t) {
          continue;
        }
        int64_t need_pc, need_bg, after_pc, after_bg;
        after_pc = cfg_.sep_pc(col.w, w);
        after_bg = cfg_.sep_bg(col.w, w);
        if (c < t) {
          need_pc = after_pc;
          need_bg = after_bg;
        } else {
          need_pc = cfg_.sep_pc(w, col.w);
          need_bg = cfg_.sep_bg(w, col.w);
        }
        const int64_t dist = (c < t) ? (t - c) : (c - t);
        if (dist < need_pc) {
          next = std::max(next, c + std::max<int64_t>(after_pc, 1));
        }
        if (col.pg == pg && dist < need_bg) {
          next = std::max(next, c + std::max<int64_t>(after_bg, 1));
        }
      }
      if (next == t) {
        return t;
      }
      t = next;
    }
    return kNone;
  }

  // True if an ACT to (pc, pg) may take clock t: the command and row buses
  // are free, tRRD holds against every ACT of the pseudo-channel (tRRD_L
  // within the bank group) on either side, and no tFAW window holding t
  // holds four other ACTs of the pseudo-channel.
  bool act_ok(int64_t t, uint32_t pc, uint32_t pg) const {
    if (!this->cmd_free(t) || !this->row_free(t)) return false;
    if (cfg_.geom.two_clock_act && !this->row_free(t + 1)) return false;
    const int64_t F = cfg_.t.FAW;
    const int64_t reach = std::max<int64_t>({F, cfg_.t.RRDS, cfg_.t.RRDL});
    std::vector<int64_t> near;
    for (auto it = acts_.lower_bound(t - reach); it != acts_.end() && it->first <= t + reach; ++it) {
      const int64_t a = it->first;
      const Act& act = it->second;
      if (act.pc != pc) continue;
      const int64_t dist = (a < t) ? (t - a) : (a - t);
      if (dist < int64_t(cfg_.t.RRDS)) return false;
      if (act.pg == pg && dist < int64_t(cfg_.t.RRDL)) return false;
      if (dist < F) near.push_back(a);
    }
    near.push_back(t);
    std::sort(near.begin(), near.end());
    for (size_t i = 0; i + 4 < near.size(); ++i) {
      if (near[i + 4] - near[i] < F) return false;
    }
    return true;
  }

  int64_t place_act(int64_t lb, int64_t end, uint32_t pc, uint32_t pg) const {
    for (int64_t t = lb; t <= end; ++t) {
      if (this->act_ok(t, pc, pg)) return t;
    }
    return kNone;
  }

  int64_t place_pre(int64_t lb, int64_t end) const {
    for (int64_t t = lb; t <= end; ++t) {
      if (this->cmd_free(t) && this->row_free(t)) return t;
    }
    return kNone;
  }

  int64_t max_col_sep() const {
    return std::max<int64_t>({cfg_.t.CCDS, cfg_.t.CCDL, cfg_.rtw(), cfg_.wtr_s(), cfg_.wtr_l()});
  }

  // Places the request's commands; false (and nothing changed) if its queue
  // is full or it does not fit within the horizon.
  bool reserve(const Request& r, int64_t e, int64_t end, int64_t floor) {
    const auto& T = cfg_.t;
    const uint32_t nb = cfg_.bursts;
    const uint32_t b = r.bank;
    const uint32_t pc = cfg_.pc_of(b);
    const uint32_t pg = pc * (1u << cfg_.geom.log2_bgs) + cfg_.bg_of(b);
    const bool hit = open_[b] && row_[b] == r.row;
    const bool conflict = open_[b] && !hit;
    int64_t pre = kNone, act = kNone;
    int64_t col_ready = col_ready_[b];
    int64_t pre_ready = pre_ready_[b];
    if (!hit) {
      int64_t act_lb = e;
      if (conflict) {
        pre = this->place_pre(std::max(e, pre_ready_[b]), end);
        if (pre == kNone) return false;
        act_lb = pre + T.RP;
      }
      act_lb = std::max(act_lb, act_[b] + int64_t(T.RC));
      act = this->place_act(act_lb, end, pc, pg);
      if (act == kNone) return false;
      col_ready = act + (r.write ? T.RCDWR : T.RCDRD);
      pre_ready = act + T.RAS;
    }
    int64_t col[8];
    for (uint32_t k = 0; k < nb; ++k) {
      int64_t lb = std::max({e + int64_t(k), col_ready, floor});
      if (k != 0) {
        lb = std::max(lb, col[k - 1] + int64_t(std::max<uint32_t>(T.CCDL, 1)));
      }
      col[k] = this->place_col(lb, end, pc, pg, r.write);
      if (col[k] == kNone) return false;
      col_ready = std::max(col_ready, col[k] + 1);
      pre_ready = std::max(pre_ready, r.write ? col[k] + T.CWL + T.BL + T.WR : col[k] + T.RTP);
    }
    const int64_t last_col = col[nb - 1];
    const int64_t departure = last_col + T.CL + T.BL;
    int64_t last = last_col + cfg_.col_reach(r.write);
    if (!r.write) last = std::max(last, departure);
    if (act != kNone) last = std::max(last, act + int64_t(cfg_.act_reach()));
    if (last > end) {
      return false;
    }

    // Commit.
    if (pre != kNone) {
      cmd_.insert(pre);
      rowbus_.insert(pre);
    }
    if (act != kNone) {
      cmd_.insert(act);
      rowbus_.insert(act);
      if (cfg_.geom.two_clock_act) rowbus_.insert(act + 1);
      acts_[act] = Act{pc, pg};
      act_[b] = act;
      open_[b] = true;
      row_[b] = r.row;
    }
    for (uint32_t k = 0; k < nb; ++k) {
      cmd_.insert(col[k]);
      cols_[col[k]] = Col{pc, pg, r.write};
      colkind_[col[k]] = r.write;
      frontier_ = std::max(frontier_, col[k]);
      if (r.write) batch_last_ = std::max(batch_last_, col[k]);
    }
    (r.write ? wq_ : rq_) += nb;
    col_ready_[b] = col_ready;
    pre_ready_[b] = pre_ready;
    if (!r.write) {
      dep_[departure] = r.tag;
    }
    if (r.write) ++stats_.writes; else ++stats_.reads;
    if (hit) ++stats_.hits; else if (conflict) ++stats_.conflicts; else ++stats_.misses;
    return true;
  }

  // A refresh operation: the bank in turn (all banks), precharged if open --
  // after every command reserved to it allows -- then REF, once the precharge
  // and the last activate allow; until rfc after it the bank may not
  // activate. Both commands take the row-command bus as a precharge does.
  // False (and nothing changed) if they do not fit within the horizon.
  bool refresh(int64_t e, int64_t end) {
    const auto& T = cfg_.t;
    const uint32_t nb = cfg_.num_banks();
    const bool all = (cfg_.refresh == Refresh::AllBank);
    bool open = false;
    int64_t pre_lb = e, act_lb = e;
    for (uint32_t b = 0; b < nb; ++b) {
      if (!all && b != ref_bank_) continue;
      if (open_[b]) {
        open = true;
        pre_lb = std::max(pre_lb, pre_ready_[b]);
      }
      act_lb = std::max(act_lb, act_[b] + int64_t(T.RC));
    }
    int64_t pre = kNone, lb = act_lb;
    if (open) {
      pre = this->place_pre(pre_lb, end);
      if (pre == kNone) return false;
      lb = std::max(lb, pre + int64_t(T.RP));
    }
    const int64_t ref = this->place_pre(lb, end);
    if (ref == kNone) return false;

    // Commit.
    if (pre != kNone) {
      cmd_.insert(pre);
      rowbus_.insert(pre);
    }
    cmd_.insert(ref);
    rowbus_.insert(ref);
    for (uint32_t b = 0; b < nb; ++b) {
      if (!all && b != ref_bank_) continue;
      open_[b] = false;
      act_[b] = ref + int64_t(cfg_.rfc) - int64_t(T.RC);
    }
    if (!all) ref_bank_ = (ref_bank_ + 1) % nb;
    return true;
  }

  // Drops what can no longer constrain a command at d+1 or later.
  void prune(int64_t d) {
    const int64_t keep_cols = d + 1 - this->max_col_sep();
    while (!cols_.empty() && cols_.begin()->first < keep_cols) cols_.erase(cols_.begin());
    const int64_t keep_acts = d + 1 - std::max<int64_t>({cfg_.t.FAW, cfg_.t.RRDS, cfg_.t.RRDL});
    while (!acts_.empty() && acts_.begin()->first < keep_acts) acts_.erase(acts_.begin());
    while (!cmd_.empty() && *cmd_.begin() <= d) cmd_.erase(cmd_.begin());
    while (!rowbus_.empty() && *rowbus_.begin() <= d) rowbus_.erase(rowbus_.begin());
  }

  Config cfg_;
  uint64_t d_;
  std::deque<Request> ingress_;
  int64_t ingress_ready_;
  std::vector<bool> open_;
  std::vector<uint32_t> row_;
  std::vector<int64_t> act_, col_ready_, pre_ready_;
  std::set<int64_t> cmd_, rowbus_;
  std::map<int64_t, Col> cols_;
  std::map<int64_t, Act> acts_;
  std::map<int64_t, bool> colkind_;
  std::map<int64_t, uint32_t> dep_;
  uint32_t rq_, wq_;
  std::deque<Request> rbuf_, wbuf_;
  bool write_mode_;
  int64_t frontier_, batch_floor_, batch_last_, read_floor_;
  int64_t ref_due_;
  uint32_t ref_bank_;
  Stats stats_;
};

///////////////////////////////////////////////////////////////////////////////

// The bank model behind DramSim's interface (sim/common/dram_sim.h): one
// channel per memory bank, the same core-clock credit loop, one request
// reserved per bursts' worth of DRAM clocks per channel, the fixed latency
// after a read's data departs, and responses in completion order per channel.
class DramBankSim {
public:
  typedef bool (*ResponseCallback)(void* arg);

  struct Config {
    dram_timing::Standard type;
    uint32_t num_channels;
    uint32_t block_size;
    uint32_t core_clock;   // MHz
    uint32_t data_rate;    // MT/s
    uint32_t latency;      // core cycles added after a read's data departs
    uint32_t queue_size;   // requests in flight per channel; 0 = unbounded
    DramBankModel::Refresh refresh = DramBankModel::Refresh::Off;
  };

  explicit DramBankSim(const Config& cfg) : DramBankSim(cfg, DramBankModel::Config()) {}

  // `policy` gives the controller's queues, horizon and write policy.
  DramBankSim(const Config& cfg, const DramBankModel::Config& policy)
    : cfg_(cfg)
    , model_cfg_(DramBankModel::Config::make(cfg.type, cfg.data_rate, cfg.block_size, cfg.refresh)) {
    model_cfg_.read_queue  = policy.read_queue;
    model_cfg_.write_queue = policy.write_queue;
    model_cfg_.wr_high     = policy.wr_high;
    model_cfg_.wr_low      = policy.wr_low;
    model_cfg_.horizon     = policy.horizon;
    for (uint32_t i = 0; i < cfg.num_channels; ++i) {
      channels_.emplace_back(model_cfg_);
    }
    this->reset();
  }

  void reset() {
    cycle_ = 0;
    credit_ = 0;
    for (auto& ch : channels_) {
      ch.model.reset();
      ch.read_rsps.clear();
      ch.write_rsps.clear();
      ch.inflight = 0;
    }
    requests_.clear();
    free_.clear();
  }

  void tick() {
    ++cycle_;
    credit_ += cfg_.data_rate;
    for (auto& ch : channels_) {
      this->drain(ch.read_rsps, ch);
      this->drain(ch.write_rsps, ch);
    }
    while (credit_ >= 2ull * cfg_.core_clock) {
      for (auto& ch : channels_) {
        ch.model.tick(
          [&](uint32_t tag) { ch.read_rsps.push_back({tag, cycle_ + 1 + cfg_.latency}); },
          [&](uint32_t tag) { ch.write_rsps.push_back({tag, cycle_}); });
      }
      credit_ -= 2ull * cfg_.core_clock;
    }
  }

  bool can_accept(uint32_t channel) const {
    return (cfg_.queue_size == 0) || (channels_.at(channel).inflight < cfg_.queue_size);
  }

  void send_request(uint32_t channel, uint64_t addr, bool is_write, ResponseCallback cb, void* arg) {
    uint32_t tag;
    if (!free_.empty()) {
      tag = free_.back();
      free_.pop_back();
      requests_[tag] = {cb, arg};
    } else {
      tag = uint32_t(requests_.size());
      requests_.push_back({cb, arg});
    }
    auto& ch = channels_.at(channel);
    ch.model.push(tag, is_write, addr & ~uint64_t(cfg_.block_size - 1));
    ++ch.inflight;
  }

  const DramBankModel& channel(uint32_t i) const { return channels_.at(i).model; }

private:
  struct Rsp {
    uint32_t tag;
    uint64_t ready;
  };

  struct Channel {
    explicit Channel(const DramBankModel::Config& c) : model(c) {}
    DramBankModel model;
    std::deque<Rsp> read_rsps, write_rsps;
    uint32_t inflight = 0;
  };

  struct Req {
    ResponseCallback cb;
    void* arg;
  };

  void drain(std::deque<Rsp>& rsps, Channel& ch) {
    while (!rsps.empty() && rsps.front().ready <= cycle_) {
      Req& req = requests_[rsps.front().tag];
      if (req.cb && !req.cb(req.arg)) {
        break;
      }
      free_.push_back(rsps.front().tag);
      --ch.inflight;
      rsps.pop_front();
    }
  }

  Config cfg_;
  DramBankModel::Config model_cfg_;
  std::vector<Channel> channels_;
  std::vector<Req> requests_;
  std::vector<uint32_t> free_;
  uint64_t cycle_;
  uint64_t credit_;
};

///////////////////////////////////////////////////////////////////////////////

// The bank model as one AXI4 memory port, in AxiLatencyPipe's style
// (sim/common/axi_latency_pipe.h): every output is a function of registered
// state, the caller samples the request side before the clock edge and calls
// step() after it. Single-beat transactions; a burst sets the error, and from
// the next cycle on the port is ready for nothing.
//
// A read takes the lowest free read slot from acceptance until R takes it; it
// reaches the channel in the cycle after it was accepted, and is ready to
// drive R LATENCY + 1 cycles after the cycle whose DRAM clocks its data
// departed in. An AW and its W, in either order, form a write, which reaches
// the channel in the same way and is acknowledged on B from the cycle after
// the controller buffered it. Requests reach the channel in the order they
// formed, a write before a read of the same cycle.
// R and B take the oldest ready response. A request counts against `queue`
// from acceptance until it is ready.
class DramBankAxi {
public:
  struct Config {
    DramBankModel::Config model;
    uint32_t core_clock = 400;   // MHz
    uint32_t data_rate  = 800;   // MT/s
    uint32_t latency    = 0;     // core cycles
    uint32_t queue      = 0;     // requests in flight; 0 = only the slots bound them
    uint32_t read_slots = 32;
    uint32_t write_slots = 16;   // AWs, and Ws, accepted and not acknowledged on B
  };

  struct Inputs {
    bool     ar_valid = false;
    uint32_t ar_id    = 0;
    uint64_t ar_addr  = 0;
    uint32_t ar_len   = 0;
    bool     aw_valid = false;
    uint32_t aw_id    = 0;
    uint64_t aw_addr  = 0;
    uint32_t aw_len   = 0;
    bool     w_valid  = false;
    bool     w_last   = true;
    bool     r_ready  = false;
    bool     b_ready  = false;
  };

  explicit DramBankAxi(const Config& cfg) : cfg_(cfg), model_(cfg.model) { this->reset(); }

  const DramBankModel& model() const { return model_; }

  void reset() {
    model_.reset();
    cycle_ = 0;
    credit_ = 0;
    error_ = false;
    inflight_ = 0;
    slot_busy_.assign(cfg_.read_slots, false);
    slot_id_.assign(cfg_.read_slots, 0);
    slots_used_ = 0;
    pending_aw_ = 0;
    pending_w_ = 0;
    aw_queue_.clear();
    ready_reads_.clear();
    uncounted_ = 0;
    acks_.clear();
    cur_r_valid_ = false;
    cur_b_valid_ = false;
    full_cycles_ = 0;
  }

  bool ar_ready() const { return !error_ && this->room() && slots_used_ < cfg_.read_slots; }
  bool aw_ready() const { return !error_ && this->room() && pending_aw_ < cfg_.write_slots; }
  bool w_ready()  const { return !error_ && pending_w_ < cfg_.write_slots; }

  bool     r_valid() const { return cur_r_valid_; }
  uint32_t r_id()    const { return cur_r_id_; }
  uint32_t r_slot()  const { return cur_r_slot_; }
  bool     b_valid() const { return cur_b_valid_; }
  uint32_t b_id()    const { return cur_b_id_; }
  bool     error()   const { return error_; }
  // Target cycles the port could take no read: no slot, or no room.
  uint64_t full_cycles() const { return full_cycles_; }
  // The slot a read accepted this cycle takes: the lowest free once R has
  // taken its read, if it does.
  uint32_t ar_slot(bool r_ready) const {
    for (uint32_t s = 0; s < cfg_.read_slots; ++s) {
      if (!slot_busy_[s] || (r_ready && cur_r_valid_ && s == cur_r_slot_)) return s;
    }
    return cfg_.read_slots;
  }
  // Nothing accepted is unanswered.
  bool idle() const {
    return slots_used_ == 0 && pending_aw_ == 0 && pending_w_ == 0;
  }

  void step(const Inputs& in) {
    const bool ar_hs = in.ar_valid && this->ar_ready();
    const bool aw_hs = in.aw_valid && this->aw_ready();
    const bool w_hs  = in.w_valid && this->w_ready();
    const bool burst = (ar_hs && in.ar_len != 0) || (aw_hs && in.aw_len != 0) || (w_hs && !in.w_last);
    const bool ar_fire = ar_hs && in.ar_len == 0;
    const bool aw_fire = aw_hs && in.aw_len == 0;
    const bool w_fire  = w_hs && in.w_last;
    const bool r_fire  = cur_r_valid_ && in.r_ready;
    const bool b_fire  = cur_b_valid_ && in.b_ready;
    if (!(this->room() && slots_used_ < cfg_.read_slots)) {
      ++full_cycles_;
    }

    // This cycle's DRAM clocks, on what was accepted before it.
    credit_ += cfg_.data_rate;
    while (credit_ >= 2ull * cfg_.core_clock) {
      model_.tick(
        [&](uint32_t slot) { ready_reads_.push_back({slot, cycle_ + 1 + cfg_.latency}); },
        [&](uint32_t id) { acks_.push_back(id); ++uncounted_acks_; });
      credit_ -= 2ull * cfg_.core_clock;
    }

    // Responses: the stage driving R and B, then what is ready next cycle.
    if (r_fire) {
      slot_busy_[cur_r_slot_] = false;
      --slots_used_;
      cur_r_valid_ = false;
    }
    if (b_fire) {
      cur_b_valid_ = false;
    }
    const uint64_t next = cycle_ + 1;
    while (uncounted_ < ready_reads_.size() && ready_reads_[uncounted_].ready <= next) {
      ++uncounted_;
      --inflight_;
    }
    inflight_ -= uncounted_acks_;
    uncounted_acks_ = 0;
    if (!cur_r_valid_ && uncounted_ != 0) {
      cur_r_valid_ = true;
      cur_r_slot_ = ready_reads_.front().slot;
      cur_r_id_ = slot_id_[cur_r_slot_];
      ready_reads_.pop_front();
      --uncounted_;
    }
    if (!cur_b_valid_ && !acks_.empty()) {
      cur_b_valid_ = true;
      cur_b_id_ = acks_.front();
      acks_.pop_front();
    }

    // Requests: writes form from an AW and its W, then join the channel ahead
    // of this cycle's read.
    const bool new_write = (pending_w_ > pending_aw_ && aw_fire)
                        || (pending_w_ < pending_aw_ && w_fire)
                        || (aw_fire && w_fire);
    if (aw_fire) {
      aw_queue_.push_back({in.aw_id, in.aw_addr});
    }
    if (new_write) {
      auto aw = aw_queue_.front();
      aw_queue_.pop_front();
      model_.push(aw.id, true, aw.addr);
    }
    if (ar_fire) {
      uint32_t s = 0;
      while (slot_busy_[s]) ++s;
      slot_busy_[s] = true;
      slot_id_[s] = in.ar_id;
      ++slots_used_;
      model_.push(s, false, in.ar_addr);
    }
    inflight_ += uint32_t(ar_fire) + uint32_t(aw_fire);
    pending_aw_ = pending_aw_ + uint32_t(aw_fire) - uint32_t(b_fire && pending_aw_ != 0);
    pending_w_  = pending_w_ + uint32_t(w_fire) - uint32_t(b_fire && pending_w_ != 0);
    if (burst) {
      error_ = true;
    }
    ++cycle_;
  }

private:
  struct Ready {
    uint32_t slot;
    uint64_t ready;
  };

  struct Aw {
    uint32_t id;
    uint64_t addr;
  };

  bool room() const { return cfg_.queue == 0 || inflight_ < cfg_.queue; }

  Config cfg_;
  DramBankModel model_;
  uint64_t cycle_;
  uint64_t credit_;
  bool error_;
  uint32_t inflight_;
  std::vector<bool> slot_busy_;
  std::vector<uint32_t> slot_id_;
  uint32_t slots_used_;
  uint32_t pending_aw_, pending_w_;
  std::deque<Aw> aw_queue_;
  std::deque<Ready> ready_reads_;  // reads departed; the first `uncounted_` are ready next cycle
  size_t uncounted_;
  std::deque<uint32_t> acks_;
  uint32_t uncounted_acks_ = 0;
  bool cur_r_valid_, cur_b_valid_;
  uint32_t cur_r_id_ = 0, cur_r_slot_ = 0, cur_b_id_ = 0;
  uint64_t full_cycles_ = 0;
};

} // namespace vortex
