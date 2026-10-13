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

// foldsim's memory endpoint (hw/rtl/afu/foldsim/foldsim_mem.sv), driven as the
// simulator and the control endpoint drive it -- one token per channel per
// target cycle -- by a model of an AXI4 master, against a model of HBM that
// checks the AXI4 protocol on the device side. The properties under test:
//   * read data, write acknowledgements and ids are right, and no response
//     comes before its latency;
//   * a reset with HBM transactions in flight finishes only once HBM has
//     answered them all, takes no request, and delivers nothing from before
//     it to the target after it;
//   * a burst sets `error` and reaches HBM in no part, and a reset clears it.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "Vfoldsim_mem.h"
#include "verilated.h"

namespace {

constexpr int kWords = 16;  // DATA_W = 512
constexpr uint32_t kReadLatency = 10;
constexpr uint32_t kWriteLatency = 12;
constexpr uint64_t kBase = 0x10000000ull;
constexpr uint64_t kLines = 64;              // target addresses in use
constexpr uint64_t kMagic = 0x7f000ull;      // offered only in reset cycles
constexpr uint64_t kBurstAddr = 0x7e000ull;  // offered only as a burst

using Line = std::array<uint32_t, kWords>;

Vfoldsim_mem* dut = nullptr;
std::mt19937 rng(7);
uint64_t host_cycle = 0;
uint64_t tcycle = 0;
int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      if (failures++ < 20) {                                                   \
        printf("FAIL host %lu target %lu: ", (unsigned long)host_cycle,        \
               (unsigned long)tcycle);                                         \
        printf(__VA_ARGS__);                                                   \
        printf("\n");                                                          \
      }                                                                        \
    }                                                                          \
  } while (0)

bool chance(double p) { return std::uniform_real_distribution<>(0, 1)(rng) < p; }
int rand_int(int lo, int hi) { return std::uniform_int_distribution<>(lo, hi)(rng); }

Line make_line(uint64_t addr, uint32_t version) {
  Line l;
  for (int i = 0; i < kWords; ++i)
    l[i] = uint32_t(addr * 2654435761u) ^ (version << 12) ^ (uint32_t(i) * 0x9e3779b9u);
  return l;
}

// ---- HBM: an AXI4 slave with random ready and latency ----
struct Hbm {
  std::map<uint64_t, Line> mem;
  double ready_p = 0.7;
  int lat_lo = 1, lat_hi = 40;
  struct Pending { uint64_t at; Line data; };
  std::deque<Pending> r;         // reads accepted, not returned
  std::deque<uint64_t> b;        // writes done, not acknowledged
  std::deque<uint64_t> aw;       // AWs waiting for their W
  std::deque<std::pair<Line, uint64_t>> w;  // Ws waiting for their AW
  std::set<uint64_t> ar_seen;
  uint64_t n_ar = 0, n_aw = 0, n_w = 0;
  bool arready = false, awready = false, wready = false;
  bool ar_held = false, aw_held = false, w_held = false;
  uint64_t ar_a = 0, aw_a = 0;
  Line w_d{};

  size_t outstanding() const { return r.size() + b.size() + aw.size() + w.size(); }

  uint64_t when(uint64_t last) {
    return std::max(last, host_cycle + uint64_t(rand_int(lat_lo, lat_hi)));
  }

  void drive() {
    arready = chance(ready_p);
    awready = chance(ready_p);
    wready = chance(ready_p);
    dut->m_arready = arready;
    dut->m_awready = awready;
    dut->m_wready = wready;
    dut->m_rvalid = !r.empty() && r.front().at <= host_cycle;
    for (int i = 0; i < kWords; ++i)
      dut->m_rdata[i] = r.empty() ? 0 : r.front().data[i];
    dut->m_bvalid = !b.empty() && b.front() <= host_cycle;
  }

  Line wdata() const {
    Line l;
    for (int i = 0; i < kWords; ++i)
      l[i] = dut->m_wdata[i];
    return l;
  }

  void sample() {
    if (ar_held)
      CHECK(dut->m_arvalid && dut->m_araddr == ar_a, "HBM: AR withdrawn or changed");
    if (aw_held)
      CHECK(dut->m_awvalid && dut->m_awaddr == aw_a, "HBM: AW withdrawn or changed");
    if (w_held)
      CHECK(dut->m_wvalid && wdata() == w_d, "HBM: W withdrawn or changed");
    if (dut->m_arvalid)
      CHECK(dut->m_arlen == 0, "HBM: AR burst");
    if (dut->m_awvalid)
      CHECK(dut->m_awlen == 0, "HBM: AW burst");
    if (dut->m_wvalid)
      CHECK(dut->m_wlast, "HBM: W without WLAST");
    CHECK(dut->m_rready && dut->m_bready, "HBM: R or B not ready");

    const bool ar_hs = dut->m_arvalid && arready;
    const bool aw_hs = dut->m_awvalid && awready;
    const bool w_hs = dut->m_wvalid && wready;
    ar_held = dut->m_arvalid && !arready;
    aw_held = dut->m_awvalid && !awready;
    w_held = dut->m_wvalid && !wready;
    ar_a = dut->m_araddr;
    aw_a = dut->m_awaddr;
    w_d = wdata();

    if (dut->m_rvalid)
      r.pop_front();
    if (dut->m_bvalid)
      b.pop_front();
    if (ar_hs) {
      ++n_ar;
      const uint64_t a = dut->m_araddr;
      ar_seen.insert(a);
      auto it = mem.find(a);
      r.push_back({when(r.empty() ? 0 : r.back().at), it != mem.end() ? it->second : Line{}});
    }
    if (aw_hs) {
      ++n_aw;
      aw.push_back(dut->m_awaddr);
    }
    if (w_hs) {
      ++n_w;
      w.push_back({wdata(), dut->m_wstrb});
    }
    while (!aw.empty() && !w.empty()) {
      Line& m = mem[aw.front()];
      const Line& d = w.front().first;
      const uint64_t strb = w.front().second;
      for (int i = 0; i < kWords; ++i)
        for (int k = 0; k < 4; ++k)
          if ((strb >> (4 * i + k)) & 1)
            m[i] = (m[i] & ~(0xffu << (8 * k))) | (d[i] & (0xffu << (8 * k)));
      aw.pop_front();
      w.pop_front();
      b.push_back(when(b.empty() ? 0 : b.back()));
    }
  }
};

Hbm hbm;

// ---- The target: an AXI4 master issuing single-beat reads and writes ----
struct Op { uint64_t addr; int id; Line data; };

struct Master {
  bool generate = false;
  bool reset = false;      // this target cycle is a reset cycle
  bool offer_magic = false;
  // A burst to offer: 0 none, 1 AR, 2 AW, 3 a W without WLAST.
  int burst = 0;

  // What each channel offers this cycle, held until it fires.
  bool ar_v = false, aw_v = false, w_v = false;
  Op ar{}, aw{}, w{};
  int ar_len = 0, aw_len = 0;
  bool w_last = true;

  std::deque<Op> ar_q, aw_q, w_q;  // queued, not yet offered
  struct ReadExp { Op op; bool known; uint64_t fired; };
  std::deque<ReadExp> reads;       // fired, awaiting R
  std::deque<Op> aw_only, w_only;  // fired halves not yet formed
  struct WriteExp { Op op; uint64_t formed; };
  std::deque<WriteExp> writes;     // formed, awaiting B

  std::map<uint64_t, Line> known;  // data of every acknowledged write
  std::set<uint64_t> unknown;      // writes a reset left undecided
  std::map<uint64_t, int> busy;    // reads or writes in flight per address
  uint32_t version = 1;
  int next_id = 0;
  uint64_t n_r = 0, n_b = 0, stale = 0;

  bool idle() const {
    return ar_q.empty() && aw_q.empty() && w_q.empty() && reads.empty() && aw_only.empty()
        && w_only.empty() && writes.empty() && !ar_v && !aw_v && !w_v;
  }

  void make_requests() {
    if (!generate || reset)
      return;
    if (ar_q.size() + reads.size() < 24 && chance(0.3)) {
      const uint64_t a = uint64_t(rand_int(0, kLines - 1)) * 64;
      if (busy[a] == 0 || (writes_to(a) == 0)) {
        ++busy[a];
        ar_q.push_back({a, next_id++ & 15, {}});
      }
    }
    if (aw_q.size() + writes.size() < 24 && chance(0.25)) {
      const uint64_t a = uint64_t(rand_int(0, kLines - 1)) * 64;
      if (busy[a] == 0) {
        ++busy[a];
        Op op{a, next_id++ & 15, make_line(a, version++)};
        aw_q.push_back(op);
        w_q.push_back(op);
        pending_writes.insert(a);
      }
    }
  }

  std::set<uint64_t> pending_writes;
  int writes_to(uint64_t a) const { return pending_writes.count(a) ? 1 : 0; }

  // The channels' tokens for this cycle.
  void offer() {
    if (reset) {
      ar_v = offer_magic;
      ar = {kMagic, 0, {}};
      ar_len = 0;
      aw_v = w_v = false;
      return;
    }
    if (burst == 1 && !ar_v) {
      ar_v = true;
      ar = {kBurstAddr, 1, {}};
      ar_len = 3;
    } else if (burst == 2 && !aw_v) {
      aw_v = true;
      aw = {kBurstAddr, 2, {}};
      aw_len = 1;
    } else if (burst == 3 && !w_v) {
      w_v = true;
      w = {kBurstAddr, 3, make_line(kBurstAddr, 0)};
      w_last = false;
    }
    if (!ar_v && !ar_q.empty() && chance(0.8)) {
      ar_v = true;
      ar = ar_q.front();
      ar_len = 0;
      ar_q.pop_front();
    }
    if (!aw_v && !aw_q.empty() && chance(0.6)) {
      aw_v = true;
      aw = aw_q.front();
      aw_len = 0;
      aw_q.pop_front();
    }
    if (!w_v && !w_q.empty() && chance(0.6)) {
      w_v = true;
      w = w_q.front();
      w_last = true;
      w_q.pop_front();
    }
  }

  void read_done(uint64_t a) {
    if (--busy[a] == 0)
      busy.erase(a);
  }

  // One target cycle, given the endpoint's tokens for it.
  void cycle(bool arready, bool awready, bool wready, bool rvalid, const Line& rdata, int rid,
             bool bvalid, int bid) {
    if (reset) {
      // The target is in reset: it ignores its inputs and loses its state.
      for (auto& r : reads)
        read_done(r.op.addr);
      for (auto* q : {&aw_only, &w_only})
        for (auto& op : *q)
          unknown.insert(op.addr);
      for (auto& wr : writes)
        unknown.insert(wr.op.addr);
      for (auto* q : {&ar_q})
        for (auto& op : *q)
          read_done(op.addr);
      for (auto& op : aw_q)
        unknown.insert(op.addr);
      for (auto& op : w_q)
        unknown.insert(op.addr);
      if (aw_v) unknown.insert(aw.addr);
      if (w_v) unknown.insert(w.addr);
      if (ar_v && ar.addr != kMagic) read_done(ar.addr);
      reads.clear();
      aw_only.clear();
      w_only.clear();
      writes.clear();
      ar_q.clear();
      aw_q.clear();
      w_q.clear();
      ar_v = aw_v = w_v = false;
      busy.clear();
      pending_writes.clear();
      return;
    }
    if (rvalid) {
      ++n_r;
      if (reads.empty()) {
        ++stale;
        CHECK(false, "R with no read outstanding");
      } else {
        const ReadExp e = reads.front();
        reads.pop_front();
        CHECK(rid == e.op.id, "R id %d, expected %d", rid, e.op.id);
        CHECK(tcycle >= e.fired + kReadLatency, "R at %lu for a read fired at %lu",
              (unsigned long)tcycle, (unsigned long)e.fired);
        if (e.known)
          CHECK(rdata == e.op.data, "R data wrong at 0x%lx", (unsigned long)e.op.addr);
        read_done(e.op.addr);
      }
    }
    if (bvalid) {
      ++n_b;
      if (writes.empty()) {
        ++stale;
        CHECK(false, "B with no write outstanding");
      } else {
        const WriteExp e = writes.front();
        writes.pop_front();
        CHECK(bid == e.op.id, "B id %d, expected %d", bid, e.op.id);
        CHECK(tcycle >= e.formed + kWriteLatency, "B at %lu for a write formed at %lu",
              (unsigned long)tcycle, (unsigned long)e.formed);
        known[e.op.addr] = e.op.data;
        unknown.erase(e.op.addr);
        pending_writes.erase(e.op.addr);
        read_done(e.op.addr);
      }
    }
    if (ar_v && arready) {
      ar_v = false;
      if (ar_len == 0) {
        const auto it = known.find(ar.addr);
        const bool k = !unknown.count(ar.addr);
        ReadExp e{ar, k, tcycle};
        e.op.data = (it != known.end()) ? it->second : Line{};
        reads.push_back(e);
      } else {
        burst = 0;
      }
    }
    const bool aw_fire = aw_v && awready;
    const bool w_fire = w_v && wready;
    if (aw_fire) {
      aw_v = false;
      if (aw_len == 0)
        aw_only.push_back(aw);
      else
        burst = 0;
    }
    if (w_fire) {
      w_v = false;
      if (w_last)
        w_only.push_back(w);
      else
        burst = 0;
    }
    while (!aw_only.empty() && !w_only.empty()) {
      writes.push_back({aw_only.front(), tcycle});
      aw_only.pop_front();
      w_only.pop_front();
    }
  }
};

Master master;

// ---- The token channels ----
enum { I_AWVALID, I_AWADDR, I_AWID, I_AWLEN, I_WVALID, I_WDATA, I_WSTRB, I_WLAST, I_ARVALID,
       I_ARADDR, I_ARID, I_ARLEN, I_RREADY, I_BREADY, I_TRESET, NUM_IN };
enum { O_AWREADY, O_WREADY, O_ARREADY, O_RVALID, O_RDATA, O_RLAST, O_RID, O_RRESP, O_BVALID,
       O_BRESP, O_BID, NUM_OUT };

bool taken[NUM_IN], got[NUM_OUT];
bool o_awready, o_wready, o_arready, o_rvalid, o_bvalid;
int o_rid, o_bid;
Line o_rdata;
// Set when a reset cycle's tokens are all exchanged: the endpoint's next
// output token says it has finished that cycle.
bool after_reset = false;
uint64_t reset_cycles_done = 0, drained_at_reset = 0;

void drive_tokens() {
  const Master& m = master;
#define IN(ch, name, value)                                                    \
  dut->name##_valid = !taken[ch];                                              \
  dut->name##_bits = (value)
  IN(I_AWVALID, awvalid, m.aw_v);
  IN(I_AWADDR, awaddr, m.aw.addr);
  IN(I_AWID, awid, m.aw.id);
  IN(I_AWLEN, awlen, m.aw_len);
  IN(I_WVALID, wvalid, m.w_v);
  dut->wdata_valid = !taken[I_WDATA];
  for (int i = 0; i < kWords; ++i)
    dut->wdata_bits[i] = m.w.data[i];
  IN(I_WSTRB, wstrb, ~0ull);
  IN(I_WLAST, wlast, m.w_last);
  IN(I_ARVALID, arvalid, m.ar_v);
  IN(I_ARADDR, araddr, m.ar.addr);
  IN(I_ARID, arid, m.ar.id);
  IN(I_ARLEN, arlen, m.ar_len);
  IN(I_RREADY, rready, 1);
  IN(I_BREADY, bready, 1);
  IN(I_TRESET, treset, m.reset);
#undef IN
#define OUT(ch, name) dut->name##_ready = !got[ch]
  OUT(O_AWREADY, awready); OUT(O_WREADY, wready); OUT(O_ARREADY, arready);
  OUT(O_RVALID, rvalid); OUT(O_RDATA, rdata); OUT(O_RLAST, rlast); OUT(O_RID, rid);
  OUT(O_RRESP, rresp); OUT(O_BVALID, bvalid); OUT(O_BRESP, bresp); OUT(O_BID, bid);
#undef OUT
}

void sample_tokens(bool in_take[NUM_IN], bool out_take[NUM_OUT]) {
#define IN(ch, name) in_take[ch] = dut->name##_valid && dut->name##_ready
  IN(I_AWVALID, awvalid); IN(I_AWADDR, awaddr); IN(I_AWID, awid); IN(I_AWLEN, awlen);
  IN(I_WVALID, wvalid); IN(I_WDATA, wdata); IN(I_WSTRB, wstrb); IN(I_WLAST, wlast);
  IN(I_ARVALID, arvalid); IN(I_ARADDR, araddr); IN(I_ARID, arid); IN(I_ARLEN, arlen);
  IN(I_RREADY, rready); IN(I_BREADY, bready); IN(I_TRESET, treset);
#undef IN
#define OUT(ch, name) out_take[ch] = dut->name##_valid && dut->name##_ready
  OUT(O_AWREADY, awready); OUT(O_WREADY, wready); OUT(O_ARREADY, arready);
  OUT(O_RVALID, rvalid); OUT(O_RDATA, rdata); OUT(O_RLAST, rlast); OUT(O_RID, rid);
  OUT(O_RRESP, rresp); OUT(O_BVALID, bvalid); OUT(O_BRESP, bresp); OUT(O_BID, bid);
#undef OUT
  if (out_take[O_AWREADY]) o_awready = dut->awready_bits;
  if (out_take[O_WREADY]) o_wready = dut->wready_bits;
  if (out_take[O_ARREADY]) o_arready = dut->arready_bits;
  if (out_take[O_RVALID]) o_rvalid = dut->rvalid_bits;
  if (out_take[O_RDATA])
    for (int i = 0; i < kWords; ++i)
      o_rdata[i] = dut->rdata_bits[i];
  if (out_take[O_RID]) o_rid = dut->rid_bits;
  if (out_take[O_BVALID]) o_bvalid = dut->bvalid_bits;
  if (out_take[O_BID]) o_bid = dut->bid_bits;
  if (out_take[O_RLAST]) CHECK(dut->rlast_bits, "RLAST low");
  if (out_take[O_RRESP]) CHECK(dut->rresp_bits == 0, "RRESP not OKAY");
  if (out_take[O_BRESP]) CHECK(dut->bresp_bits == 0, "BRESP not OKAY");
}

void host_step() {
  drive_tokens();
  hbm.drive();
  dut->clk = 0;
  dut->eval();
  bool in_take[NUM_IN], out_take[NUM_OUT];
  sample_tokens(in_take, out_take);
  bool any_out = false;
  for (int j = 0; j < NUM_OUT; ++j)
    any_out |= out_take[j];
  if (after_reset && any_out) {
    // The first token of the cycle after a reset cycle: that cycle has
    // finished, so HBM has answered everything and nothing is offered to it.
    after_reset = false;
    ++reset_cycles_done;
    CHECK(hbm.outstanding() == 0, "reset cycle finished with %zu HBM transactions open",
          hbm.outstanding());
    CHECK(!dut->m_arvalid && !dut->m_awvalid && !dut->m_wvalid,
          "reset cycle finished with requests offered to HBM");
  }
  hbm.sample();
  dut->clk = 1;
  dut->eval();
  ++host_cycle;
  for (int i = 0; i < NUM_IN; ++i)
    taken[i] |= in_take[i];
  for (int j = 0; j < NUM_OUT; ++j)
    got[j] |= out_take[j];
}

// Runs target cycles until `n` have completed.
void run(uint64_t n) {
  for (uint64_t k = 0; k < n; ++k) {
    master.make_requests();
    master.offer();
    if (master.reset && hbm.outstanding() != 0)
      ++drained_at_reset;
    for (int i = 0; i < NUM_IN; ++i)
      taken[i] = false;
    for (int j = 0; j < NUM_OUT; ++j)
      got[j] = false;
    uint64_t spins = 0;
    for (;;) {
      bool all = true;
      for (int i = 0; i < NUM_IN; ++i)
        all &= taken[i];
      for (int j = 0; j < NUM_OUT; ++j)
        all &= got[j];
      if (all)
        break;
      host_step();
      if (++spins > 200000) {
        printf("FAIL: target cycle %lu never completes\n", (unsigned long)tcycle);
        exit(1);
      }
    }
    if (master.reset)
      after_reset = true;
    master.cycle(o_arready, o_awready, o_wready, o_rvalid, o_rdata, o_rid, o_bvalid, o_bid);
    ++tcycle;
  }
}

void reset_cycles(int n, bool magic) {
  master.reset = true;
  master.offer_magic = magic;
  run(n);
  master.reset = false;
  master.offer_magic = false;
}

void drain() {
  master.generate = false;
  for (int i = 0; i < 100000 && !master.idle(); ++i)
    run(1);
  CHECK(master.idle(), "the target's transactions never complete");
  run(4);
}

void burst_case(int kind, const char* what) {
  drain();
  const uint64_t ar = hbm.n_ar, aw = hbm.n_aw, w = hbm.n_w;
  master.burst = kind;
  for (int i = 0; i < 64 && master.burst != 0; ++i)
    run(1);
  CHECK(master.burst == 0, "%s: never taken", what);
  run(2);
  CHECK(dut->error, "%s: no error", what);
  CHECK(!o_arready && !o_awready && !o_wready, "%s: still ready after the error", what);
  // Requests now wait; the endpoint stays stopped.
  master.generate = true;
  run(50);
  master.generate = false;
  CHECK(hbm.n_ar == ar && hbm.n_aw == aw && hbm.n_w == w, "%s: reached HBM", what);
  CHECK(dut->error, "%s: error cleared without a reset", what);
  reset_cycles(3, false);
  CHECK(!dut->error, "%s: reset left the error", what);
  master.generate = true;
  run(800);
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vfoldsim_mem();
  dut->read_latency = kReadLatency;
  dut->write_latency = kWriteLatency;
  dut->max_reads = 16;
  dut->max_writes = 16;
  dut->base = kBase;
  dut->reset = 1;
  for (int i = 0; i < 8; ++i) {
    hbm.drive();
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
  }
  dut->reset = 0;

  reset_cycles(4, true);
  master.generate = true;
  run(3000);

  // Resets with HBM slow and busy, at random points.
  for (int k = 0; k < 8; ++k) {
    hbm.lat_lo = 100;
    hbm.lat_hi = 400;
    hbm.ready_p = 0.3;
    run(rand_int(50, 300));
    hbm.lat_lo = 1;
    hbm.lat_hi = 40;
    hbm.ready_p = 0.7;
    reset_cycles(rand_int(1, 4), true);
    run(1500);
  }

  burst_case(1, "AR burst");
  burst_case(2, "AW burst");
  burst_case(3, "W without WLAST");

  drain();
  CHECK(hbm.n_aw == hbm.n_w, "HBM took %lu AWs and %lu Ws", (unsigned long)hbm.n_aw,
        (unsigned long)hbm.n_w);
  CHECK(hbm.outstanding() == 0, "HBM left with %zu transactions", hbm.outstanding());
  CHECK(!hbm.ar_seen.count(kBase + kMagic), "a request made in reset reached HBM");
  CHECK(!hbm.ar_seen.count(kBase + kBurstAddr), "a burst reached HBM");
  CHECK(dut->idle, "endpoint not idle at the end");
  CHECK(master.stale == 0, "%lu stale responses", (unsigned long)master.stale);
  CHECK(drained_at_reset >= 4, "only %lu resets found HBM busy", (unsigned long)drained_at_reset);

  printf("target cycles %lu, host cycles %lu; %lu reads, %lu writes acknowledged; "
         "%lu reset cycles, %lu of them with HBM busy\n",
         (unsigned long)tcycle, (unsigned long)host_cycle, (unsigned long)master.n_r,
         (unsigned long)master.n_b, (unsigned long)reset_cycles_done,
         (unsigned long)drained_at_reset);
  delete dut;
  if (failures != 0) {
    printf("FAILED: %d check(s)\n", failures);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
