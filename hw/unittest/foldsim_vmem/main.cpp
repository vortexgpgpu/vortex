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

// foldsim's virtual-memory controller (hw/rtl/afu/foldsim/foldsim_vmem.sv),
// driven as a folded simulator drives it -- an instance issues only when its
// read has been answered, and only as much as `free` allowed L host cycles
// earlier -- against a model of HBM that answers with random latency and
// ready, returns reads out of order and interleaved across ids, applies a
// write and samples a read at random points of their flight, and checks the
// AXI4 protocol. Under several traffic setups:
//   * every response equals a reference RAM's row, with requests applied in
//     order and a same-cycle read and write of one row by RDW_READ_FIRST;
//   * free never drops by more than the requests taken, never exceeds QUEUE;
//   * a reset with transactions in flight answers nothing from before it,
//     keeps free at 0 until HBM has answered everything, then gives QUEUE;
//     the writes taken before it reach HBM;
//   * after a final reset (which writes the host cache's dirty lines back),
//     HBM's image equals the reference RAM, padding untouched.
//
//   ./foldsim_vmem            every setup
//   ./foldsim_vmem <name>     one setup

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <random>
#include <set>
#include <type_traits>
#include <vector>

#include "Vfoldsim_vmem.h"
#include "verilated.h"

// The controller's parameters (the variant's Makefile passes the same to
// both sides).
#ifndef VM_IW
#define VM_IW 8
#endif
#ifndef VM_AW
#define VM_AW 10
#endif
#ifndef VM_DW
#define VM_DW 512
#endif
#ifndef VM_MW
#define VM_MW (VM_DW / 8)
#endif
#ifndef VM_RDW_READ_FIRST
#define VM_RDW_READ_FIRST 1
#endif
#ifndef VM_QUEUE
#define VM_QUEUE 32
#endif
#ifndef VM_OUTSTANDING
#define VM_OUTSTANDING 16
#endif
#ifndef VM_AXI_ID_W
#define VM_AXI_ID_W 4
#endif
#ifndef VM_INSTANCES
#define VM_INSTANCES (1u << VM_IW)
#endif
#ifndef VM_CACHE_ROWS
#define VM_CACHE_ROWS 0
#endif

namespace {

constexpr int kBytes = VM_DW / 8;
constexpr int kGroup = kBytes / VM_MW;
constexpr int kBeats = (VM_DW + 511) / 512;
constexpr int kRowLog2 = [] {
  int l = 6;
  while ((1 << l) < kBytes) ++l;
  return l;
}();
constexpr uint32_t kRows = 1u << VM_AW;
constexpr uint32_t kInstances = VM_INSTANCES;
constexpr uint64_t kBase = 0x123400000ull;
constexpr int kLead = 3;

using Row = std::vector<uint8_t>;
using Line = std::array<uint8_t, 64>;

int failures = 0;
const char* setup_name = "";
uint64_t now = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      if (failures++ < 20) {                                                   \
        printf("FAIL %s cycle %lu: ", setup_name, (unsigned long)now);         \
        printf(__VA_ARGS__);                                                   \
        printf("\n");                                                          \
      }                                                                        \
    }                                                                          \
  } while (0)

std::mt19937_64 rng(1);
bool chance(double p) { return std::uniform_real_distribution<>(0, 1)(rng) < p; }
uint64_t rand_in(uint64_t lo, uint64_t hi) {
  return std::uniform_int_distribution<uint64_t>(lo, hi)(rng);
}

// ---- Ports of any width: Verilator's integral types or VlWide ----
template <typename T>
void put_bytes(T& port, const uint8_t* b, int n) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t v = 0;
    for (int i = 0; i < n && i < 8; ++i) v |= uint64_t(b[i]) << (8 * i);
    port = T(v);
  } else {
    for (size_t w = 0; w < port.size(); ++w) {
      uint32_t v = 0;
      for (int k = 0; k < 4; ++k) {
        const size_t i = w * 4 + k;
        if (int(i) < n) v |= uint32_t(b[i]) << (8 * k);
      }
      port[w] = v;
    }
  }
}

template <typename T>
void get_bytes(const T& port, uint8_t* b, int n) {
  if constexpr (std::is_integral_v<T>) {
    const uint64_t v = uint64_t(port);
    for (int i = 0; i < n; ++i) b[i] = i < 8 ? uint8_t(v >> (8 * i)) : 0;
  } else {
    for (int i = 0; i < n; ++i) {
      const size_t w = size_t(i) / 4;
      b[i] = w < port.size() ? uint8_t(port[w] >> (8 * (i % 4))) : 0;
    }
  }
}

template <typename T>
bool get_bit(const T& port, int i) {
  if constexpr (std::is_integral_v<T>)
    return (uint64_t(port) >> i) & 1;
  else
    return (port[i / 32] >> (i % 32)) & 1;
}

template <typename T>
void put_bits(T& port, const std::vector<bool>& bits) {
  std::vector<uint8_t> b((bits.size() + 7) / 8, 0);
  for (size_t i = 0; i < bits.size(); ++i)
    if (bits[i]) b[i / 8] |= uint8_t(1u << (i % 8));
  put_bytes(port, b.data(), int(b.size()));
}

uint64_t row_addr(uint32_t inst, uint32_t row) {
  return kBase + ((uint64_t(row) * kInstances + inst) << kRowLog2);
}

uint8_t init_byte(uint64_t addr) {
  uint64_t x = addr * 0x9e3779b97f4a7c15ull;
  x ^= x >> 29;
  return uint8_t(x * 0xbf58476d1ce4e5b9ull >> 56);
}

// ---- HBM: an AXI4 slave ----
struct Hbm {
  std::map<uint64_t, Line> mem;
  double ready_p = 0.7, rvalid_p = 0.8, interleave_p = 0.3;
  uint64_t lat_lo = 20, lat_hi = 600;

  struct Read {
    uint64_t addr;
    uint64_t sample_at, ready_at;
    bool sampled = false;
    std::vector<Line> data;
    int beat = 0;
  };
  std::map<int, Read> reads;  // by id
  struct Aw { uint64_t addr; };
  std::deque<Aw> aw;
  struct Beat { std::array<uint8_t, 64> data; uint64_t strb; bool last; };
  std::deque<Beat> w;
  struct Write { uint64_t addr; std::vector<Beat> beats; uint64_t apply_at, b_at; bool applied = false; };
  std::deque<Write> writes;  // in order: B in order
  uint64_t last_apply = 0, last_b = 0;

  // Presented this cycle.
  bool arready = false, awready = false, wready = false;
  int r_id = -1;  // the R beat held, -1 none
  bool bvalid = false;
  // The DUT's request channels last cycle, for the hold rule.
  bool ar_held = false, aw_held = false, w_held = false;
  uint64_t ar_addr = 0, aw_addr = 0;
  int ar_id = 0;
  std::array<uint8_t, 64> w_data{};
  uint64_t w_strb = 0;
  bool w_last = false;

  uint64_t n_ar = 0, n_aw = 0, n_w = 0, n_b = 0, n_r = 0, max_lat = 0, min_lat = ~0ull;

  Line& line(uint64_t a) {
    auto it = mem.find(a);
    if (it == mem.end()) {
      Line l;
      for (int i = 0; i < 64; ++i) l[i] = init_byte(a + i);
      it = mem.emplace(a, l).first;
    }
    return it->second;
  }

  size_t outstanding() const { return reads.size() + aw.size() + w.size() + writes.size(); }

  void check_addr(uint64_t a, const char* ch) {
    CHECK((a & ((1ull << kRowLog2) - 1)) == 0, "%s address 0x%lx not row aligned", ch, (unsigned long)a);
    CHECK(a >= kBase && a < kBase + ((uint64_t(kRows) * kInstances) << kRowLog2),
          "%s address 0x%lx out of the image", ch, (unsigned long)a);
    CHECK((a >> 12) == ((a + uint64_t(kBeats) * 64 - 1) >> 12), "%s burst at 0x%lx crosses 4 KiB", ch,
          (unsigned long)a);
  }

  template <typename D>
  void drive(D* dut) {
    arready = chance(ready_p);
    awready = chance(ready_p);
    wready = chance(ready_p);
    dut->m_arready = arready;
    dut->m_awready = awready;
    dut->m_wready = wready;
    // R: hold a beat not taken; else maybe offer one.
    if (r_id < 0 && chance(rvalid_p)) {
      std::vector<int> ok, started;
      for (auto& kv : reads) {
        if (kv.second.sampled && kv.second.ready_at <= now) {
          ok.push_back(kv.first);
          if (kv.second.beat > 0) started.push_back(kv.first);
        }
      }
      // A burst under way usually continues; sometimes another id's beats
      // interleave.
      if (!started.empty() && !chance(interleave_p))
        r_id = started[rand_in(0, started.size() - 1)];
      else if (!ok.empty())
        r_id = ok[rand_in(0, ok.size() - 1)];
    }
    dut->m_rvalid = r_id >= 0;
    if (r_id >= 0) {
      const Read& r = reads.at(r_id);
      put_bytes(dut->m_rdata, r.data[r.beat].data(), 64);
      dut->m_rid = r_id;
      dut->m_rlast = r.beat == kBeats - 1;
    } else {
      dut->m_rid = rand_in(0, (1 << VM_AXI_ID_W) - 1);
      dut->m_rlast = chance(0.5);
    }
    bvalid = !writes.empty() && writes.front().b_at <= now && writes.front().applied;
    dut->m_bvalid = bvalid;
    dut->m_bid = 0;
  }

  // Before the edge: the DUT's outputs, and our handshakes.
  template <typename D>
  void sample(D* dut) {
    const bool arv = dut->m_arvalid, awv = dut->m_awvalid, wv = dut->m_wvalid;
    std::array<uint8_t, 64> wd;
    get_bytes(dut->m_wdata, wd.data(), 64);
    if (ar_held)
      CHECK(arv && dut->m_araddr == ar_addr && int(dut->m_arid) == ar_id, "AR withdrawn or changed");
    if (aw_held)
      CHECK(awv && dut->m_awaddr == aw_addr, "AW withdrawn or changed");
    if (w_held)
      CHECK(wv && wd == w_data && uint64_t(dut->m_wstrb) == w_strb && bool(dut->m_wlast) == w_last,
            "W withdrawn or changed");
    CHECK(dut->m_bready, "B not ready");
    if (arv) CHECK(dut->m_arlen == kBeats - 1, "AR length %d", int(dut->m_arlen));
    if (awv) CHECK(dut->m_awlen == kBeats - 1, "AW length %d", int(dut->m_awlen));
    if (awv) CHECK(dut->m_awid == 0, "AW id %d", int(dut->m_awid));

    const bool ar_hs = arv && arready, aw_hs = awv && awready, w_hs = wv && wready;
    ar_held = arv && !arready;
    aw_held = awv && !awready;
    w_held = wv && !wready;
    ar_addr = dut->m_araddr;
    ar_id = dut->m_arid;
    aw_addr = dut->m_awaddr;
    w_data = wd;
    w_strb = dut->m_wstrb;
    w_last = dut->m_wlast;

    if (r_id >= 0 && dut->m_rready) {
      Read& r = reads.at(r_id);
      ++n_r;
      if (++r.beat == kBeats)
        reads.erase(r_id);
      r_id = -1;
    }
    if (bvalid) {
      ++n_b;
      writes.pop_front();
    }
    if (ar_hs) {
      ++n_ar;
      const int id = dut->m_arid;
      CHECK(id < VM_OUTSTANDING, "AR id %d beyond OUTSTANDING", id);
      CHECK(!reads.count(id), "AR id %d reused while outstanding", id);
      check_addr(dut->m_araddr, "AR");
      Read r;
      r.addr = dut->m_araddr;
      r.ready_at = now + rand_in(lat_lo, lat_hi);
      r.sample_at = rand_in(now + 1, r.ready_at);
      const uint64_t lat = r.ready_at - now;
      max_lat = std::max(max_lat, lat);
      min_lat = std::min(min_lat, lat);
      reads[id] = r;
    }
    if (aw_hs) {
      ++n_aw;
      check_addr(dut->m_awaddr, "AW");
      aw.push_back({dut->m_awaddr});
    }
    if (w_hs) {
      ++n_w;
      w.push_back({wd, uint64_t(dut->m_wstrb), bool(dut->m_wlast)});
    }
    // A write is whole with its AW and its beats.
    while (!aw.empty() && int(w.size()) >= kBeats) {
      Write wr;
      wr.addr = aw.front().addr;
      for (int k = 0; k < kBeats; ++k) {
        const Beat& b = w.front();
        CHECK(b.last == (k == kBeats - 1), "WLAST %d on beat %d of %d", int(b.last), k, kBeats);
        for (int i = 0; i < 64; ++i)
          if (k * 64 + i >= kBytes)
            CHECK(!((b.strb >> i) & 1), "strobe on padding byte %d", k * 64 + i);
        wr.beats.push_back(b);
        w.pop_front();
      }
      aw.pop_front();
      last_apply = std::max(last_apply, now + rand_in(1, lat_hi / 2 + 1));
      wr.apply_at = last_apply;
      last_b = std::max(last_b, last_apply + rand_in(0, lat_hi / 2 + 1));
      wr.b_at = last_b;
      writes.push_back(wr);
    }
    if (!aw.empty() || !w.empty())
      CHECK(aw.size() < 64 && w.size() < 64 * kBeats, "AW and W out of step");
  }

  // After the edge: memory moves on. Writes land in order; a read samples
  // memory at a random point of its flight.
  void advance() {
    for (auto& wr : writes) {
      if (wr.applied) continue;
      if (wr.apply_at > now) break;
      for (int k = 0; k < kBeats; ++k) {
        Line& l = line(wr.addr + 64 * k);
        for (int i = 0; i < 64; ++i)
          if ((wr.beats[k].strb >> i) & 1) l[i] = wr.beats[k].data[i];
      }
      wr.applied = true;
    }
    for (auto& kv : reads) {
      Read& r = kv.second;
      if (r.sampled || r.sample_at > now) continue;
      for (int k = 0; k < kBeats; ++k) r.data.push_back(line(r.addr + 64 * k));
      r.sampled = true;
    }
  }
};

// ---- The reference RAM and the simulator's view ----
struct Ref {
  std::map<uint64_t, Row> rows;  // by row address
  Row get(uint64_t a) {
    auto it = rows.find(a);
    if (it != rows.end()) return it->second;
    Row r(kBytes);
    for (int i = 0; i < kBytes; ++i) r[i] = init_byte(a + i);
    return r;
  }
  void write(uint64_t a, const Row& d, const std::vector<bool>& mask) {
    Row r = get(a);
    for (int i = 0; i < kBytes; ++i)
      if (mask[i / kGroup]) r[i] = d[i];
    rows[a] = r;
  }
};

struct Setup {
  const char* name;
  uint64_t cycles;
  double p_issue, p_read, p_write, p_both, p_same, p_full;
  uint32_t active;   // instances in play at once (0: all)
  uint32_t hot_rows; // rows most requests take
  double p_hot;
  int lead;          // the simulator's lead; 0: it sees free the same cycle
  uint64_t lat_lo, lat_hi;
  double ready_p;
  int resets;        // resets with traffic in flight
};

const Setup kSetups[] = {
  // name        cycles  issue read  write both  same  full active hot  p_hot lead lat        ready resets
  {"hot",        300000,  0.9,  0.6,  0.6,  0.4,  0.5,  0.5, 8,     4,   0.8,  3,   20,  600,  0.7,  4},
  {"wide",       300000,  0.9,  0.7,  0.5,  0.3,  0.3,  0.5, 0,     0,   0.0,  3,   20,  200,  0.8,  4},
  {"full",       300000,  1.0,  0.7,  0.7,  0.5,  0.5,  0.5, 0,     8,   0.5,  0,   300, 600,  0.3,  4},
  {"pairs",      300000,  0.9,  0.2,  0.2,  0.9,  0.8,  0.4, 16,    2,   0.9,  3,   20,  600,  0.6,  4},
  {"partial",    300000,  0.9,  0.6,  0.6,  0.4,  0.5,  0.1, 8,     2,   0.9,  0,   20,  300,  0.7,  4},
  {"fastmem",    300000,  1.0,  0.6,  0.6,  0.4,  0.4,  0.5, 32,    4,   0.7,  0,   1,   20,   1.0,  4},
  {"sparse",     300000,  0.1,  0.6,  0.6,  0.4,  0.6,  0.5, 4,     2,   0.9,  3,   20,  600,  0.5,  4},
};

struct Tb {
  Vfoldsim_vmem* dut = nullptr;
  Hbm hbm;
  Ref ref;
  const Setup* s = nullptr;

  std::vector<bool> pending;
  std::vector<Row> expect;
  std::vector<uint64_t> issued_at;
  std::deque<uint32_t> free_hist;
  bool in_reset = false, draining = false;
  uint32_t last_free = 0;
  int last_taken = 0;

  // Writes taken and not yet acknowledged by HBM, by row address: without a
  // host cache the controller issues writes in request order, so B's come in
  // that order.
  std::deque<uint64_t> wr_order;
  std::map<uint64_t, int> wr_inflight;

  uint64_t n_reads = 0, n_writes = 0, n_checked = 0, n_pairs_same = 0, n_raw = 0, n_resets = 0,
           n_resets_busy = 0, max_rsp_lat = 0, full_cycles = 0, dropped = 0, n_quick = 0;

  void tick() {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
  }

  void reset_dut(int n) {
    in_reset = true;
    for (int i = 0; i < n; ++i) cycle(true);
    in_reset = false;
  }

  bool lead_ok(int n) {
    if (draining) return false;
    if (s->lead == 0) return int(last_free) >= n;
    if (free_hist.size() < size_t(kLead) + 1) return false;
    return free_hist.front() >= uint32_t(2 * (kLead + 1));
  }

  uint32_t pick_row() {
    if (s->hot_rows && chance(s->p_hot)) return uint32_t(rand_in(0, s->hot_rows - 1));
    return uint32_t(rand_in(0, kRows - 1));
  }

  // One host cycle. `rst`: the controller's reset is high.
  void cycle(bool rst = false, bool issue = true) {
    // Requests: an instance whose read has been answered.
    bool rd = false, wr = false;
    uint32_t inst = 0, rrow = 0, wrow = 0;
    Row wdata(kBytes);
    std::vector<bool> mask(VM_MW, true);
    if (!rst && issue && chance(s->p_issue)) {
      const uint32_t span = s->active ? s->active : kInstances;
      const uint32_t first = s->active ? uint32_t((now / 5000) * 7 % kInstances) : 0;
      for (int tries = 0; tries < 8; ++tries) {
        const uint32_t i = (first + uint32_t(rand_in(0, span - 1))) % kInstances;
        if (!pending[i]) {
          inst = i;
          if (chance(s->p_both)) {
            rd = wr = true;
          } else {
            rd = chance(s->p_read);
            wr = !rd && chance(s->p_write);
          }
          break;
        }
      }
      const int n = int(rd) + int(wr);
      if (n == 2 && !lead_ok(2) && lead_ok(1)) wr = false;
      if (!lead_ok(int(rd) + int(wr))) rd = wr = false;
      if (rd) rrow = pick_row();
      if (wr) {
        wrow = (rd && chance(s->p_same)) ? rrow : pick_row();
        for (auto& b : wdata) b = uint8_t(rand_in(0, 255));
        if (!chance(s->p_full)) {
          if (VM_MW == 1) {
            mask[0] = chance(0.8);
          } else {
            for (size_t k = 0; k < mask.size(); ++k) mask[k] = chance(0.5);
          }
        }
      }
    }

    dut->reset = rst;
    dut->rd_valid = rd;
    dut->rd_inst = rd ? inst : uint32_t(rand_in(0, kInstances - 1));
    dut->rd_row = rd ? rrow : uint32_t(rand_in(0, kRows - 1));
    dut->wr_valid = wr;
    dut->wr_inst = inst;
    dut->wr_row = wrow;
    put_bytes(dut->wr_data, wdata.data(), kBytes);
    put_bits(dut->wr_mask, mask);
    hbm.drive(dut);
    dut->clk = 0;
    dut->eval();

    // The DUT's outputs of this cycle.
    const uint32_t fr = dut->free;
    if (!rst && !in_reset_tail()) {
      CHECK(fr <= VM_QUEUE, "free %u above QUEUE", fr);
      if (!draining)
        CHECK(fr + uint32_t(last_taken) >= last_free, "free dropped from %u to %u with %d taken", last_free,
              fr, last_taken);
    }
    if (dut->rsp_valid) {
      const uint32_t ri = dut->rsp_inst;
      Row got(kBytes);
      get_bytes(dut->rsp_data, got.data(), kBytes);
      CHECK(!draining && !rst, "a response during reset or its drain (instance %u)", ri);
      if (ri >= kInstances || !pending[ri]) {
        CHECK(false, "a response for instance %u, which has no read", ri);
      } else {
        ++n_checked;
        CHECK(got == expect[ri], "instance %u: wrong row", ri);
        pending[ri] = false;
        max_rsp_lat = std::max(max_rsp_lat, now - issued_at[ri]);
        // Answered without memory: the cache or the write buffer.
        if (now - issued_at[ri] <= 6) ++n_quick;
      }
    }
    if (fr == 0 && !draining && !rst) ++full_cycles;

    // The reference, in request order.
    if (rd || wr) {
      const uint64_t ra = row_addr(inst, rrow), wa = row_addr(inst, wrow);
      if (rd && wr && rrow == wrow) ++n_pairs_same;
      const bool read_first = VM_RDW_READ_FIRST != 0;
      if (rd && (read_first || !wr)) expect[inst] = ref.get(ra);
      if (wr) ref.write(wa, wdata, mask);
      if (rd && !read_first && wr) expect[inst] = ref.get(ra);
      if (rd) {
        ++n_reads;
        if (VM_CACHE_ROWS == 0 && wr_inflight.count(ra)) ++n_raw;
        pending[inst] = true;
        issued_at[inst] = now;
      }
      if (wr) {
        ++n_writes;
        if (VM_CACHE_ROWS == 0) {
          wr_order.push_back(wa);
          ++wr_inflight[wa];
        }
      }
    }
    if (VM_CACHE_ROWS == 0 && hbm.bvalid) {
      const uint64_t a = wr_order.front();
      wr_order.pop_front();
      if (--wr_inflight[a] == 0) wr_inflight.erase(a);
    }

    hbm.sample(dut);
    tick();
    hbm.advance();

    // The drain: free 0 from the reset until HBM has answered everything.
    if (rst) {
      if (!draining) {
        ++n_resets;
        if (hbm.outstanding() != 0) ++n_resets_busy;
        for (uint32_t i = 0; i < kInstances; ++i)
          if (pending[i]) {
            pending[i] = false;
            ++dropped;
          }
      }
      draining = true;
    } else if (draining && dut->free != 0) {
      CHECK(dut->free == VM_QUEUE, "free %u after the drain", uint32_t(dut->free));
      CHECK(hbm.outstanding() == 0 && hbm.r_id < 0, "the drain ended with %zu transactions in HBM",
            hbm.outstanding());
      draining = false;
    }
    last_taken = int(rd) + int(wr);
    last_free = dut->free;
    free_hist.push_back(rst || draining ? 0 : last_free);
    if (free_hist.size() > size_t(kLead) + 1) free_hist.pop_front();
    ++now;
  }

  bool in_reset_tail() const { return draining; }

  void quiesce() {
    for (int i = 0; i < 200000; ++i) {
      bool any = false;
      for (bool p : pending) any |= p;
      if (!any && hbm.outstanding() == 0 && dut->free == VM_QUEUE && !draining) break;
      cycle(false, false);
    }
    for (int i = 0; i < 8; ++i) cycle(false, false);
    bool any = false;
    for (bool p : pending) any |= p;
    CHECK(!any, "reads never answered");
    CHECK(hbm.outstanding() == 0, "HBM left with %zu transactions", hbm.outstanding());
    CHECK(dut->free == VM_QUEUE, "free %u at rest", uint32_t(dut->free));
  }

  // HBM's image against the reference: rows, and the padding past them.
  void check_image() {
    uint64_t rows = 0;
    for (auto& kv : ref.rows) {
      ++rows;
      for (int k = 0; k < kBeats; ++k) {
        const Line& l = hbm.line(kv.first + 64 * k);
        for (int i = 0; i < 64; ++i) {
          const int j = k * 64 + i;
          const uint8_t want = j < kBytes ? kv.second[j] : init_byte(kv.first + j);
          if (l[i] != want) {
            CHECK(false, "HBM row 0x%lx byte %d: 0x%02x, want 0x%02x", (unsigned long)kv.first, j, l[i],
                  want);
            return;
          }
        }
      }
    }
    (void)rows;
  }

  int run(const Setup& setup, uint64_t seed) {
    s = &setup;
    setup_name = setup.name;
    rng.seed(seed);
    now = 0;
    const int before = failures;
    hbm = Hbm();
    hbm.lat_lo = setup.lat_lo;
    hbm.lat_hi = setup.lat_hi;
    hbm.ready_p = setup.ready_p;
    ref = Ref();
    pending.assign(kInstances, false);
    expect.assign(kInstances, Row(kBytes));
    issued_at.assign(kInstances, 0);
    free_hist.clear();
    wr_order.clear();
    wr_inflight.clear();
    draining = false;
    last_free = 0;
    last_taken = 0;
    n_reads = n_writes = n_checked = n_pairs_same = n_raw = n_resets = n_resets_busy = max_rsp_lat = 0;
    full_cycles = dropped = n_quick = 0;

    dut = new Vfoldsim_vmem();
    dut->base = kBase;
    dut->clk = 0;
    dut->eval();
    reset_dut(4);
    while (draining) cycle(false, false);

    const uint64_t slice = setup.cycles / uint64_t(setup.resets + 1);
    for (int k = 0; k <= setup.resets; ++k) {
      for (uint64_t c = 0; c < slice; ++c) cycle();
      if (k == setup.resets) break;
      // A reset while HBM is slow and busy.
      const auto lo = hbm.lat_lo, hi = hbm.lat_hi;
      const double rp = hbm.ready_p;
      hbm.lat_lo = 200;
      hbm.lat_hi = 600;
      hbm.ready_p = 0.3;
      for (int c = 0; c < 300; ++c) cycle();
      // Sparse traffic can leave HBM idle at the random point; reset once it is not.
      for (int c = 0; c < 5000 && hbm.outstanding() == 0; ++c) cycle();
      reset_dut(int(rand_in(1, 5)));
      uint64_t drain = 0;
      while (draining && drain++ < 100000) cycle(false, false);
      CHECK(!draining, "the drain never ends");
      hbm.lat_lo = lo;
      hbm.lat_hi = hi;
      hbm.ready_p = rp;
    }
    quiesce();
    reset_dut(2);
    while (draining) cycle(false, false);
    CHECK(hbm.outstanding() == 0, "HBM left with %zu transactions after the last drain", hbm.outstanding());
    check_image();

    printf("%-8s %lu cycles: %lu reads checked of %lu (%lu meeting a write in flight, %lu answered "
           "locally), %lu writes, "
           "%lu same-row pairs; HBM %lu AR %lu AW, latency %lu-%lu, response latency max %lu; "
           "%lu cycles with free 0; %lu resets (%lu with HBM busy, %lu reads dropped): %s\n",
           setup.name, (unsigned long)now, (unsigned long)n_checked, (unsigned long)n_reads,
           (unsigned long)n_raw, (unsigned long)n_quick, (unsigned long)n_writes,
           (unsigned long)n_pairs_same,
           (unsigned long)hbm.n_ar, (unsigned long)hbm.n_aw, (unsigned long)hbm.min_lat,
           (unsigned long)hbm.max_lat, (unsigned long)max_rsp_lat, (unsigned long)full_cycles,
           (unsigned long)n_resets, (unsigned long)n_resets_busy, (unsigned long)dropped,
           failures == before ? "ok" : "FAILED");
    CHECK(n_resets_busy >= uint64_t(setup.resets), "only %lu of the resets found HBM busy",
          (unsigned long)n_resets_busy);
    delete dut;
    dut = nullptr;
    return failures - before;
  }
};

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  const char* only = nullptr;
  for (int i = 1; i < argc; ++i)
    if (argv[i][0] != '+') only = argv[i];
  printf("foldsim_vmem: IW %d AW %d DW %d MW %d RDW_READ_FIRST %d QUEUE %d OUTSTANDING %d INSTANCES %u "
         "CACHE_ROWS %d\n",
         VM_IW, VM_AW, VM_DW, VM_MW, VM_RDW_READ_FIRST, VM_QUEUE, VM_OUTSTANDING, kInstances,
         VM_CACHE_ROWS);
  Tb tb;
  uint64_t seed = 1;
  for (const auto& s : kSetups) {
    ++seed;
    if (only && std::strcmp(only, s.name)) continue;
    tb.run(s, seed);
  }
  if (failures != 0) {
    printf("FAILED: %d check(s)\n", failures);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
