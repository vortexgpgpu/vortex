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

// foldsim's grouped virtual-memory controller (foldsim_vmem_group: N members
// of one class copy behind one foldsim_vmem), driven as a folded simulator
// drives it -- a copy's state pass issues a read and/or a write to any of the
// members at once for one instance, only once that instance has no read
// outstanding in any member, and only to a member whose `free` allowed it L =
// 4 host cycles earlier (the simulator's decision comes four host cycles
// before its requests: five decisions' entries may follow a `free` it saw) --
// against a model of HBM with random latency and
// ready, reads out of order and interleaved across ids, writes applied and
// reads sampled at random points of their flight, and the AXI4 protocol
// checked. The members differ in rows, data width (one not a multiple of 8)
// and mask; the bits of their ports above their own widths carry noise. Under
// several traffic setups:
//   * every response goes to the member that issued the read, for its
//     instance, one a cycle at most, and equals that member's reference RAM
//     row (requests applied in order, a same-cycle read and write of one row
//     by RDW_READ_FIRST);
//   * each member's free stays within 2 (FIFO - 1) and never drops by more
//     than twice the requests it took;
//   * a reset with requests queued and transactions in flight answers
//     nothing from before it, keeps every free at 0 until HBM has answered
//     everything, and the queued writes still reach HBM;
//   * a move with requests queued and in flight answers them from the old
//     region, with free 0 until the controller's move ends;
//   * after a final reset, HBM's image equals every member's reference RAM,
//     each in its own rows (GROW_m + row) of the controller's image, padding
//     past a member's bytes zero and past the controller's untouched.
// The z* setups clear the image and raise `zero` (known-zero rows), dropping
// and raising it at random; the m* setups move the region under traffic.
//
//   ./foldsim_vmem_group            every setup
//   ./foldsim_vmem_group <name>     one setup

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

#include "Vfoldsim_vmem_group_tb.h"
#include "verilated.h"

#ifndef VG_CFG
#define VG_CFG 0
#endif

namespace {

// The setups of foldsim_vmem_group_tb.sv (VG_CFG_<n>).
struct Member { int aw, dw, mw, row0; };
struct Cfg {
  int iw, instances, aw, dw, mw, rdwf, cache;
  std::vector<Member> m;
};
const Cfg kCfgs[] = {
  {3, 8, 7, 512, 64, 1, 0, {{6, 512, 64, 0}, {5, 64, 8, 64}, {4, 26, 1, 96}}},
  {3, 8, 7, 512, 64, 0, 64, {{6, 512, 64, 0}, {5, 64, 8, 64}, {4, 26, 1, 96}}},
  {2, 3, 6, 128, 16, 1, 32, {{5, 128, 16, 0}, {4, 40, 5, 32}}},
  {4, 16, 6, 64, 1, 0, 16, {{4, 64, 1, 0}, {4, 64, 1, 16}, {3, 16, 1, 32}, {3, 8, 1, 40}}},
  {1, 1, 8, 512, 64, 1, 64, {{6, 512, 64, 0}, {6, 512, 64, 64}, {6, 512, 64, 128}, {6, 512, 64, 192}}},
};
const Cfg& C = kCfgs[VG_CFG];
const int N = int(C.m.size());

constexpr int kAxiIdW = 4, kOutstanding = 16, kQueue = 32, kFifo = 8, kLead = 4;
constexpr uint32_t kFull = 2 * (kFifo - 1);
const int kBytes = C.dw / 8;
const int kBeats = (C.dw + 511) / 512;
const int kRowLog2 = [] {
  int l = 6;
  while ((1 << l) < kBytes) ++l;
  return l;
}();
const uint32_t kInstances = uint32_t(C.instances);
const uint64_t kImageBytes = (uint64_t(1u << C.aw) * kInstances) << kRowLog2;
constexpr uint64_t kBase = 0x123400000ull;
constexpr uint64_t kStride = 0x104000000ull;

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

// Port bits as bytes, at any width.
template <typename T>
void put_bits(T& port, const std::vector<bool>& bits) {
  std::vector<uint8_t> b((bits.size() + 7) / 8, 0);
  for (size_t i = 0; i < bits.size(); ++i)
    if (bits[i]) b[i / 8] |= uint8_t(1u << (i % 8));
  put_bytes(port, b.data(), int(b.size()));
}

uint64_t g_base = kBase;
std::set<uint64_t> g_zeroed;

bool zero_at(uint64_t a) {
  auto it = g_zeroed.upper_bound(a);
  if (it == g_zeroed.begin()) return false;
  --it;
  return a < *it + kImageBytes;
}

uint64_t row_addr(int m, uint32_t inst, uint32_t row) {
  return g_base + ((uint64_t(C.m[m].row0 + row) * kInstances + inst) << kRowLog2);
}

uint8_t init_byte(uint64_t addr) {
  uint64_t x = addr * 0x9e3779b97f4a7c15ull;
  x ^= x >> 29;
  return uint8_t(x * 0xbf58476d1ce4e5b9ull >> 56);
}

// A member's bytes of a row, and the bits of its last byte it has.
int mbytes(int m) { return (C.m[m].dw + 7) / 8; }
uint8_t last_mask(int m) { return C.m[m].dw % 8 ? uint8_t((1u << (C.m[m].dw % 8)) - 1) : 0xff; }

// ---- HBM: an AXI4 slave (as foldsim_vmem's test) ----
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
  std::map<int, Read> reads;
  std::deque<uint64_t> aw;
  struct Beat { std::array<uint8_t, 64> data; uint64_t strb; bool last; };
  std::deque<Beat> w;
  struct Write { uint64_t addr; std::vector<Beat> beats; uint64_t apply_at, b_at; bool applied = false; };
  std::deque<Write> writes;
  uint64_t last_apply = 0, last_b = 0;
  bool arready = false, awready = false, wready = false;
  int r_id = -1;
  bool bvalid = false;
  bool ar_held = false, aw_held = false, w_held = false;
  uint64_t ar_addr = 0, aw_addr = 0;
  int ar_id = 0;
  std::array<uint8_t, 64> w_data{};
  uint64_t w_strb = 0;
  bool w_last = false;
  uint64_t n_ar = 0, n_aw = 0;

  Line& line(uint64_t a) {
    auto it = mem.find(a);
    if (it == mem.end()) {
      Line l;
      for (int i = 0; i < 64; ++i) l[i] = zero_at(a) ? 0 : init_byte(a + i);
      it = mem.emplace(a, l).first;
    }
    return it->second;
  }
  size_t outstanding() const { return reads.size() + aw.size() + w.size() + writes.size(); }
  bool busy_ahead() const {
    for (auto& kv : reads)
      if (kv.second.ready_at > now + 2) return true;
    for (auto& wr : writes)
      if (wr.b_at > now + 2) return true;
    return false;
  }
  void check_addr(uint64_t a, const char* ch) {
    CHECK((a & ((1ull << kRowLog2) - 1)) == 0, "%s address 0x%lx not row aligned", ch, (unsigned long)a);
    CHECK(a >= g_base && a < g_base + kImageBytes, "%s address 0x%lx out of the image at 0x%lx", ch,
          (unsigned long)a, (unsigned long)g_base);
  }
  template <typename D>
  void drive(D* dut) {
    arready = chance(ready_p);
    awready = chance(ready_p);
    wready = chance(ready_p);
    dut->m_arready = arready;
    dut->m_awready = awready;
    dut->m_wready = wready;
    if (r_id < 0 && chance(rvalid_p)) {
      std::vector<int> ok, started;
      for (auto& kv : reads)
        if (kv.second.sampled && kv.second.ready_at <= now) {
          ok.push_back(kv.first);
          if (kv.second.beat > 0) started.push_back(kv.first);
        }
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
      dut->m_rid = rand_in(0, (1 << kAxiIdW) - 1);
      dut->m_rlast = chance(0.5);
    }
    bvalid = !writes.empty() && writes.front().b_at <= now && writes.front().applied;
    dut->m_bvalid = bvalid;
    dut->m_bid = 0;
  }
  template <typename D>
  void sample(D* dut) {
    const bool arv = dut->m_arvalid, awv = dut->m_awvalid, wv = dut->m_wvalid;
    std::array<uint8_t, 64> wd;
    get_bytes(dut->m_wdata, wd.data(), 64);
    if (ar_held) CHECK(arv && dut->m_araddr == ar_addr && int(dut->m_arid) == ar_id, "AR withdrawn or changed");
    if (aw_held) CHECK(awv && dut->m_awaddr == aw_addr, "AW withdrawn or changed");
    if (w_held)
      CHECK(wv && wd == w_data && uint64_t(dut->m_wstrb) == w_strb && bool(dut->m_wlast) == w_last,
            "W withdrawn or changed");
    CHECK(dut->m_bready, "B not ready");
    if (arv) CHECK(dut->m_arlen == kBeats - 1, "AR length %d", int(dut->m_arlen));
    if (awv) CHECK(dut->m_awlen == kBeats - 1 && dut->m_awid == 0, "AW length %d id %d", int(dut->m_awlen),
                   int(dut->m_awid));
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
      if (++r.beat == kBeats) reads.erase(r_id);
      r_id = -1;
    }
    if (bvalid) writes.pop_front();
    if (ar_hs) {
      ++n_ar;
      const int id = dut->m_arid;
      CHECK(id < kOutstanding && !reads.count(id), "AR id %d beyond OUTSTANDING or reused", id);
      check_addr(dut->m_araddr, "AR");
      Read r;
      r.addr = dut->m_araddr;
      r.ready_at = now + rand_in(lat_lo, lat_hi);
      r.sample_at = rand_in(now + 1, r.ready_at);
      reads[id] = r;
    }
    if (aw_hs) {
      ++n_aw;
      check_addr(dut->m_awaddr, "AW");
      aw.push_back(dut->m_awaddr);
    }
    if (w_hs) w.push_back({wd, uint64_t(dut->m_wstrb), bool(dut->m_wlast)});
    while (!aw.empty() && int(w.size()) >= kBeats) {
      Write wr;
      wr.addr = aw.front();
      for (int k = 0; k < kBeats; ++k) {
        const Beat& b = w.front();
        CHECK(b.last == (k == kBeats - 1), "WLAST %d on beat %d of %d", int(b.last), k, kBeats);
        for (int i = 0; i < 64; ++i)
          if (k * 64 + i >= kBytes) CHECK(!((b.strb >> i) & 1), "strobe on padding byte %d", k * 64 + i);
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
  }
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

// ---- The reference: the controller's image rows, by address ----
struct Ref {
  std::map<uint64_t, Row> rows;
  Row get(uint64_t a) {
    auto it = rows.find(a);
    if (it != rows.end()) return it->second;
    Row r(kBytes);
    for (int i = 0; i < kBytes; ++i) r[i] = zero_at(a) ? 0 : init_byte(a + i);
    return r;
  }
  // A member's write: its bytes as its mask says, its last byte's bits above
  // its width and the padding past its bytes zero (enabled).
  void write(int m, uint64_t a, const Row& d, const std::vector<bool>& mask) {
    Row r = get(a);
    const int nb = mbytes(m), group = C.m[m].mw > 1 ? nb / C.m[m].mw : nb;
    for (int i = 0; i < kBytes; ++i) {
      if (i >= nb)
        r[i] = 0;
      else if (mask[i / group])
        r[i] = i == nb - 1 ? uint8_t(d[i] & last_mask(m)) : d[i];
    }
    rows[a] = r;
  }
};

struct Setup {
  const char* name;
  uint64_t cycles;
  double p_issue, p_member, p_read, p_write, p_both, p_same, p_full;
  uint32_t hot_rows;
  double p_hot;
  int lead;
  uint64_t lat_lo, lat_hi;
  double ready_p;
  int resets, zero, moves;
};

const Setup kSetups[] = {
  // name     cycles  issue member read write both same full  hot p_hot lead lat       ready rst z mv
  {"hot",     200000, 0.9,  0.7,   0.6, 0.6,  0.4, 0.5, 0.5,  4,  0.8,  3,   20,  600, 0.7,  3, 0, 0},
  {"wide",    200000, 0.9,  0.5,   0.7, 0.5,  0.3, 0.3, 0.5,  0,  0.0,  3,   20,  200, 0.8,  3, 0, 0},
  {"full",    200000, 1.0,  1.0,   0.7, 0.7,  0.5, 0.5, 0.5,  8,  0.5,  0,   300, 600, 0.3,  3, 0, 0},
  {"pairs",   200000, 0.9,  0.8,   0.2, 0.2,  0.9, 0.8, 0.4,  2,  0.9,  3,   20,  600, 0.6,  3, 0, 0},
  {"fastmem", 200000, 1.0,  0.9,   0.6, 0.6,  0.4, 0.4, 0.5,  4,  0.7,  0,   1,   20,  1.0,  3, 0, 0},
  {"sparse",  200000, 0.1,  0.5,   0.6, 0.6,  0.4, 0.6, 0.5,  2,  0.9,  3,   20,  600, 0.5,  3, 0, 0},
  // Write bursts into a slow memory: every member, nearly only writes, a host cycle each, while the
  // controller is out of room for long: the FIFOs fill to the room the simulator's lead needs.
  {"wburst",  200000, 1.0,  1.0,   0.03, 1.0, 0.0, 0.0, 0.9,  0,  0.0,  3,   300, 600, 0.15, 3, 0, 0},
  {"zhot",    200000, 0.9,  0.7,   0.6, 0.6,  0.4, 0.5, 0.5,  4,  0.8,  3,   20,  600, 0.7,  3, 1, 0},
  {"zfast",   200000, 1.0,  0.9,   0.6, 0.6,  0.4, 0.4, 0.5,  4,  0.7,  0,   1,   20,  1.0,  3, 1, 0},
  {"mhot",    200000, 0.9,  0.7,   0.6, 0.6,  0.4, 0.5, 0.5,  4,  0.8,  3,   20,  600, 0.7,  1, 0, 5},
  {"mfull",   200000, 1.0,  1.0,   0.7, 0.7,  0.5, 0.5, 0.5,  8,  0.5,  0,   300, 600, 0.3,  1, 0, 5},
  {"mzfast",  200000, 1.0,  0.9,   0.6, 0.6,  0.4, 0.4, 0.5,  4,  0.7,  0,   1,   20,  1.0,  1, 1, 5},
};

struct Tb {
  Vfoldsim_vmem_group_tb* dut = nullptr;
  Hbm hbm;
  Ref ref;
  const Setup* s = nullptr;
  std::vector<std::vector<bool>> pending;   // [member][inst]
  std::vector<std::vector<Row>> expect;     // [member][inst]
  std::vector<std::deque<uint32_t>> free_hist;
  std::vector<uint32_t> last_free;
  std::vector<int> last_taken;
  bool draining = false, zero = false, zeroed = false, toggles = false, on = true;
  bool move_pulse = false, moving_tb = false, switch_at_drain = false;
  uint64_t base_in = kBase, move_to = kBase;
  uint64_t n_reads = 0, n_writes = 0, n_checked = 0, n_multi = 0, n_resets = 0, n_resets_busy = 0, n_moves = 0,
           n_moves_busy = 0, dropped = 0, drain_cycles = 0, move_cycles = 0, n_toggles = 0, lead_taken = 0,
           max_queue = 0, n_moves_queued = 0, n_resets_queued = 0;

  void tick() {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
  }

  bool any_pending(uint32_t i) const {
    for (int m = 0; m < N; ++m)
      if (pending[m][i]) return true;
    return false;
  }

  bool lead_ok(int m, int n) const {
    if (draining) return false;
    if (s->lead == 0) return on && int(last_free[m]) >= 2 * n;
    if (free_hist[m].size() < size_t(kLead) + 1) return false;
    return free_hist[m].front() >= 8u;
  }

  uint32_t pick_row(int m) {
    const uint32_t rows = 1u << C.m[m].aw;
    if (s->hot_rows && chance(s->p_hot)) return uint32_t(rand_in(0, std::min(s->hot_rows, rows) - 1));
    return uint32_t(rand_in(0, rows - 1));
  }

  // force: an instance every member reads and writes this cycle (room
  // allowing), as a state pass touching them all.
  void cycle(bool rst = false, bool issue = true, int force = -1) {
    std::vector<bool> rd(N, false), wr(N, false);
    std::vector<uint32_t> rrow(N, 0), wrow(N, 0);
    std::vector<Row> wdata(N);
    std::vector<std::vector<bool>> mask(N);
    uint32_t inst = 0;
    bool go = false;
    if (!rst && force >= 0) {
      inst = uint32_t(force);
      go = !any_pending(inst);
    } else if (!rst && issue && chance(s->p_issue)) {
      for (int tries = 0; tries < 8 && !go; ++tries) {
        inst = uint32_t(rand_in(0, kInstances - 1));
        go = !any_pending(inst);
      }
    }
    int members = 0;
    for (int m = 0; m < N; ++m) {
      wdata[m].assign(kBytes, 0);
      for (auto& b : wdata[m]) b = uint8_t(rand_in(0, 255));  // noise above the member's bytes too
      mask[m].assign(C.mw, false);
      for (size_t k = 0; k < mask[m].size(); ++k) mask[m][k] = chance(0.5);
      if (!go || (force < 0 && !chance(s->p_member))) continue;
      if (force >= 0 || chance(s->p_both)) {
        rd[m] = wr[m] = true;
      } else {
        rd[m] = chance(s->p_read);
        wr[m] = !rd[m] && chance(s->p_write);
      }
      if (!lead_ok(m, int(rd[m]) + int(wr[m]))) rd[m] = wr[m] = false;
      if (rd[m]) rrow[m] = pick_row(m);
      if (wr[m]) {
        wrow[m] = (rd[m] && chance(s->p_same)) ? rrow[m] : pick_row(m);
        // A whole-row enable is always on (the simulator drives it so).
        const int mw = C.m[m].mw;
        const bool full = chance(s->p_full);
        for (int k = 0; k < mw; ++k) mask[m][k] = mw == 1 || full || chance(0.5);
      }
      members += rd[m] || wr[m];
    }
    if (members > 1) ++n_multi;

    if (toggles && zeroed && !rst && !draining && !moving_tb && chance(2e-4)) {
      zero = !zero;
      ++n_toggles;
    }
    dut->reset = rst;
    dut->zero = zero;
    dut->move = move_pulse;
    move_pulse = false;
    dut->base = base_in;
    if (rst && moving_tb) {
      moving_tb = false;
      switch_at_drain = true;
    }
    for (int m = 0; m < N; ++m) {
      // Bits above a member's own (its rows, instance range, data and mask)
      // are noise the group must ignore; the simulator drives zeros there,
      // but nothing should depend on it.
      const uint32_t noise_row = uint32_t(rand_in(0, (1u << C.aw) - 1)) & ~((1u << C.m[m].aw) - 1);
      dut->rd_valid[m] = rd[m];
      dut->rd_inst[m] = rd[m] ? inst : uint32_t(rand_in(0, kInstances - 1));
      dut->rd_row[m] = (rd[m] ? rrow[m] : uint32_t(rand_in(0, (1u << C.m[m].aw) - 1))) | noise_row;
      dut->wr_valid[m] = wr[m];
      dut->wr_inst[m] = wr[m] ? inst : uint32_t(rand_in(0, kInstances - 1));
      dut->wr_row[m] = wrow[m] | noise_row;
      put_bytes(dut->wr_data[m], wdata[m].data(), kBytes);
      put_bits(dut->wr_mask[m], mask[m]);
    }
    hbm.drive(dut);
    dut->clk = 0;
    dut->eval();

    // Outputs: room, and the answers.
    int answers = 0;
    for (int m = 0; m < N; ++m) {
      const uint32_t fr = dut->free[m];
      if (!rst && !draining) {
        CHECK(fr <= kFull, "member %d: free %u above 2 (FIFO - 1)", m, fr);
        if (!moving_tb)
          CHECK(fr + 2 * uint32_t(last_taken[m]) >= last_free[m], "member %d: free dropped from %u to %u", m,
                last_free[m], fr);
      }
      if (!dut->rsp_valid[m]) continue;
      ++answers;
      const uint32_t ri = dut->rsp_inst;
      Row got(kBytes);
      get_bytes(dut->rsp_data, got.data(), kBytes);
      CHECK(!draining && !rst, "an answer during reset or its drain (member %d)", m);
      if (ri >= kInstances || !pending[m][ri]) {
        CHECK(false, "member %d: an answer for instance %u, which has no read", m, ri);
        continue;
      }
      ++n_checked;
      const int nb = mbytes(m);
      bool same = true;
      for (int i = 0; i < nb; ++i) {
        const uint8_t msk = i == nb - 1 ? last_mask(m) : 0xff;
        same &= (got[i] & msk) == (expect[m][ri][i] & msk);
      }
      CHECK(same, "member %d instance %u: wrong row", m, ri);
      pending[m][ri] = false;
    }
    CHECK(answers <= 1, "%d answers in one cycle", answers);

    // The reference, in request order within a member (members are apart).
    for (int m = 0; m < N; ++m) {
      if (!rd[m] && !wr[m]) continue;
      const uint64_t ra = row_addr(m, inst, rrow[m]), wa = row_addr(m, inst, wrow[m]);
      const bool rf = C.rdwf != 0;
      if (rd[m] && (rf || !wr[m])) expect[m][inst] = ref.get(ra);
      if (wr[m]) ref.write(m, wa, wdata[m], mask[m]);
      if (rd[m] && !rf && wr[m]) expect[m][inst] = ref.get(ra);
      if (moving_tb) ++lead_taken;
      if (rd[m]) {
        ++n_reads;
        pending[m][inst] = true;
      }
      if (wr[m]) ++n_writes;
    }

    hbm.sample(dut);
    tick();
    hbm.advance();

    // A reset: every free 0 until HBM has answered everything, then 2 FIFO.
    bool all_zero = true, all_full = true;
    for (int m = 0; m < N; ++m) {
      all_zero &= dut->free[m] == 0;
      all_full &= dut->free[m] == kFull;
    }
    if (rst) {
      if (!draining) {
        ++n_resets;
        if (hbm.outstanding() != 0) ++n_resets_busy;
        for (int m = 0; m < N; ++m)
          for (uint32_t i = 0; i < kInstances; ++i)
            if (pending[m][i]) {
              pending[m][i] = false;
              ++dropped;
            }
      }
      draining = true;
    } else if (draining && all_zero) {
      ++drain_cycles;
    } else if (draining) {
      CHECK(all_full, "free not 2 (FIFO - 1) for every member after the drain");
      CHECK(hbm.outstanding() == 0 && hbm.r_id < 0, "the drain ended with %zu transactions in HBM",
            hbm.outstanding());
      draining = false;
      if (switch_at_drain) {
        check_image();
        g_base = move_to;
        switch_at_drain = false;
        on = true;
      }
    }
    if (moving_tb && !rst) {
      if (dut->moving) {
        CHECK(all_zero, "free not 0 during a move");
        ++move_cycles;
      } else {
        end_move();
      }
    }
    for (int m = 0; m < N; ++m) {
      last_taken[m] = int(rd[m]) + int(wr[m]) > 0 ? 1 : 0;
      last_free[m] = dut->free[m];
      free_hist[m].push_back(rst || draining || !on ? 0 : last_free[m]);
      if (free_hist[m].size() > size_t(kLead) + 1) free_hist[m].pop_front();
      if (!draining && !rst) max_queue = std::max<uint64_t>(max_queue, uint64_t(kFull - last_free[m]) / 2);
    }
    ++now;
  }

  void end_move() {
    moving_tb = false;
    for (int m = 0; m < N; ++m) {
      CHECK(dut->free[m] == kFull, "member %d: free %u as the move ended", m, uint32_t(dut->free[m]));
      for (uint32_t i = 0; i < kInstances; ++i)
        CHECK(!pending[m][i], "the move ended with member %d instance %u's read unanswered", m, i);
    }
    CHECK(hbm.outstanding() == 0 && hbm.r_id < 0, "the move ended with %zu transactions in HBM",
          hbm.outstanding());
    check_image();
    g_base = move_to;
  }

  bool queued() const {
    for (int m = 0; m < N; ++m)
      if (last_free[m] < kFull) return true;
    return false;
  }

  int free_instance() const {
    for (uint32_t i = 0; i < kInstances; ++i)
      if (!any_pending(i)) return int(i);
    return -1;
  }

  void begin_move(uint64_t to) {
    ++n_moves;
    if (hbm.outstanding() != 0) ++n_moves_busy;
    if (queued()) ++n_moves_queued;
    move_to = to;
    moving_tb = true;
    move_pulse = true;
    on = false;
    zero = zeroed = false;
    base_in = to;
  }

  void finish_move(bool clear, bool raise) {
    for (uint64_t c = 0; moving_tb && c < 400000; ++c) cycle(false, true);
    CHECK(!moving_tb, "the move never ends");
    if (clear) host_zero(raise);
    on = true;
  }

  void reset_dut(int n) {
    if (queued()) ++n_resets_queued;
    zero = zeroed = false;
    for (int i = 0; i < n; ++i) cycle(true);
  }

  void quiesce() {
    for (int i = 0; i < 200000; ++i) {
      bool any = false;
      for (uint32_t k = 0; k < kInstances; ++k) any |= any_pending(k);
      bool full = true;
      for (int m = 0; m < N; ++m) full &= dut->free[m] == kFull;
      if (!any && hbm.outstanding() == 0 && full && !draining && !moving_tb) break;
      cycle(false, false);
    }
    for (int i = 0; i < 8; ++i) cycle(false, false);
    for (uint32_t k = 0; k < kInstances; ++k) CHECK(!any_pending(k), "reads never answered");
  }

  void check_image() {
    for (auto& kv : ref.rows) {
      for (int k = 0; k < kBeats; ++k) {
        const Line& l = hbm.line(kv.first + 64 * k);
        for (int i = 0; i < 64; ++i) {
          const int j = k * 64 + i;
          const uint8_t want = j < kBytes ? kv.second[j] : zero_at(kv.first) ? 0 : init_byte(kv.first + j);
          if (l[i] != want) {
            CHECK(false, "HBM row 0x%lx byte %d: 0x%02x, want 0x%02x", (unsigned long)kv.first, j, l[i], want);
            return;
          }
        }
      }
    }
  }

  void host_zero(bool raise = true) {
    CHECK(hbm.outstanding() == 0 && !draining && !moving_tb, "the host clears the image with HBM busy");
    for (auto it = hbm.mem.lower_bound(g_base); it != hbm.mem.end() && it->first < g_base + kImageBytes;)
      it = hbm.mem.erase(it);
    for (auto it = ref.rows.lower_bound(g_base); it != ref.rows.end() && it->first < g_base + kImageBytes;)
      it = ref.rows.erase(it);
    g_zeroed.insert(g_base);
    zeroed = true;
    zero = raise;
  }

  void start(const Setup& setup, uint64_t seed) {
    s = &setup;
    setup_name = setup.name;
    rng.seed(seed);
    now = 0;
    hbm = Hbm();
    hbm.lat_lo = setup.lat_lo;
    hbm.lat_hi = setup.lat_hi;
    hbm.ready_p = setup.ready_p;
    ref = Ref();
    pending.assign(N, std::vector<bool>(kInstances, false));
    expect.assign(N, std::vector<Row>(kInstances, Row(kBytes)));
    free_hist.assign(N, {});
    last_free.assign(N, 0);
    last_taken.assign(N, 0);
    draining = zero = zeroed = false;
    g_base = base_in = move_to = kBase;
    g_zeroed.clear();
    on = true;
    move_pulse = moving_tb = switch_at_drain = false;
    n_reads = n_writes = n_checked = n_multi = n_resets = n_resets_busy = n_moves = n_moves_busy = dropped = 0;
    drain_cycles = move_cycles = n_toggles = lead_taken = max_queue = n_moves_queued = n_resets_queued = 0;
    dut = new Vfoldsim_vmem_group_tb();
    dut->base = kBase;
    dut->move = 0;
    dut->clk = 0;
    dut->eval();
    reset_dut(4);
    while (draining) cycle(false, false);
    if (setup.zero) host_zero();
  }

  int run(const Setup& setup, uint64_t seed) {
    const int before = failures;
    start(setup, seed);
    toggles = setup.zero != 0;
    std::vector<bool> events;
    for (int k = 0; k < setup.moves / 2; ++k) events.push_back(true);
    for (int k = 0; k < setup.resets; ++k) events.push_back(false);
    for (int k = setup.moves / 2; k < setup.moves; ++k) events.push_back(true);
    std::vector<uint64_t> regions = {kBase};
    const uint64_t slice = setup.cycles / uint64_t(events.size() + 1);
    for (size_t k = 0; k <= events.size(); ++k) {
      for (uint64_t c = 0; c < slice; ++c) cycle();
      if (k == events.size()) break;
      const auto lo = hbm.lat_lo, hi = hbm.lat_hi;
      const double rp = hbm.ready_p;
      hbm.lat_lo = 200;
      hbm.lat_hi = 600;
      hbm.ready_p = 0.3;
      for (int c = 0; c < 300; ++c) cycle();
      for (int c = 0; c < 200000 && !hbm.busy_ahead(); ++c) cycle();
      if (events[k]) {
        const uint64_t pick = rand_in(0, 2);
        uint64_t to = g_base;
        if (pick == 1) {
          to = kBase + regions.size() * kStride;
          regions.push_back(to);
        } else if (pick == 2) {
          to = regions[rand_in(0, regions.size() - 1)];
        }
        // A state pass touching every member just before the move, and one
        // in its own cycle (within the simulator's lead), when an instance
        // is free: queued in the FIFOs as the move comes.
        int i = free_instance();
        for (int c = 0; c < 20000 && i < 0; ++c) {
          cycle(false, false);
          i = free_instance();
        }
        if (i >= 0) cycle(false, true, i);
        begin_move(to);
        cycle(false, true, free_instance());
        finish_move(chance(setup.zero ? 0.75 : 0.4), setup.zero != 0);
      } else {
        // A state pass touching every member just before the reset: its
        // requests are still queued as the reset comes.
        int i = free_instance();
        for (int c = 0; c < 20000 && i < 0; ++c) {
          cycle(false, false);
          i = free_instance();
        }
        if (i >= 0) cycle(false, true, i);
        reset_dut(int(rand_in(1, 5)));
        uint64_t drain = 0;
        while (draining && drain++ < 400000) cycle(false, false);
        CHECK(!draining, "the drain never ends");
        if (setup.zero && chance(0.75)) host_zero();
      }
      hbm.lat_lo = lo;
      hbm.lat_hi = hi;
      hbm.ready_p = rp;
    }
    quiesce();
    reset_dut(2);
    while (draining) cycle(false, false);
    check_image();
    printf("%-8s %lu cycles: %lu reads checked of %lu, %lu writes, %lu host cycles with several members; "
           "HBM %lu AR %lu AW; a member's FIFO up to %lu; %lu resets (%lu with HBM busy, %lu with requests "
           "queued, %lu reads dropped, %lu drain cycles); %lu moves (%lu with HBM busy, %lu with requests "
           "queued, %lu cycles, %lu requests taken in them); "
           "zero toggled %lu times: %s\n",
           setup.name, (unsigned long)now, (unsigned long)n_checked, (unsigned long)n_reads,
           (unsigned long)n_writes, (unsigned long)n_multi, (unsigned long)hbm.n_ar, (unsigned long)hbm.n_aw,
           (unsigned long)max_queue, (unsigned long)n_resets, (unsigned long)n_resets_busy,
           (unsigned long)n_resets_queued, (unsigned long)dropped, (unsigned long)drain_cycles,
           (unsigned long)n_moves, (unsigned long)n_moves_busy, (unsigned long)n_moves_queued,
           (unsigned long)move_cycles, (unsigned long)lead_taken,
           (unsigned long)n_toggles, failures == before ? "ok" : "FAILED");
    CHECK(n_multi > 0 || setup.p_issue < 0.2, "no host cycle with several members");
    CHECK(n_moves_queued >= uint64_t(setup.moves) && n_resets_queued >= uint64_t(setup.resets),
          "too few resets (%lu) or moves (%lu) with requests queued", (unsigned long)n_resets_queued,
          (unsigned long)n_moves_queued);
    toggles = false;
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
  printf("foldsim_vmem_group: setup %d, %d members, IW %d INSTANCES %u AW %d DW %d MW %d RDW_READ_FIRST %d "
         "CACHE_ROWS %d FIFO %d\n",
         VG_CFG, N, C.iw, kInstances, C.aw, C.dw, C.mw, C.rdwf, C.cache, kFifo);
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
