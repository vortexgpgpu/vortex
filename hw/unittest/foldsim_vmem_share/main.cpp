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

// foldsim_vmem_share (hw/rtl/afu/foldsim/foldsim_vmem_share.sv): NC virtual-
// memory controllers on one AXI4 master, each driven as a folded simulator
// drives its controller (an instance issues only once its read is answered,
// and only as much as `free` allowed L host cycles earlier), each in a region
// of its own, against a model of HBM that answers with random latency and
// ready on every channel (R and B held until taken), returns reads out of
// order and interleaved across ids and in order within one, applies a write
// and samples a read at random points of their flight, and checks the AXI4
// protocol on the shared master. The controllers' ids collide (the kernel's
// 1-bit id gives each controller two): reads of one id from several
// controllers are in flight at once. Under several traffic setups:
//   * every controller's every response equals a reference RAM's row;
//   * every controller makes progress (none starves);
//   * free never drops by more than the requests taken, never exceeds QUEUE;
//   * resets (all controllers at once, as the kernel's) with transactions in
//     flight: each controller's drain ends with free at QUEUE and nothing of
//     its region left in HBM, though the others' still are;
//   * region moves (all at once, as the kernel's): each controller's ends with
//     its region idle in HBM and its reads answered; then the image is the
//     reference's in every region;
//   * after a final reset, HBM's image equals the reference RAM.
//
//   ./foldsim_vmem_share_tb            every setup
//   ./foldsim_vmem_share_tb <name>     one setup

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

#include "Vfoldsim_vmem_share_tb.h"
#include "verilated.h"

#ifndef VM_NC
#define VM_NC 3
#endif
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
#ifndef VM_AXI_ID_W
#define VM_AXI_ID_W 1
#endif
#ifndef VM_OUTSTANDING
#define VM_OUTSTANDING (1 << VM_AXI_ID_W)
#endif
#ifndef VM_INSTANCES
#define VM_INSTANCES (1u << VM_IW)
#endif
#ifndef VM_CACHE_ROWS
#define VM_CACHE_ROWS 0
#endif

namespace {

constexpr int kNc = VM_NC;
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
constexpr int kFw = [] {
  int w = 0;
  while ((1 << w) < VM_QUEUE + 1) ++w;
  return w;
}();
constexpr uint64_t kImageBytes = (uint64_t(kRows) * kInstances) << kRowLog2;
// Controller k's region: the vmem space's base + k regions, 4 KiB aligned as
// the compiler places them.
constexpr uint64_t kRegion = (kImageBytes + 4095) & ~4095ull;
constexpr uint64_t kSpace = kRegion * kNc;
constexpr uint64_t kBase = 0x123400000ull;
constexpr uint64_t kStride = 0x104000000ull;
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

// ---- Ports of any width, at a bit offset ----
template <typename T>
bool get_bit(const T& p, int i) {
  if constexpr (std::is_integral_v<T>)
    return (uint64_t(p) >> i) & 1;
  else
    return (p[i / 32] >> (i % 32)) & 1;
}
template <typename T>
void set_bit(T& p, int i, bool v) {
  if constexpr (std::is_integral_v<T>) {
    const T m = T(T(1) << i);
    p = v ? T(p | m) : T(p & ~m);
  } else {
    const uint32_t m = 1u << (i % 32);
    p[i / 32] = v ? (p[i / 32] | m) : (p[i / 32] & ~m);
  }
}
template <typename T>
uint64_t get_u(const T& p, int lo, int w) {
  uint64_t v = 0;
  for (int i = 0; i < w; ++i) v |= uint64_t(get_bit(p, lo + i)) << i;
  return v;
}
template <typename T>
void set_u(T& p, int lo, int w, uint64_t v) {
  for (int i = 0; i < w; ++i) set_bit(p, lo + i, (v >> i) & 1);
}
// Bytes at bit `lo`: by words where the offset allows it.
template <typename T>
void set_bytes(T& p, int lo, const uint8_t* b, int n) {
  if constexpr (!std::is_integral_v<T>) {
    if (lo % 32 == 0 && n % 4 == 0) {
      for (int w = 0; w < n / 4; ++w)
        p[lo / 32 + w] = uint32_t(b[4 * w]) | uint32_t(b[4 * w + 1]) << 8 | uint32_t(b[4 * w + 2]) << 16 |
                         uint32_t(b[4 * w + 3]) << 24;
      return;
    }
  }
  for (int i = 0; i < n; ++i) set_u(p, lo + 8 * i, 8, b[i]);
}
template <typename T>
void get_bytes(const T& p, int lo, uint8_t* b, int n) {
  if constexpr (!std::is_integral_v<T>) {
    if (lo % 32 == 0) {
      for (int i = 0; i < n; ++i) b[i] = uint8_t(p[lo / 32 + i / 4] >> (8 * (i % 4)));
      return;
    }
  }
  for (int i = 0; i < n; ++i) b[i] = uint8_t(get_u(p, lo + 8 * i, 8));
}

// The vmem space's base (it changes as a move ends) and the spaces the host
// cleared.
uint64_t g_base = kBase;
std::set<uint64_t> g_zeroed;

bool zero_at(uint64_t a) {
  auto it = g_zeroed.upper_bound(a);
  if (it == g_zeroed.begin()) return false;
  --it;
  return a < *it + kSpace;
}
uint64_t region(int k) { return g_base + uint64_t(k) * kRegion; }
uint64_t row_addr(int k, uint32_t inst, uint32_t row) {
  return region(k) + ((uint64_t(row) * kInstances + inst) << kRowLog2);
}
// The controller whose region holds `a` (in the current space), or -1.
int owner(uint64_t a) {
  if (a < g_base || a >= g_base + kSpace) return -1;
  const int k = int((a - g_base) / kRegion);
  return (a - region(k)) < kImageBytes ? k : -1;
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
  // By id, in AR order: a burst answers only after the earlier ones of its id.
  std::map<int, std::deque<Read>> reads;
  struct Aw { uint64_t addr; int id; };
  std::deque<Aw> aw;
  struct Beat { std::array<uint8_t, 64> data; uint64_t strb; bool last; };
  std::deque<Beat> w;
  struct Write { uint64_t addr; int id; std::vector<Beat> beats; uint64_t apply_at, b_at; bool applied = false; };
  std::deque<Write> writes;  // in order: B in order (one id)
  uint64_t last_apply = 0, last_b = 0;

  bool arready = false, awready = false, wready = false;
  int r_id = -1;  // the R beat offered, held until taken
  bool b_held = false;
  bool ar_held = false, aw_held = false, w_held = false;
  uint64_t ar_addr = 0, aw_addr = 0;
  int ar_id = 0;
  std::array<uint8_t, 64> w_data{};
  uint64_t w_strb = 0;
  bool w_last = false;

  uint64_t n_ar = 0, n_aw = 0, n_w = 0, n_b = 0, n_r = 0, n_shared_id = 0, n_interleaved = 0,
           n_r_stalled = 0, n_b_stalled = 0;

  Line& line(uint64_t a) {
    auto it = mem.find(a);
    if (it == mem.end()) {
      Line l;
      for (int i = 0; i < 64; ++i) l[i] = zero_at(a) ? 0 : init_byte(a + i);
      it = mem.emplace(a, l).first;
    }
    return it->second;
  }

  size_t n_reads() const {
    size_t n = 0;
    for (auto& kv : reads) n += kv.second.size();
    return n;
  }
  size_t outstanding() const { return n_reads() + aw.size() + w.size() + writes.size(); }
  // Transactions of controller k's region.
  size_t outstanding(int k) const {
    size_t n = 0;
    for (auto& kv : reads)
      for (auto& r : kv.second) n += owner(r.addr) == k;
    for (auto& a : aw) n += owner(a.addr) == k;
    for (auto& wr : writes) n += owner(wr.addr) == k;
    return n;
  }
  bool busy_ahead() const {
    for (auto& kv : reads)
      for (auto& r : kv.second)
        if (r.ready_at > now + 2) return true;
    for (auto& wr : writes)
      if (wr.b_at > now + 2) return true;
    return false;
  }

  void check_addr(uint64_t a, const char* ch) {
    CHECK((a & ((1ull << kRowLog2) - 1)) == 0, "%s address 0x%lx not row aligned", ch, (unsigned long)a);
    CHECK(owner(a) >= 0, "%s address 0x%lx in no controller's region of the space at 0x%lx", ch,
          (unsigned long)a, (unsigned long)g_base);
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
    if (r_id < 0 && chance(rvalid_p)) {
      std::vector<int> ok, started;
      for (auto& kv : reads) {
        if (kv.second.empty()) continue;
        const Read& r = kv.second.front();
        if (r.sampled && r.ready_at <= now) {
          ok.push_back(kv.first);
          if (r.beat > 0) started.push_back(kv.first);
        }
      }
      if (!started.empty() && !chance(interleave_p)) {
        r_id = started[rand_in(0, started.size() - 1)];
      } else if (!ok.empty()) {
        r_id = ok[rand_in(0, ok.size() - 1)];
        if (!started.empty() && std::find(started.begin(), started.end(), r_id) == started.end())
          ++n_interleaved;
      }
    }
    dut->m_rvalid = r_id >= 0;
    if (r_id >= 0) {
      const Read& r = reads.at(r_id).front();
      set_bytes(dut->m_rdata, 0, r.data[r.beat].data(), 64);
      dut->m_rid = r_id;
      dut->m_rlast = r.beat == kBeats - 1;
    } else {
      dut->m_rid = rand_in(0, (1 << VM_AXI_ID_W) - 1);
      dut->m_rlast = chance(0.5);
    }
    if (!b_held)
      b_held = !writes.empty() && writes.front().b_at <= now && writes.front().applied && chance(rvalid_p);
    dut->m_bvalid = b_held;
    dut->m_bid = b_held ? writes.front().id : rand_in(0, (1 << VM_AXI_ID_W) - 1);
  }

  template <typename D>
  void sample(D* dut) {
    const bool arv = dut->m_arvalid, awv = dut->m_awvalid, wv = dut->m_wvalid;
    std::array<uint8_t, 64> wd;
    get_bytes(dut->m_wdata, 0, wd.data(), 64);
    if (ar_held)
      CHECK(arv && dut->m_araddr == ar_addr && int(dut->m_arid) == ar_id, "AR withdrawn or changed");
    if (aw_held)
      CHECK(awv && dut->m_awaddr == aw_addr, "AW withdrawn or changed");
    if (w_held)
      CHECK(wv && wd == w_data && uint64_t(dut->m_wstrb) == w_strb && bool(dut->m_wlast) == w_last,
            "W withdrawn or changed");
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

    if (r_id >= 0) {
      if (dut->m_rready) {
        auto& q = reads.at(r_id);
        ++n_r;
        if (++q.front().beat == kBeats) q.pop_front();
        if (q.empty()) reads.erase(r_id);
        r_id = -1;
      } else {
        ++n_r_stalled;
      }
    }
    if (b_held) {
      if (dut->m_bready) {
        ++n_b;
        writes.pop_front();
        b_held = false;
      } else {
        ++n_b_stalled;
      }
    }
    if (ar_hs) {
      ++n_ar;
      const int id = dut->m_arid;
      CHECK(id < VM_OUTSTANDING, "AR id %d beyond OUTSTANDING", id);
      check_addr(dut->m_araddr, "AR");
      Read r;
      r.addr = dut->m_araddr;
      r.ready_at = now + rand_in(lat_lo, lat_hi);
      r.sample_at = rand_in(now + 1, r.ready_at);
      auto& q = reads[id];
      if (!q.empty()) ++n_shared_id;
      q.push_back(r);
    }
    if (aw_hs) {
      ++n_aw;
      check_addr(dut->m_awaddr, "AW");
      aw.push_back({dut->m_awaddr, int(dut->m_awid)});
    }
    if (w_hs) {
      ++n_w;
      w.push_back({wd, uint64_t(dut->m_wstrb), bool(dut->m_wlast)});
    }
    while (!aw.empty() && int(w.size()) >= kBeats) {
      Write wr;
      wr.addr = aw.front().addr;
      wr.id = aw.front().id;
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
    // W comes once its AW is on the bus: at most that burst ahead of AW.
    CHECK(aw.empty() ? int(w.size()) <= kBeats : int(w.size()) < kBeats, "%zu W beats ahead of AW", w.size());
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
    for (auto& kv : reads)
      for (auto& r : kv.second) {
        if (r.sampled || r.sample_at > now) continue;
        for (int k = 0; k < kBeats; ++k) r.data.push_back(line(r.addr + 64 * k));
        r.sampled = true;
      }
  }
};

struct Ref {
  std::map<uint64_t, Row> rows;
  Row get(uint64_t a) {
    auto it = rows.find(a);
    if (it != rows.end()) return it->second;
    Row r(kBytes);
    for (int i = 0; i < kBytes; ++i) r[i] = zero_at(a) ? 0 : init_byte(a + i);
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
  uint32_t hot_rows;
  double p_hot;
  int lead;
  uint64_t lat_lo, lat_hi;
  double ready_p;
  int resets, zero, moves;
  // Controllers that issue little (the others busy): a skewed load.
  double p_quiet;
};

const Setup kSetups[] = {
  // name     cycles  issue read  write both  same  full hot p_hot lead lat        ready resets zero moves quiet
  {"hot",     200000, 0.9,  0.6,  0.6,  0.4,  0.5,  0.5, 4,  0.8,  3,   20,  600,  0.7,  3,     0,   0,    0.0},
  {"wide",    200000, 0.9,  0.7,  0.5,  0.3,  0.3,  0.5, 0,  0.0,  3,   20,  200,  0.8,  3,     0,   0,    0.0},
  {"full",    200000, 1.0,  0.7,  0.7,  0.5,  0.5,  0.5, 8,  0.5,  0,   300, 600,  0.3,  3,     0,   0,    0.0},
  {"pairs",   200000, 0.9,  0.2,  0.2,  0.9,  0.8,  0.4, 2,  0.9,  3,   20,  600,  0.6,  3,     0,   0,    0.0},
  {"fastmem", 200000, 1.0,  0.6,  0.6,  0.4,  0.4,  0.5, 4,  0.7,  0,   1,   20,   1.0,  3,     0,   0,    0.0},
  {"skew",    200000, 0.9,  0.6,  0.6,  0.4,  0.5,  0.5, 4,  0.8,  3,   20,  300,  0.5,  2,     0,   0,    0.9},
  {"zhot",    200000, 0.9,  0.6,  0.6,  0.4,  0.5,  0.5, 4,  0.8,  3,   20,  600,  0.7,  3,     1,   0,    0.0},
  {"mhot",    200000, 0.9,  0.6,  0.6,  0.4,  0.5,  0.5, 4,  0.8,  3,   20,  600,  0.7,  1,     0,   4,    0.0},
  {"mzfull",  200000, 1.0,  0.7,  0.7,  0.5,  0.5,  0.5, 8,  0.5,  0,   300, 600,  0.3,  1,     1,   4,    0.0},
};

// A controller's simulator: its instances, what it expects, its view of free.
struct Ctl {
  std::vector<bool> pending;
  std::vector<Row> expect;
  std::vector<uint64_t> issued_at;
  std::deque<uint32_t> free_hist;
  bool draining = false, moving = false;
  uint32_t last_free = 0;
  int last_taken = 0;
  uint64_t n_reads = 0, n_writes = 0, n_checked = 0, dropped = 0;
};

struct Tb {
  Vfoldsim_vmem_share_tb* dut = nullptr;
  Hbm hbm;
  Ref ref;
  const Setup* s = nullptr;
  std::array<Ctl, kNc> c;
  std::array<bool, kNc> quiet{};
  bool zero = false, zeroed = false, on = true, move_pulse = false, moving = false, switch_at_drain = false;
  uint64_t base_in = kBase, move_to = kBase;
  uint64_t n_resets = 0, n_resets_busy = 0, n_moves = 0, n_moves_busy = 0;

  void tick() {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
  }

  bool draining_any() const {
    for (auto& x : c)
      if (x.draining) return true;
    return false;
  }
  bool moving_any() const {
    for (auto& x : c)
      if (x.moving) return true;
    return false;
  }

  bool lead_ok(const Ctl& x, int n) {
    if (x.draining) return false;
    if (s->lead == 0) return on && int(x.last_free) >= n;
    if (x.free_hist.size() < size_t(kLead) + 1) return false;
    return x.free_hist.front() >= uint32_t(2 * (kLead + 1));
  }

  uint32_t pick_row() {
    if (s->hot_rows && chance(s->p_hot)) return uint32_t(rand_in(0, s->hot_rows - 1));
    return uint32_t(rand_in(0, kRows - 1));
  }

  void cycle(bool rst = false, bool issue = true) {
    struct Rq { bool rd = false, wr = false; uint32_t inst = 0, rrow = 0, wrow = 0; Row data; std::vector<bool> mask; };
    std::array<Rq, kNc> rq;
    for (int k = 0; k < kNc; ++k) {
      Ctl& x = c[k];
      Rq& q = rq[k];
      q.data.assign(kBytes, 0);
      q.mask.assign(VM_MW, true);
      if (rst || !issue || !chance(quiet[k] ? 1.0 - s->p_quiet : s->p_issue)) continue;
      for (int tries = 0; tries < 8; ++tries) {
        const uint32_t i = uint32_t(rand_in(0, kInstances - 1));
        if (!x.pending[i]) {
          q.inst = i;
          if (chance(s->p_both)) {
            q.rd = q.wr = true;
          } else {
            q.rd = chance(s->p_read);
            q.wr = !q.rd && chance(s->p_write);
          }
          break;
        }
      }
      if (q.rd && q.wr && !lead_ok(x, 2) && lead_ok(x, 1)) q.wr = false;
      if (!lead_ok(x, int(q.rd) + int(q.wr))) q.rd = q.wr = false;
      if (q.rd) q.rrow = pick_row();
      if (q.wr) {
        q.wrow = (q.rd && chance(s->p_same)) ? q.rrow : pick_row();
        for (auto& b : q.data) b = uint8_t(rand_in(0, 255));
        if (!chance(s->p_full)) {
          if (VM_MW == 1)
            q.mask[0] = chance(0.8);
          else
            for (size_t m = 0; m < q.mask.size(); ++m) q.mask[m] = chance(0.5);
        }
      }
    }

    dut->reset = rst;
    dut->zero = zero;
    dut->move = move_pulse;
    move_pulse = false;
    if (rst && moving) {
      moving = false;
      switch_at_drain = true;
      for (auto& x : c) x.moving = false;
    }
    for (int k = 0; k < kNc; ++k) {
      const Rq& q = rq[k];
      set_u(dut->base, 64 * k, 64, base_in + uint64_t(k) * kRegion);
      set_bit(dut->rd_valid, k, q.rd);
      set_u(dut->rd_inst, k * VM_IW, VM_IW, q.rd ? q.inst : rand_in(0, kInstances - 1));
      set_u(dut->rd_row, k * VM_AW, VM_AW, q.rd ? q.rrow : rand_in(0, kRows - 1));
      set_bit(dut->wr_valid, k, q.wr);
      set_u(dut->wr_inst, k * VM_IW, VM_IW, q.inst);
      set_u(dut->wr_row, k * VM_AW, VM_AW, q.wrow);
      set_bytes(dut->wr_data, k * VM_DW, q.data.data(), kBytes);
      for (int m = 0; m < VM_MW; ++m) set_bit(dut->wr_mask, k * VM_MW + m, q.mask[m]);
    }
    hbm.drive(dut);
    dut->clk = 0;
    dut->eval();

    for (int k = 0; k < kNc; ++k) {
      Ctl& x = c[k];
      const uint32_t fr = uint32_t(get_u(dut->free, k * kFw, kFw));
      if (!rst && !x.draining) {
        CHECK(fr <= VM_QUEUE, "controller %d: free %u above QUEUE", k, fr);
        if (!x.moving)
          CHECK(fr + uint32_t(x.last_taken) >= x.last_free, "controller %d: free dropped from %u to %u with %d taken",
                k, x.last_free, fr, x.last_taken);
      }
      if (get_bit(dut->rsp_valid, k)) {
        const uint32_t ri = uint32_t(get_u(dut->rsp_inst, k * VM_IW, VM_IW));
        Row got(kBytes);
        get_bytes(dut->rsp_data, k * VM_DW, got.data(), kBytes);
        CHECK(!x.draining && !rst, "controller %d: a response during reset or its drain", k);
        if (ri >= kInstances || !x.pending[ri]) {
          CHECK(false, "controller %d: a response for instance %u, which has no read", k, ri);
        } else {
          ++x.n_checked;
          CHECK(got == x.expect[ri], "controller %d instance %u: wrong row", k, ri);
          x.pending[ri] = false;
        }
      }
      const Rq& q = rq[k];
      if (q.rd || q.wr) {
        const uint64_t ra = row_addr(k, q.inst, q.rrow), wa = row_addr(k, q.inst, q.wrow);
        const bool read_first = VM_RDW_READ_FIRST != 0;
        if (q.rd && (read_first || !q.wr)) x.expect[q.inst] = ref.get(ra);
        if (q.wr) ref.write(wa, q.data, q.mask);
        if (q.rd && !read_first && q.wr) x.expect[q.inst] = ref.get(ra);
        if (q.rd) {
          ++x.n_reads;
          x.pending[q.inst] = true;
          x.issued_at[q.inst] = now;
        }
        if (q.wr) ++x.n_writes;
      }
    }

    hbm.sample(dut);
    tick();
    hbm.advance();

    for (int k = 0; k < kNc; ++k) {
      Ctl& x = c[k];
      const uint32_t fr = uint32_t(get_u(dut->free, k * kFw, kFw));
      if (rst) {
        if (!x.draining)
          for (uint32_t i = 0; i < kInstances; ++i)
            if (x.pending[i]) {
              x.pending[i] = false;
              ++x.dropped;
            }
        x.draining = true;
      } else if (x.draining && fr != 0) {
        // Its drain is over: nothing of its region in HBM, whatever the
        // others still have there.
        CHECK(fr == VM_QUEUE, "controller %d: free %u after the drain", k, fr);
        CHECK(hbm.outstanding(k) == 0, "controller %d: the drain ended with %zu of its transactions in HBM", k,
              hbm.outstanding(k));
        x.draining = false;
      }
      if (x.moving && !rst) {
        if (get_bit(dut->moving, k)) {
          CHECK(fr == 0, "controller %d: free %u during a move", k, fr);
        } else {
          CHECK(fr == VM_QUEUE, "controller %d: free %u as the move ended", k, fr);
          CHECK(hbm.outstanding(k) == 0, "controller %d: the move ended with %zu of its transactions in HBM", k,
                hbm.outstanding(k));
          for (uint32_t i = 0; i < kInstances; ++i)
            CHECK(!x.pending[i], "controller %d: the move ended with instance %u's read unanswered", k, i);
          x.moving = false;
        }
      }
      x.last_taken = int(rq[k].rd) + int(rq[k].wr);
      x.last_free = fr;
      x.free_hist.push_back(rst || x.draining || !on ? 0 : fr);
      if (x.free_hist.size() > size_t(kLead) + 1) x.free_hist.pop_front();
    }
    if (moving && !moving_any()) {
      // Every controller's move is over: the old space holds the image.
      moving = false;
      check_image();
      g_base = move_to;
    }
    if (switch_at_drain && !draining_any()) {
      check_image();
      g_base = move_to;
      switch_at_drain = false;
      on = true;
    }
    ++now;
  }

  void reset_dut(int n) {
    zero = zeroed = false;
    for (int i = 0; i < n; ++i) cycle(true);
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

  void host_zero() {
    CHECK(hbm.outstanding() == 0 && !draining_any() && !moving, "the host clears the space with HBM busy");
    for (auto it = hbm.mem.lower_bound(g_base); it != hbm.mem.end() && it->first < g_base + kSpace;)
      it = hbm.mem.erase(it);
    for (auto it = ref.rows.lower_bound(g_base); it != ref.rows.end() && it->first < g_base + kSpace;)
      it = ref.rows.erase(it);
    g_zeroed.insert(g_base);
    zeroed = zero = true;
  }

  void quiesce() {
    for (int i = 0; i < 300000; ++i) {
      bool any = false;
      for (auto& x : c)
        for (bool p : x.pending) any |= p;
      bool full = true;
      for (int k = 0; k < kNc; ++k) full &= get_u(dut->free, k * kFw, kFw) == VM_QUEUE;
      if (!any && hbm.outstanding() == 0 && full && !draining_any() && !moving) break;
      cycle(false, false);
    }
    for (int i = 0; i < 8; ++i) cycle(false, false);
    for (int k = 0; k < kNc; ++k) {
      bool any = false;
      for (bool p : c[k].pending) any |= p;
      CHECK(!any, "controller %d: reads never answered", k);
      CHECK(get_u(dut->free, k * kFw, kFw) == VM_QUEUE, "controller %d: free %lu at rest", k,
            (unsigned long)get_u(dut->free, k * kFw, kFw));
    }
    CHECK(hbm.outstanding() == 0, "HBM left with %zu transactions", hbm.outstanding());
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
    hbm.rvalid_p = std::min(0.9, setup.ready_p + 0.1);
    ref = Ref();
    for (int k = 0; k < kNc; ++k) {
      c[k] = Ctl();
      c[k].pending.assign(kInstances, false);
      c[k].expect.assign(kInstances, Row(kBytes));
      c[k].issued_at.assign(kInstances, 0);
      // The skewed load: all controllers but the first issue little.
      quiet[k] = setup.p_quiet > 0 && k != 0;
    }
    g_base = base_in = move_to = kBase;
    g_zeroed.clear();
    zero = zeroed = false;
    on = true;
    move_pulse = moving = switch_at_drain = false;
    n_resets = n_resets_busy = n_moves = n_moves_busy = 0;
    dut = new Vfoldsim_vmem_share_tb();
    dut->clk = 0;
    dut->eval();
    reset_dut(4);
    while (draining_any()) cycle(false, false);
    if (setup.zero) host_zero();
  }

  int run(const Setup& setup, uint64_t seed) {
    const int before = failures;
    start(setup, seed);
    std::vector<bool> events;
    for (int k = 0; k < setup.moves / 2; ++k) events.push_back(true);
    for (int k = 0; k < setup.resets; ++k) events.push_back(false);
    for (int k = setup.moves / 2; k < setup.moves; ++k) events.push_back(true);
    std::vector<uint64_t> spaces = {kBase};
    const uint64_t slice = setup.cycles / uint64_t(events.size() + 1);
    for (size_t e = 0; e <= events.size(); ++e) {
      for (uint64_t i = 0; i < slice; ++i) cycle();
      if (e == events.size()) break;
      const auto lo = hbm.lat_lo, hi = hbm.lat_hi;
      const double rp = hbm.ready_p;
      hbm.lat_lo = 200;
      hbm.lat_hi = 600;
      hbm.ready_p = 0.3;
      for (int i = 0; i < 300; ++i) cycle();
      for (int i = 0; i < 200000 && !hbm.busy_ahead(); ++i) cycle();
      if (events[e]) {
        // A move: to the same space, a new one, or one used before.
        ++n_moves;
        if (hbm.outstanding() != 0) ++n_moves_busy;
        const uint64_t pick = rand_in(0, 2);
        uint64_t to = g_base;
        if (pick == 1) {
          to = kBase + spaces.size() * kStride;
          spaces.push_back(to);
        } else if (pick == 2) {
          to = spaces[rand_in(0, spaces.size() - 1)];
        }
        move_to = to;
        moving = true;
        for (auto& x : c) x.moving = true;
        move_pulse = true;
        on = false;
        zero = zeroed = false;
        base_in = to;
        for (uint64_t i = 0; moving && i < 400000; ++i) cycle();
        CHECK(!moving, "the move never ends");
        if (chance(setup.zero ? 0.75 : 0.4)) host_zero();
        if (!setup.zero) zero = false;
        on = true;
      } else {
        ++n_resets;
        if (hbm.outstanding() != 0) ++n_resets_busy;
        reset_dut(int(rand_in(1, 5)));
        for (uint64_t i = 0; draining_any() && i < 300000; ++i) cycle(false, false);
        CHECK(!draining_any(), "the drain never ends");
        if (setup.zero && chance(0.75)) host_zero();
      }
      hbm.lat_lo = lo;
      hbm.lat_hi = hi;
      hbm.ready_p = rp;
    }
    quiesce();
    reset_dut(2);
    for (uint64_t i = 0; draining_any() && i < 300000; ++i) cycle(false, false);
    CHECK(hbm.outstanding() == 0, "HBM left with %zu transactions after the last drain", hbm.outstanding());
    check_image();

    for (int k = 0; k < kNc; ++k)
      CHECK(c[k].n_checked > 0, "controller %d answered nothing", k);
    CHECK(n_resets_busy >= uint64_t(setup.resets), "only %lu of the resets found HBM busy",
          (unsigned long)n_resets_busy);
    CHECK(n_moves_busy >= uint64_t(setup.moves), "only %lu of the moves found HBM busy", (unsigned long)n_moves_busy);
    printf("%-8s %lu cycles: reads checked", setup.name, (unsigned long)now);
    for (int k = 0; k < kNc; ++k)
      printf(" %lu/%lu", (unsigned long)c[k].n_checked, (unsigned long)c[k].n_reads);
    printf("; writes");
    for (int k = 0; k < kNc; ++k) printf(" %lu", (unsigned long)c[k].n_writes);
    printf("; HBM %lu AR (%lu behind another of their id) %lu AW, %lu R beats (%lu interleaved bursts, %lu held), "
           "%lu B (%lu held); %lu resets (%lu busy), %lu moves (%lu busy): %s\n",
           (unsigned long)hbm.n_ar, (unsigned long)hbm.n_shared_id, (unsigned long)hbm.n_aw, (unsigned long)hbm.n_r,
           (unsigned long)hbm.n_interleaved, (unsigned long)hbm.n_r_stalled, (unsigned long)hbm.n_b,
           (unsigned long)hbm.n_b_stalled, (unsigned long)n_resets, (unsigned long)n_resets_busy,
           (unsigned long)n_moves, (unsigned long)n_moves_busy, failures == before ? "ok" : "FAILED");
    delete dut;
    dut = nullptr;
    return failures - before;
  }
};

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  printf("foldsim_vmem_share: %d controllers, IW %d AW %d DW %d MW %d, %d-bit ids (%d reads each), "
         "cache rows %d\n",
         kNc, VM_IW, VM_AW, VM_DW, VM_MW, VM_AXI_ID_W, VM_OUTSTANDING, VM_CACHE_ROWS);
  const char* only = nullptr;
  for (int i = 1; i < argc; ++i)
    if (argv[i][0] != '+') only = argv[i];
  Tb tb;
  uint64_t seed = 1;
  for (const auto& s : kSetups) {
    ++seed;
    if (only && strcmp(only, s.name) != 0) continue;
    tb.run(s, seed);
  }
  if (failures) {
    printf("FAILED: %d failures\n", failures);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
