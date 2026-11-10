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

// foldsim's virtual-memory controller (Foldsim shell/rtl/foldsim_vmem.sv),
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
// The z* setups clear the image (the host's zeroing, after a reset) and raise
// `zero`, dropping and raising it again at random under traffic; a reset
// drops it, and the host clears the image again, or not (then `zero` stays
// low). Every answer still equals the reference RAM's row.
// With a host cache, directed sequences first check where answers and writes
// go: a read-first pair's whole-row write fills its line, its read still
// getting the old row from memory (miss) or the line (hit); later reads hit;
// a fill evicts the line's dirty row; a partial write that misses goes
// around; and, where instances share lines, a pinned line stays in the cache
// (another instance's write goes around) until its pair's read has its data.
// With known-zero rows, directed steps check that reads of rows not written
// since the reset are answered on chip, as zeros, with no read of memory;
// that writes, partial or whole, in the cache or around it, and a pair's
// write, take their row out of it (a partial write made whole); that a row
// evicted after its write reads back its data; that `zero` raised or dropped
// with reads in memory changes only the reads looked up after it (a request
// is looked up as it is taken, or two cycles later through the multiply);
// that dropping it lets the host change the image; that a reset forgets
// every write, and answers no known-zero read taken before it.
// Region moves (`move`, as a new process places the image elsewhere without a
// reset): the m* setups move it under traffic, to a new region, back to an
// earlier one or to the same one, the base's high word written with the low
// one or later (during the move or after it). Through a move free is 0; every
// read taken before it, or within the lead after it, is answered from the old
// region; the move ends with HBM idle and the old region holding its image
// (every dirty line written back there, nothing to the new one); after it the
// image is the new region's (the host clears it and raises `zero`, or not), so
// no answer may come from a line or a known-zero bit of the old one. Directed
// steps check each of these where it shows, and a reset during a move.
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
#ifndef VM_KNOWN_ZERO
#define VM_KNOWN_ZERO 1
#endif
#ifndef VM_KZ_SHIFT
#define VM_KZ_SHIFT 0
#endif
// Writes on AXI waiting for their B (the buffer frees a slot with its last W beat).
#ifndef VM_WRITES_OUT
#define VM_WRITES_OUT 128
#endif
// 0: the controller's default (DW/8 rounded up to a power of two, at least 64).
// Reads answered from the write buffer (1) or never (0).
#ifndef VM_FORWARD
#define VM_FORWARD 1
#endif
#ifndef VM_ROW_BYTES
#define VM_ROW_BYTES 0
#endif

namespace {

// A row's whole bytes; the bits of the last above DW are zero.
constexpr int kBytes = (VM_DW + 7) / 8;
constexpr uint8_t kTopMask = (VM_DW % 8) ? uint8_t((1u << (VM_DW % 8)) - 1) : uint8_t(0xff);
constexpr int kGroup = kBytes / VM_MW;
constexpr int kBeats = (VM_DW + 511) / 512;
constexpr int kRowLog2 = [] {
  int l = VM_ROW_BYTES ? 0 : 6;
  while ((1 << l) < (VM_ROW_BYTES ? VM_ROW_BYTES : kBytes)) ++l;
  return l;
}();
// Rows below 64 bytes share lines: read as the line, written with the row's
// strobes; AR and AW then address the line.
constexpr bool kSub = kRowLog2 < 6;
constexpr int kRowSpan = kSub ? (1 << kRowLog2) : kBeats * 64;
constexpr int kAddrAlign = kSub ? 6 : kRowLog2;
constexpr uint32_t kRows = 1u << VM_AW;
constexpr uint32_t kInstances = VM_INSTANCES;
constexpr uint64_t kBase = 0x123400000ull;
// Regions a move takes: kBase + k * kStride, apart in both words of the base.
constexpr uint64_t kStride = 0x104000000ull;
// The simulator's lead: its requests leave four host cycles after the
// decision on the `free` it saw (VMEM_INTERFACE.md guarantee 2).
constexpr int kLead = 4;
constexpr bool kPow2 = (kInstances & (kInstances - 1)) == 0;
// A read answered on chip: 5 host cycles, 7 through the layout's multiply.
constexpr uint64_t kLocal = kPow2 ? 5 : 7;
// A request reaches the lookups (known-zero bits, the cache's tags) this
// many host cycles after it is taken: the multiply's two stages.
constexpr int kIntake = kPow2 ? 0 : 2;
// The image's rows; instances share lines: read-first pairs pin theirs.
constexpr uint64_t kImageRows = uint64_t(kRows) * kInstances;
constexpr bool kPin = VM_CACHE_ROWS > 0 && uint64_t(VM_CACHE_ROWS) < kImageRows &&
                      (!kPow2 || uint32_t(VM_CACHE_ROWS) < kInstances);

using Row = std::vector<uint8_t>;

// A row as the RAM has it: DW bits.
Row clip(Row r) {
  if (!r.empty()) r.back() &= kTopMask;
  return r;
}
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

constexpr uint64_t kImageBytes = (uint64_t(kRows) * kInstances) << kRowLog2;

// The region the controller uses (it changes as a move ends), and the regions
// the host cleared (rows nobody wrote since read zero there).
uint64_t g_base = kBase;
std::set<uint64_t> g_zeroed;

bool zero_at(uint64_t a) {
  auto it = g_zeroed.upper_bound(a);
  if (it == g_zeroed.begin()) return false;
  --it;
  return a < *it + kImageBytes;
}

uint64_t row_addr(uint32_t inst, uint32_t row) {
  return g_base + ((uint64_t(row) * kInstances + inst) << kRowLog2);
}

uint8_t init_byte(uint64_t addr) {
  uint64_t x = addr * 0x9e3779b97f4a7c15ull;
  x ^= x >> 29;
  return uint8_t(x * 0xbf58476d1ce4e5b9ull >> 56);
}

// ---- HBM: an AXI4 slave ----
struct Hbm {
  std::map<uint64_t, Line> mem;
  // Lines never written hold zero where the host cleared the region
  // (zero_at), else a pseudo-random pattern.
  double ready_p = 0.7, rvalid_p = 0.8, interleave_p = 0.3;
  uint64_t lat_lo = 20, lat_hi = 600;
  // A read samples memory at the end of its flight (else at a random point).
  bool late_sample = false;

  struct Read {
    uint64_t addr;
    uint64_t sample_at, ready_at;
    bool sampled = false;
    std::vector<Line> data;
    int beat = 0;
  };
  std::map<int, std::deque<Read>> reads;  // by id, in AR order (AXI answers an id in order)
  size_t n_reads = 0;
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
  std::vector<uint64_t> aw_log;  // AW addresses, in order
  std::vector<bool> whole_log;   // each write, in order: every byte of its row enabled

  Line& line(uint64_t a) {
    CHECK((a & 63) == 0, "line address 0x%lx", (unsigned long)a);
    auto it = mem.find(a);
    if (it == mem.end()) {
      Line l;
      for (int i = 0; i < 64; ++i) l[i] = zero_at(a) ? 0 : init_byte(a + i);
      it = mem.emplace(a, l).first;
    }
    return it->second;
  }

  // A byte of memory.
  uint8_t& byte(uint64_t a) { return line(a & ~63ull)[a & 63]; }

  size_t outstanding() const { return n_reads + aw.size() + w.size() + writes.size(); }

  // A transaction that cannot finish within the next two cycles.
  bool busy_ahead() const {
    for (auto& kv : reads)
      for (auto& r : kv.second)
        if (r.ready_at > now + 2) return true;
    for (auto& wr : writes)
      if (wr.b_at > now + 2) return true;
    return false;
  }

  void check_addr(uint64_t a, const char* ch) {
    CHECK((a & ((1ull << kAddrAlign) - 1)) == 0, "%s address 0x%lx not row aligned", ch, (unsigned long)a);
    CHECK(a >= (g_base & ~63ull) && a < g_base + kImageBytes, "%s address 0x%lx out of the image at 0x%lx", ch,
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
    // R: hold a beat not taken; else maybe offer one.
    if (r_id < 0 && chance(rvalid_p)) {
      std::vector<int> ok, started;
      for (auto& kv : reads) {
        const Read& r = kv.second.front();
        if (r.sampled && r.ready_at <= now) {
          ok.push_back(kv.first);
          if (r.beat > 0) started.push_back(kv.first);
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
      const Read& r = reads.at(r_id).front();
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
      auto& q = reads.at(r_id);
      ++n_r;
      if (++q.front().beat == kBeats) {
        q.pop_front();
        --n_reads;
        if (q.empty()) reads.erase(r_id);
      }
      r_id = -1;
    }
    if (bvalid) {
      ++n_b;
      writes.pop_front();
    }
    if (ar_hs) {
      ++n_ar;
      const int id = dut->m_arid;
      CHECK(id < (1 << VM_AXI_ID_W), "AR id %d", id);
      CHECK(n_reads < size_t(VM_OUTSTANDING), "AR beyond OUTSTANDING reads in flight");
      check_addr(dut->m_araddr, "AR");
      Read r;
      r.addr = dut->m_araddr;
      r.ready_at = now + rand_in(lat_lo, lat_hi);
      r.sample_at = late_sample ? r.ready_at : rand_in(now + 1, r.ready_at);
      const uint64_t lat = r.ready_at - now;
      max_lat = std::max(max_lat, lat);
      min_lat = std::min(min_lat, lat);
      reads[id].push_back(r);
      ++n_reads;
    }
    if (aw_hs) {
      CHECK(n_aw - n_b < uint64_t(VM_WRITES_OUT), "AW beyond WRITES_OUT writes waiting for B");
      ++n_aw;
      check_addr(dut->m_awaddr, "AW");
      aw.push_back({dut->m_awaddr});
      aw_log.push_back(dut->m_awaddr);
    }
    if (w_hs) {
      ++n_w;
      w.push_back({wd, uint64_t(dut->m_wstrb), bool(dut->m_wlast)});
    }
    // A write is whole with its AW and its beats.
    while (!aw.empty() && int(w.size()) >= kBeats) {
      Write wr;
      wr.addr = aw.front().addr;
      bool whole = true;
      // A row of a shared line: its strobes in one row's place.
      int place = -1;
      for (int k = 0; k < kBeats; ++k) {
        const Beat& b = w.front();
        CHECK(b.last == (k == kBeats - 1), "WLAST %d on beat %d of %d", int(b.last), k, kBeats);
        for (int i = 0; i < 64; ++i) {
          const bool on = (b.strb >> i) & 1;
          if (kSub) {
            if (on) {
              if (place < 0) place = i >> kRowLog2;
              CHECK(place == (i >> kRowLog2), "strobes in two rows of a line");
            }
            continue;
          }
          if (k * 64 + i >= kBytes)
            CHECK(!on, "strobe on padding byte %d", k * 64 + i);
          else
            whole &= on;
        }
        if (kSub) {
          whole = place >= 0;
          for (int j = 0; j < (1 << kRowLog2) && place >= 0; ++j) {
            const bool on = (b.strb >> ((place << kRowLog2) + j)) & 1;
            if (j >= kBytes)
              CHECK(!on, "strobe on padding byte %d", j);
            else
              whole &= on;
          }
        }
        wr.beats.push_back(b);
        w.pop_front();
      }
      // A row of a shared line: the row the strobes name.
      if (kSub && place >= 0) aw_log[whole_log.size()] = wr.addr + (uint64_t(place) << kRowLog2);
      whole_log.push_back(whole);
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
    for (auto& kv : reads)
      for (Read& r : kv.second) {
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
    for (int i = 0; i < kBytes; ++i) r[i] = zero_at(a) ? 0 : init_byte(a + i);
    return clip(r);
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
  int zero;          // the host clears the image and raises `zero` (0: never)
  int moves;         // region moves with traffic in flight
};

const Setup kSetups[] = {
  // name        cycles  issue read  write both  same  full active hot  p_hot lead lat        ready resets zero
  {"hot",        300000,  0.9,  0.6,  0.6,  0.4,  0.5,  0.5, 8,     4,   0.8,  3,   20,  600,  0.7,  4,     0, 0},
  {"wide",       300000,  0.9,  0.7,  0.5,  0.3,  0.3,  0.5, 0,     0,   0.0,  3,   20,  200,  0.8,  4,     0, 0},
  {"full",       300000,  1.0,  0.7,  0.7,  0.5,  0.5,  0.5, 0,     8,   0.5,  0,   300, 600,  0.3,  4,     0, 0},
  {"pairs",      300000,  0.9,  0.2,  0.2,  0.9,  0.8,  0.4, 16,    2,   0.9,  3,   20,  600,  0.6,  4,     0, 0},
  {"partial",    300000,  0.9,  0.6,  0.6,  0.4,  0.5,  0.1, 8,     2,   0.9,  0,   20,  300,  0.7,  4,     0, 0},
  {"fastmem",    300000,  1.0,  0.6,  0.6,  0.4,  0.4,  0.5, 32,    4,   0.7,  0,   1,   20,   1.0,  4,     0, 0},
  {"sparse",     300000,  0.1,  0.6,  0.6,  0.4,  0.6,  0.5, 4,     2,   0.9,  3,   20,  600,  0.5,  4,     0, 0},
  {"zhot",       300000,  0.9,  0.6,  0.6,  0.4,  0.5,  0.5, 8,     4,   0.8,  3,   20,  600,  0.7,  4,     1, 0},
  {"zwide",      300000,  0.9,  0.7,  0.5,  0.3,  0.3,  0.5, 0,     0,   0.0,  3,   20,  200,  0.8,  4,     1, 0},
  {"zpairs",     300000,  0.9,  0.2,  0.2,  0.9,  0.8,  0.4, 16,    2,   0.9,  3,   20,  600,  0.6,  4,     1, 0},
  {"zpartial",   300000,  0.9,  0.6,  0.6,  0.4,  0.5,  0.1, 8,     2,   0.9,  0,   20,  300,  0.7,  4,     1, 0},
  {"zfast",      300000,  1.0,  0.6,  0.6,  0.4,  0.4,  0.5, 32,    4,   0.7,  0,   1,   20,   1.0,  4,     1, 0},
  // name        cycles  issue read  write both  same  full active hot  p_hot lead lat        ready resets zero moves
  {"mhot",       300000,  0.9,  0.6,  0.6,  0.4,  0.5,  0.5, 8,     4,   0.8,  3,   20,  600,  0.7,  1,     0,   5},
  {"mpairs",     300000,  0.9,  0.2,  0.2,  0.9,  0.8,  0.4, 16,    2,   0.9,  3,   20,  600,  0.6,  1,     0,   5},
  {"mfull",      300000,  1.0,  0.7,  0.7,  0.5,  0.5,  0.5, 0,     8,   0.5,  0,   300, 600,  0.3,  1,     0,   5},
  {"mzhot",      300000,  0.9,  0.6,  0.6,  0.4,  0.5,  0.5, 8,     4,   0.8,  3,   20,  600,  0.7,  1,     1,   5},
  {"mzpartial",  300000,  0.9,  0.6,  0.6,  0.4,  0.5,  0.1, 8,     2,   0.9,  0,   20,  300,  0.7,  1,     1,   5},
  {"mzfast",     300000,  1.0,  0.6,  0.6,  0.4,  0.4,  0.5, 32,    4,   0.7,  0,   1,   20,   1.0,  1,     1,   5},
};

// A request the directed sequences force.
struct Req {
  bool rd = false, wr = false;
  uint32_t inst = 0, rrow = 0, wrow = 0;
  Row data;
  std::vector<bool> mask;
};

struct Tb {
  Vfoldsim_vmem* dut = nullptr;
  const Req* force = nullptr;
  std::vector<uint64_t> rsp_lat;
  Hbm hbm;
  Ref ref;
  const Setup* s = nullptr;

  std::vector<bool> pending;
  std::vector<Row> expect;
  std::vector<uint64_t> issued_at;
  std::deque<uint32_t> free_hist;
  bool in_reset = false, draining = false;
  // `zero`, and whether the host cleared the image since the last reset
  // (only then may `zero` be raised).
  bool zero = false, zeroed = false, toggles = false;
  uint64_t n_toggles = 0, drain_cycles = 0;
  uint32_t last_free = 0;
  int last_taken = 0;
  // The host's side of a move: the base it gives the controller, and whether
  // the simulator may issue (the AFU's vmem_on: off from the move, the write of
  // the base's low word, until the host is done with the region); the move
  // under way, and where to (the region the controller uses once it ends, or
  // once a reset that came during it has drained).
  uint64_t base_in = kBase, move_to = kBase, moved_at = 0;
  bool on = true, move_pulse = false, moving_tb = false, switch_at_drain = false;
  uint64_t n_moves = 0, n_moves_busy = 0, move_cycles = 0, lead_taken = 0;

  // Writes taken and not yet acknowledged by HBM, by row address: without a
  // host cache the controller issues writes in request order, so B's come in
  // that order.
  std::deque<uint64_t> wr_order;
  std::map<uint64_t, int> wr_inflight;

  uint64_t n_reads = 0, n_writes = 0, n_checked = 0, n_pairs_same = 0, n_raw = 0, n_resets = 0,
           n_resets_busy = 0, max_rsp_lat = 0, full_cycles = 0, dropped = 0, n_quick = 0,
           n_fwd = 0;
  // Without a cache: whether each instance's read met a write of its row in
  // flight (answered on chip, it came from the write buffer).
  std::vector<bool> raw_at = std::vector<bool>(kInstances, false);

  void tick() {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
  }

  void reset_dut(int n) {
    // The register holding `zero` is reset with the controller.
    zero = zeroed = false;
    in_reset = true;
    for (int i = 0; i < n; ++i) cycle(true);
    in_reset = false;
  }

  bool lead_ok(int n) {
    if (draining) return false;
    if (s->lead == 0) return on && int(last_free) >= n;
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
    if (!rst && force) {
      rd = force->rd;
      wr = force->wr;
      inst = force->inst;
      rrow = force->rrow;
      wrow = force->wrow;
      wdata = force->data;
      mask = force->mask;
    } else if (!rst && issue && chance(s->p_issue)) {
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
        wdata = clip(wdata);
        if (!chance(s->p_full)) {
          if (VM_MW == 1) {
            mask[0] = chance(0.8);
          } else {
            for (size_t k = 0; k < mask.size(); ++k) mask[k] = chance(0.5);
          }
        }
      }
    }

    // The host drops `zero` and raises it again at random, while the image
    // stays as it cleared it.
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
      // A reset during a move: its drain writes the old region back, and the
      // new one is used after it.
      moving_tb = false;
      switch_at_drain = true;
    }
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
      if (!draining && !moving_tb)
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
        rsp_lat[ri] = now - issued_at[ri];
        // Answered without memory: the cache or the write buffer.
        if (now - issued_at[ri] <= kLocal + 1) {
          ++n_quick;
          if (raw_at[ri]) ++n_fwd;
        }
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
      if (moving_tb) ++lead_taken;
      if (rd) {
        ++n_reads;
        if (VM_CACHE_ROWS == 0 && wr_inflight.count(ra)) ++n_raw;
        raw_at[inst] = VM_CACHE_ROWS == 0 && wr_inflight.count(ra);
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
    } else if (draining && dut->free == 0) {
      ++drain_cycles;
    } else if (draining && dut->free != 0) {
      CHECK(dut->free == VM_QUEUE, "free %u after the drain", uint32_t(dut->free));
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
    // A move: free 0 while `moving`, then QUEUE, with HBM idle.
    if (moving_tb && !rst) {
      if (dut->moving) {
        CHECK(dut->free == 0, "free %u during a move", uint32_t(dut->free));
        ++move_cycles;
      } else {
        end_move();
      }
    }
    last_taken = int(rd) + int(wr);
    last_free = dut->free;
    free_hist.push_back(rst || draining || !on ? 0 : last_free);
    if (free_hist.size() > size_t(kLead) + 1) free_hist.pop_front();
    ++now;
  }

  bool in_reset_tail() const { return draining; }

  // The move is over: every read taken before it answered, HBM idle, every
  // region holding its image (the old one's dirty rows written back), free
  // back at QUEUE. The controller now uses the new region.
  void end_move() {
    moving_tb = false;
    CHECK(dut->free == VM_QUEUE, "free %u as the move ended", uint32_t(dut->free));
    CHECK(hbm.outstanding() == 0 && hbm.r_id < 0, "the move ended with %zu transactions in HBM",
          hbm.outstanding());
    for (uint32_t i = 0; i < kInstances; ++i)
      CHECK(!pending[i], "the move ended with the read of instance %u unanswered", i);
    check_image();
    g_base = move_to;
  }

  // The host writes the base's low word (`split`: the high word stays the old
  // one for now): the move starts, `zero` drops, the simulator sees no room.
  void begin_move(uint64_t to, bool split) {
    ++n_moves;
    if (hbm.outstanding() != 0) ++n_moves_busy;
    move_to = to;
    moving_tb = true;
    moved_at = now;
    move_pulse = true;
    on = false;
    zero = zeroed = false;
    base_in = split ? ((base_in & ~0xffffffffull) | (to & 0xffffffffull)) : to;
  }

  // Until the move ends (the base's high word written `hi_at` cycles after
  // the low one, during the move, or after it); then the host clears the new
  // region and raises `zero`, or not, and lets the simulator on.
  void finish_move(bool clear, bool raise, bool issue = false, int64_t hi_at = -1) {
    for (uint64_t c = 0; moving_tb && c < 400000; ++c) {
      if (hi_at >= 0 && int64_t(now - moved_at) >= hi_at) base_in = move_to;
      cycle(false, issue);
    }
    CHECK(!moving_tb, "the move never ends");
    if (clear) host_zero(raise);
    base_in = move_to;
    on = true;
  }

  void quiesce() {
    // Quiet for a few cycles in a row: free is back at QUEUE once a write's
    // last W beat is out, a cycle or two before memory has it.
    for (int i = 0, quiet = 0; i < 200000 && quiet < 8; ++i) {
      bool any = false;
      for (bool p : pending) any |= p;
      quiet = (!any && hbm.outstanding() == 0 && dut->free == VM_QUEUE && !draining && !moving_tb) ? quiet + 1 : 0;
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
      for (int j = 0; j < kRowSpan; ++j) {
        const uint8_t got = hbm.byte(kv.first + j);
        const uint8_t want = j < kBytes ? kv.second[j] : zero_at(kv.first) ? 0 : init_byte(kv.first + j);
        // A row's bits above DW are never read.
        const uint8_t have = j == kBytes - 1 ? uint8_t(got & kTopMask) : got;
        if (have != want) {
          CHECK(false, "HBM row 0x%lx byte %d: 0x%02x, want 0x%02x", (unsigned long)kv.first, j, got, want);
          return;
        }
      }
    }
    (void)rows;
  }

  // The host clears the image (the controller drained after a reset or a
  // move: memory holds the image, the cache nothing) and raises `zero`, or not.
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

  // A new controller, reset and drained, under a setup.
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
    pending.assign(kInstances, false);
    rsp_lat.assign(kInstances, 0);
    expect.assign(kInstances, Row(kBytes));
    issued_at.assign(kInstances, 0);
    free_hist.clear();
    wr_order.clear();
    wr_inflight.clear();
    draining = false;
    zero = zeroed = false;
    n_toggles = drain_cycles = 0;
    g_base = base_in = move_to = kBase;
    g_zeroed.clear();
    on = true;
    move_pulse = moving_tb = switch_at_drain = false;
    n_moves = n_moves_busy = move_cycles = lead_taken = 0;
    last_free = 0;
    last_taken = 0;
    n_reads = n_writes = n_checked = n_pairs_same = n_raw = n_resets = n_resets_busy = max_rsp_lat = 0;
    full_cycles = dropped = n_quick = n_fwd = 0;
    raw_at.assign(kInstances, false);

    dut = new Vfoldsim_vmem();
    dut->base = kBase;
    dut->move = 0;
    dut->clk = 0;
    dut->eval();
    reset_dut(4);
    while (draining) cycle(false, false);
    if (setup.zero) host_zero();
  }

  // The controller's telemetry counter c (foldsim_vmem `tel`, 48 bits each).
  uint64_t tel(int c) {
    uint8_t b[42];
    get_bytes(dut->tel, b, 42);
    uint64_t v = 0;
    for (int i = 0; i < 6; ++i) v |= uint64_t(b[6 * c + i]) << (8 * i);
    return v;
  }

  // Telemetry over a window of traffic between two quiet points, with no
  // reset or move: the counters move as the requests, AR and AW do.
  void check_telemetry() {
    enum { RD, WR, LOCAL, FETCH, WB, STALL, BUSY, N };
    uint64_t t0[N];
    for (int c = 0; c < N; ++c) t0[c] = tel(c);
    const uint64_t rd0 = n_reads, wr0 = n_writes, ar0 = hbm.n_ar, aw0 = hbm.n_aw, c0 = now;
    for (int c = 0; c < 20000; ++c) cycle();
    quiesce();
    uint64_t d[N];
    for (int c = 0; c < N; ++c) d[c] = tel(c) - t0[c];
    const uint64_t drd = n_reads - rd0, dwr = n_writes - wr0, dar = hbm.n_ar - ar0, daw = hbm.n_aw - aw0;
    CHECK(d[RD] == drd, "telemetry: %lu reads counted, %lu taken", (unsigned long)d[RD], (unsigned long)drd);
    CHECK(d[WR] == dwr, "telemetry: %lu writes counted, %lu taken", (unsigned long)d[WR], (unsigned long)dwr);
    CHECK(d[FETCH] == dar, "telemetry: %lu row fetches counted, %lu AR", (unsigned long)d[FETCH], (unsigned long)dar);
    CHECK(d[WB] == daw, "telemetry: %lu write-backs counted, %lu AW", (unsigned long)d[WB], (unsigned long)daw);
    CHECK(d[LOCAL] + d[FETCH] == d[RD], "telemetry: %lu local + %lu fetched reads, %lu taken", (unsigned long)d[LOCAL],
          (unsigned long)d[FETCH], (unsigned long)d[RD]);
    CHECK((d[BUSY] != 0) == (dar != 0) && d[BUSY] <= now - c0, "telemetry: %lu cycles with a read on AXI, %lu AR",
          (unsigned long)d[BUSY], (unsigned long)dar);
    CHECK(d[STALL] <= now - c0, "telemetry: %lu stall cycles of %lu", (unsigned long)d[STALL], (unsigned long)(now - c0));
    printf("%-8s telemetry window: %lu reads (%lu local, %lu fetched), %lu writes, %lu write-backs, %lu stall, %lu busy "
           "cycles of %lu\n", setup_name, (unsigned long)d[RD], (unsigned long)d[LOCAL], (unsigned long)d[FETCH],
           (unsigned long)d[WR], (unsigned long)d[WB], (unsigned long)d[STALL], (unsigned long)d[BUSY],
           (unsigned long)(now - c0));
  }

  // Quiet, then a last reset: memory holds the RAM's image.
  void finish() {
    quiesce();
    check_telemetry();
    reset_dut(2);
    while (draining) cycle(false, false);
    CHECK(hbm.outstanding() == 0, "HBM left with %zu transactions after the last drain", hbm.outstanding());
    check_image();
  }

  int run(const Setup& setup, uint64_t seed) {
    const int before = failures;
    start(setup, seed);
    toggles = setup.zero != 0;

    // The events between slices of traffic: the resets, or with moves, half
    // the moves, the resets, the other half.
    std::vector<bool> moves;
    for (int k = 0; k < setup.moves / 2; ++k) moves.push_back(true);
    for (int k = 0; k < setup.resets; ++k) moves.push_back(false);
    for (int k = setup.moves / 2; k < setup.moves; ++k) moves.push_back(true);
    std::vector<uint64_t> regions = {kBase};
    const uint64_t slice = setup.cycles / uint64_t(moves.size() + 1);
    for (size_t k = 0; k <= moves.size(); ++k) {
      for (uint64_t c = 0; c < slice; ++c) cycle();
      if (k == moves.size()) break;
      // A reset or a move while HBM is slow and busy.
      const auto lo = hbm.lat_lo, hi = hbm.lat_hi;
      const double rp = hbm.ready_p;
      hbm.lat_lo = 200;
      hbm.lat_hi = 600;
      hbm.ready_p = 0.3;
      for (int c = 0; c < 300; ++c) cycle();
      // Sparse traffic, or a host cache taking it all, can leave HBM idle at
      // the random point; reset once it is not.
      for (int c = 0; c < 200000 && !hbm.busy_ahead(); ++c) cycle();
      if (moves[k]) {
        // To the same region, a new one, or one used before; the base's high
        // word with the low one, or later (during the move or after it). The
        // simulator keeps issuing what its lead allows.
        const uint64_t pick = rand_in(0, 2);
        uint64_t to = g_base;
        if (pick == 1) {
          to = kBase + regions.size() * kStride;
          regions.push_back(to);
        } else if (pick == 2) {
          to = regions[rand_in(0, regions.size() - 1)];
        }
        const bool split = chance(0.5);
        const int64_t hi_at = split ? int64_t(rand_in(0, 600)) : -1;
        begin_move(to, split);
        // The host clears the new region (and raises `zero` if the setup does),
        // or not.
        finish_move(chance(setup.zero ? 0.75 : 0.4), setup.zero != 0, true, hi_at);
      } else {
        reset_dut(int(rand_in(1, 5)));
        uint64_t drain = 0;
        while (draining && drain++ < 200000) cycle(false, false);
        CHECK(!draining, "the drain never ends");
        // The host clears the image again, mostly; else `zero` stays low.
        if (setup.zero && chance(0.75)) host_zero();
      }
      hbm.lat_lo = lo;
      hbm.lat_hi = hi;
      hbm.ready_p = rp;
    }
    finish();

    printf("%-8s %lu cycles: %lu reads checked of %lu (%lu meeting a write in flight, %lu answered "
           "locally), %lu writes, "
           "%lu same-row pairs; HBM %lu AR %lu AW, latency %lu-%lu, response latency max %lu; "
           "%lu cycles with free 0; %lu resets (%lu with HBM busy, %lu reads dropped, %lu drain cycles); "
           "%lu moves (%lu with HBM busy, %lu cycles, %lu requests taken in them); "
           "zero toggled %lu times: %s\n",
           setup.name, (unsigned long)now, (unsigned long)n_checked, (unsigned long)n_reads,
           (unsigned long)n_raw, (unsigned long)n_quick, (unsigned long)n_writes,
           (unsigned long)n_pairs_same,
           (unsigned long)hbm.n_ar, (unsigned long)hbm.n_aw, (unsigned long)hbm.min_lat,
           (unsigned long)hbm.max_lat, (unsigned long)max_rsp_lat, (unsigned long)full_cycles,
           (unsigned long)n_resets, (unsigned long)n_resets_busy, (unsigned long)dropped,
           (unsigned long)drain_cycles, (unsigned long)n_moves, (unsigned long)n_moves_busy,
           (unsigned long)move_cycles, (unsigned long)lead_taken, (unsigned long)n_toggles,
           failures == before ? "ok" : "FAILED");
    // Without forwarding, no read is answered from the write buffer: one that
    // met a write of its row in flight waits for its B and reads memory.
    if (VM_CACHE_ROWS == 0) {
      printf("%-8s reads meeting a write in flight answered from the buffer: %lu of %lu\n",
             setup.name, (unsigned long)n_fwd, (unsigned long)n_raw);
      if (VM_FORWARD == 0)
        CHECK(n_fwd == 0, "FORWARD 0: %lu reads answered from the write buffer", (unsigned long)n_fwd);
    }
    // A cache with a line per row can keep HBM idle under any traffic.
    if (uint64_t(VM_CACHE_ROWS) < kImageRows) {
      CHECK(n_resets_busy >= uint64_t(setup.resets), "only %lu of the resets found HBM busy",
            (unsigned long)n_resets_busy);
      CHECK(n_moves_busy >= uint64_t(setup.moves), "only %lu of the moves found HBM busy",
            (unsigned long)n_moves_busy);
    }
    toggles = false;
    delete dut;
    dut = nullptr;
    return failures - before;
  }

  // ---- Directed: where a host cache's answers and writes go ----
  static uint32_t line_of(uint32_t inst, uint32_t row) {
    return uint32_t((uint64_t(row) * kInstances + inst) % uint64_t(VM_CACHE_ROWS ? VM_CACHE_ROWS : 1));
  }

  Row rand_row() {
    Row r(kBytes);
    for (auto& b : r) b = uint8_t(rand_in(0, 255));
    return clip(r);
  }

  // One request (the controller idle: free at QUEUE). A partial write
  // leaves its first mask group out.
  void issue(bool rd, bool wr, uint32_t inst, uint32_t row, bool full = true) {
    Req q;
    q.rd = rd;
    q.wr = wr;
    q.inst = inst;
    q.rrow = q.wrow = row;
    q.data = rand_row();
    q.mask.assign(VM_MW, true);
    if (!full) q.mask[0] = false;
    force = &q;
    cycle();
    force = nullptr;
  }

  // Idle until the instance's read has answered, then `settle` cycles more.
  void await(uint32_t inst, uint64_t settle) {
    for (int i = 0; i < 100000 && pending[inst]; ++i) cycle(false, false);
    CHECK(!pending[inst], "the read of instance %u never answered", inst);
    for (uint64_t i = 0; i < settle; ++i) cycle(false, false);
  }

  bool quick(uint32_t inst) const { return rsp_lat[inst] <= kLocal + 1; }

  // The host writes a row of the image (the controller idle after a reset).
  void dma(uint32_t inst, uint32_t row) {
    const uint64_t addr = row_addr(inst, row);
    Row d = rand_row();
    for (int i = 0; i < kBytes; ++i) hbm.byte(addr + i) = d[i];
    ref.rows[addr] = d;
  }

  int run_directed(uint64_t seed) {
    static const Setup kDirected = {"directed", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 1.0, 0, 0, 0};
    const int before = failures;
    start(kDirected, seed);
    hbm.rvalid_p = 1.0;
    const bool rf = VM_RDW_READ_FIRST != 0;
    const uint32_t a = 0, r1 = 3 % kRows;
    int steps = 0;
    uint64_t ar, aw;
    // The line K2 leaves dirty, if any: K's fill must not need it.
    int64_t k2_line = -1;

    // K2: while other writes keep the fill from X, the row's own partial
    // write gets there first (it goes around): the fill must be dropped, or
    // a later read would hit its stale row.
    if (kInstances > 1 && VM_MW > 1) {
      const uint32_t b = 1, r9 = 9 % kRows;
      uint32_t rb = kRows;
      for (uint32_t r = 0; r < kRows && rb == kRows; ++r)
        if (line_of(b, r) != line_of(a, r9)) rb = r;
      if (rb != kRows) {
        issue(true, false, a, r9);
        bool wrote = false;
        int tail = 4;
        for (int i = 0; i < 2000 && tail > 0; ++i) {
          if (!pending[a] && !wrote) {
            issue(false, true, a, r9, false);
            wrote = true;
          } else {
            issue(false, true, b, rb);
            if (wrote) --tail;
          }
        }
        CHECK(wrote, "K2: the read never answered");
        for (int i = 0; i < 200; ++i) cycle(false, false);
        issue(true, false, a, r9);
        await(a, 0);
        k2_line = line_of(b, rb);
        ++steps;
      }
    }
    // K: a read that misses fills its line: the next read of the row is on
    // chip. A line with a dirty row takes no fill (its row stays).
    {
      uint32_t r6 = 5 % kRows;
      while (int64_t(line_of(a, r6)) == k2_line) r6 = (r6 + 1) % kRows;
      ar = hbm.n_ar;
      issue(true, false, a, r6);
      await(a, 20);
      CHECK(hbm.n_ar == ar + 1, "K: a read miss did not go to memory");
      issue(true, false, a, r6);
      await(a, 0);
      CHECK(hbm.n_ar == ar + 1 && quick(a), "K: a read miss did not fill its line (%lu cycles)",
            (unsigned long)rsp_lat[a]);
      uint32_t r7 = r6, r8 = r6;
      for (uint32_t r = 0; r < kRows && r8 == r6; ++r)
        if (r != r6 && line_of(a, r) == line_of(a, r6)) {
          if (r7 == r6)
            r7 = r;
          else
            r8 = r;
        }
      if (r8 != r6) {
        issue(false, true, a, r7);
        for (int i = 0; i < 20; ++i) cycle(false, false);
        ar = hbm.n_ar;
        issue(true, false, a, r8);
        await(a, 20);
        issue(true, false, a, r8);
        await(a, 20);
        CHECK(hbm.n_ar == ar + 2, "K: a fill replaced a dirty row (%lu reads)", (unsigned long)(hbm.n_ar - ar));
        issue(true, false, a, r7);
        await(a, 0);
        CHECK(hbm.n_ar == ar + 2 && quick(a), "K: the dirty row left the cache");
      }
      ++steps;
    }
    // L: a reset leaves the cache empty, so the host may change the image:
    // a read in memory when the reset comes fills nothing.
    {
      const uint32_t r10 = 11 % kRows;
      hbm.lat_lo = hbm.lat_hi = 300;
      issue(true, false, a, r10);
      for (int i = 0; i < 20; ++i) cycle(false, false);
      CHECK(pending[a] && !hbm.reads.empty(), "L: the read is not in memory");
      reset_dut(2);
      hbm.lat_lo = hbm.lat_hi = 50;
      while (draining) cycle(false, false);
      for (int i = 0; i < 20; ++i) cycle(false, false);
      dma(a, r10);
      issue(true, false, a, r10);
      await(a, 0);
      ++steps;
    }
    // A clean start for the rest.
    quiesce();
    reset_dut(2);
    while (draining) cycle(false, false);

    // A: a pair on a row not in the cache fills its line; a read-first
    // pair's read gets the old row from memory, a write-first one's the new.
    ar = hbm.n_ar;
    aw = hbm.n_aw;
    issue(true, true, a, r1);
    await(a, 100);
    CHECK(hbm.n_aw == aw, "A: the pair's write went to memory, not to its line");
    CHECK(hbm.n_ar == ar + (rf ? 1 : 0), "A: %lu reads of memory", (unsigned long)(hbm.n_ar - ar));
    if (!rf) CHECK(quick(a), "A: a write-first pair answered in %lu cycles", (unsigned long)rsp_lat[a]);
    ++steps;
    // B: the row is in the cache.
    ar = hbm.n_ar;
    issue(true, false, a, r1);
    await(a, 0);
    CHECK(hbm.n_ar == ar && quick(a), "B: the filled line missed (%lu cycles)", (unsigned long)rsp_lat[a]);
    ++steps;
    // C: a pair on the resident row: on chip, the old row (read-first) from
    // the line its write replaces.
    ar = hbm.n_ar;
    aw = hbm.n_aw;
    issue(true, true, a, r1);
    await(a, 100);
    CHECK(hbm.n_ar == ar && hbm.n_aw == aw && quick(a), "C: a pair on a resident row left the chip");
    ++steps;
    // D: and the new row is there.
    issue(true, false, a, r1);
    await(a, 0);
    CHECK(quick(a), "D: the row left the cache");
    ++steps;
    // E: a pair on another row of the line evicts the dirty row to memory;
    // both rows read back right.
    uint32_t r2 = r1;
    for (uint32_t r = 0; r < kRows && r2 == r1; ++r)
      if (r != r1 && line_of(a, r) == line_of(a, r1)) r2 = r;
    if (VM_CACHE_ROWS > 0 && r2 != r1) {
      ar = hbm.n_ar;
      aw = hbm.n_aw;
      issue(true, true, a, r2);
      await(a, 200);
      CHECK(hbm.n_ar == ar + (rf ? 1 : 0), "E: %lu reads of memory", (unsigned long)(hbm.n_ar - ar));
      CHECK(hbm.n_aw == aw + 1 && hbm.aw_log.back() == row_addr(a, r1), "E: the dirty row was not evicted");
      issue(true, false, a, r1);
      await(a, 0);
      ar = hbm.n_ar;
      issue(true, false, a, r2);
      await(a, 0);
      CHECK(hbm.n_ar == ar && quick(a), "E: the new row is not in the cache");
      ++steps;
    }
    // F: a partial write of a row not in the cache goes around it.
    const uint32_t r4 = (r1 + 7) % kRows;
    if (VM_MW > 1 && r4 != r1 && r4 != r2) {
      aw = hbm.n_aw;
      issue(false, true, a, r4, false);
      await(a, 200);
      CHECK(hbm.n_aw == aw + 1 && hbm.aw_log.back() == row_addr(a, r4), "F: a partial write filled a line");
      ++steps;
    }
    // G: instances sharing a line: instance p's pair pins it while its read
    // is in memory (sampling it as late as it can); other instances' whole-row
    // writes of rows on that line, the next cycle and later, go around instead
    // of evicting it. Once the read has its data, such a write fills the line.
    // H: a reset drops a pin whose read it drops.
    if (kPin && rf) {
      const uint32_t used = line_of(a, r1);
      uint32_t p = kInstances, r5 = 0;
      for (uint32_t i = 1; i < kInstances && p == kInstances; ++i)
        for (uint32_t r = 0; r < kRows; ++r)
          if (line_of(i, r) != used) {
            p = i;
            r5 = r;
            break;
          }
      std::vector<std::pair<uint32_t, uint32_t>> others;  // (instance, row) on p's line
      for (uint32_t i = 1; i < kInstances && others.size() < 2 && p != kInstances; ++i)
        for (uint32_t r = 0; r < kRows && others.size() < 2; ++r)
          if (i != p && line_of(i, r) == line_of(p, r5)) others.push_back({i, r});
      CHECK(others.size() == 2, "G: no two other instances' rows share a line");
      if (others.size() == 2) {
        const auto [q1, rq1] = others[0];
        const auto [q2, rq2] = others[1];
        hbm.lat_lo = hbm.lat_hi = 400;
        hbm.late_sample = true;
        aw = hbm.n_aw;
        issue(true, true, p, r5);
        issue(false, true, q1, rq1);
        for (int i = 0; i < 30; ++i) cycle(false, false);
        issue(false, true, q2, rq2);
        for (int i = 0; i < 300; ++i) cycle(false, false);
        CHECK(pending[p], "G: the pinning read answered early");
        CHECK(hbm.n_aw == aw + 2 && hbm.aw_log[aw] == row_addr(q1, rq1) && hbm.aw_log[aw + 1] == row_addr(q2, rq2),
              "G: a write of a pinned line did not go around (%lu AW)", (unsigned long)(hbm.n_aw - aw));
        await(p, 0);
        hbm.lat_lo = hbm.lat_hi = 50;
        hbm.late_sample = false;
        ar = hbm.n_ar;
        issue(true, false, p, r5);
        await(p, 0);
        CHECK(hbm.n_ar == ar && quick(p), "G: the pinned row left the cache");
        issue(true, false, q1, rq1);
        await(q1, 0);
        CHECK(hbm.n_ar == ar + 1, "G: the row written around is in the cache");
        aw = hbm.n_aw;
        issue(false, true, q1, rq1);
        for (int i = 0; i < 200; ++i) cycle(false, false);
        CHECK(hbm.n_aw == aw + 1 && hbm.aw_log.back() == row_addr(p, r5),
              "G: the line stayed pinned after its read");
        ++steps;

        // H: with AR held back, p's pair's read waits behind another read
        // when the reset drops it; its pin must go too.
        hbm.ready_p = 0;
        ar = hbm.n_ar;
        issue(true, false, q2, rq2);
        issue(true, true, p, r5);
        for (int i = 0; i < 20; ++i) cycle(false, false);
        CHECK(pending[p] && hbm.n_ar == ar, "H: the reads did not wait for AR");
        reset_dut(2);
        hbm.ready_p = 1.0;
        while (draining) cycle(false, false);
        ar = hbm.n_ar;
        aw = hbm.n_aw;
        issue(false, true, q2, rq2);
        for (int i = 0; i < 200; ++i) cycle(false, false);
        issue(true, false, q2, rq2);
        await(q2, 0);
        CHECK(hbm.n_aw == aw && hbm.n_ar == ar && quick(q2), "H: a pin outlived the reset");
        ++steps;

        // I: a pair whose read is answered from the buffer (its row evicted,
        // the write-back not yet out: the pair right behind the evicting write,
        // as a slot is freed by its last W beat) releases its pin there.
        hbm.lat_lo = hbm.lat_hi = 400;
        issue(false, true, p, r5);
        for (int i = 0; i < 10; ++i) cycle(false, false);
        issue(false, true, q1, rq1);
        issue(true, true, p, r5);
        await(p, 0);
        // Without forwarding (FORWARD 0) the read waits for the write-back's
        // B and reads memory; the pin is released when it has its data.
        if (VM_FORWARD)
          CHECK(quick(p), "I: the pair's read was not answered from the buffer (%lu cycles)",
                (unsigned long)rsp_lat[p]);
        else
          CHECK(!quick(p), "I: the pair's read was answered from the buffer without FORWARD (%lu cycles)",
                (unsigned long)rsp_lat[p]);
        const size_t from = hbm.aw_log.size();
        issue(false, true, q2, rq2);
        for (int i = 0; i < 1500; ++i) cycle(false, false);
        bool evicted = false, around = false;
        for (size_t i = from; i < hbm.aw_log.size(); ++i) {
          evicted |= hbm.aw_log[i] == row_addr(p, r5);
          around |= hbm.aw_log[i] == row_addr(q2, rq2);
        }
        CHECK(evicted && !around, "I: the line stayed pinned after its read was answered from the buffer (evicted %d around %d)", int(evicted), int(around));
        hbm.lat_lo = hbm.lat_hi = 50;
        ++steps;

        // J: more pairs in memory at once than pins (8): the one past them goes
        // around, and the first pin holds.
        constexpr uint32_t kPins = 8;
        std::vector<std::pair<uint32_t, uint32_t>> pairs;  // (instance, row), distinct lines
        std::set<uint32_t> lines, insts;
        for (uint32_t i = 1; i < kInstances && pairs.size() < kPins + 1; ++i)
          for (uint32_t r = 0; r < kRows; ++r)
            if (!lines.count(line_of(i, r))) {
              pairs.push_back({i, r});
              lines.insert(line_of(i, r));
              insts.insert(i);
              break;
            }
        uint32_t x = kInstances, rx = 0;
        for (uint32_t i = 1; i < kInstances && x == kInstances && pairs.size() == kPins + 1; ++i)
          for (uint32_t r = 0; r < kRows && !insts.count(i); ++r)
            if (line_of(i, r) == line_of(pairs[0].first, pairs[0].second)) {
              x = i;
              rx = r;
              break;
            }
        if (x != kInstances) {
          hbm.lat_lo = hbm.lat_hi = 600;
          hbm.late_sample = true;
          for (auto [i, r] : pairs) issue(true, true, i, r);
          for (int c = 0; c < 20; ++c) cycle(false, false);
          issue(false, true, x, rx);
          for (auto [i, r] : pairs) await(i, 0);
          hbm.lat_lo = hbm.lat_hi = 50;
          hbm.late_sample = false;
          // Every row reads back its pair's write (the one past the pins'
          // read must not have filled its line with the old row).
          for (auto [i, r] : pairs) {
            issue(true, false, i, r);
            await(i, 0);
          }
          ++steps;
        }
      }
    }
    finish();
    printf("directed %d steps (read-first %d, pins %d): %s\n", steps, int(rf), int(kPin),
           failures == before ? "ok" : "FAILED");
    delete dut;
    dut = nullptr;
    return failures - before;
  }

  // ---- Directed: known-zero rows ----
  // AW entries since `from` that wrote row (i, r), and whether each was whole.
  int aws_to(size_t from, uint32_t i, uint32_t r, bool* whole) {
    int n = 0;
    *whole = true;
    for (size_t k = from; k < hbm.aw_log.size(); ++k) {
      if (hbm.aw_log[k] != row_addr(i, r)) continue;
      ++n;
      *whole &= k < hbm.whole_log.size() && hbm.whole_log[k];
    }
    return n;
  }

  int run_kz_directed(uint64_t seed) {
    static const Setup kZ = {"zdirected", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 1.0, 0, 1, 0};
    const int before = failures;
    start(kZ, seed);  // the host clears the image and raises zero
    hbm.rvalid_p = 1.0;
    const uint32_t a = 0, top = kInstances - 1, rtop = kRows - 1;
    std::set<std::pair<uint32_t, uint32_t>> used;
    // A row of instance i nobody has touched (and, with `line`, on that line).
    auto fresh = [&](uint32_t i, int64_t line = -1) {
      for (uint32_t r = 1; r < kRows; ++r)
        if (!used.count({i, r}) && (line < 0 || line_of(i, r) == uint32_t(line))) {
          used.insert({i, r});
          return r;
        }
      CHECK(false, "no fresh row of instance %u", i);
      return 0u;
    };
    int steps = 0;
    uint64_t ar;
    size_t from;
    bool whole;

    // Z1: rows nobody wrote since the reset read as zeros, on chip, at the
    // first and last instance and row (the bits' first and last banks and
    // words).
    ar = hbm.n_ar;
    const std::pair<uint32_t, uint32_t> z1[] = {{a, 0}, {top, rtop}, {top, 0}, {a, rtop}};
    for (auto [i, r] : z1) {
      used.insert({i, r});
      issue(true, false, i, r);
      await(i, 0);
      CHECK(quick(i), "Z1: known-zero row %u of instance %u answered in %lu cycles", r, i,
            (unsigned long)rsp_lat[i]);
    }
    CHECK(hbm.n_ar == ar, "Z1: %lu reads of memory for known-zero rows", (unsigned long)(hbm.n_ar - ar));
    ++steps;

    // Z2: a whole write and a partial one take their rows out; the partial
    // one is made whole (zeros where its mask is off): it fills a line, or
    // goes to memory whole. Both read back their data.
    const uint32_t rw = fresh(a), rp = fresh(a);
    from = hbm.aw_log.size();
    issue(false, true, a, rw);
    issue(false, true, a, rp, false);
    for (int i = 0; i < 300; ++i) cycle(false, false);
    const int to_rp = aws_to(from, a, rp, &whole);
    if (VM_CACHE_ROWS == 0) {
      bool whole_rw;
      CHECK(aws_to(from, a, rw, &whole_rw) == 1 && to_rp == 1, "Z2: the writes did not reach memory");
      CHECK(whole, "Z2: a partial write of a known-zero row went to memory partial");
    } else {
      CHECK(to_rp == 0, "Z2: a partial write of a known-zero row did not fill its line");
    }
    for (uint32_t r : {rw, rp}) {
      issue(true, false, a, r);
      await(a, 0);
    }
    ++steps;

    // Z3: pairs on known-zero rows, whole and partial: the read is on chip
    // (read-first: zeros; write-first: the write over zeros); the row then
    // reads back the write.
    for (bool full : {true, false}) {
      const uint32_t r = fresh(a);
      ar = hbm.n_ar;
      issue(true, true, a, r, full);
      await(a, 100);
      CHECK(quick(a) && hbm.n_ar == ar, "Z3: a %s pair on a known-zero row left the chip (%lu cycles)",
            full ? "whole" : "partial", (unsigned long)rsp_lat[a]);
      issue(true, false, a, r);
      await(a, 0);
    }
    ++steps;

    // Z4: a row written, then evicted from its line, reads back its data
    // (from the buffer or memory, never as zero).
    uint32_t r5 = 0, r6 = 0;
    for (uint32_t r = 1; r < kRows && r6 == 0; ++r)
      for (uint32_t t = r + 1; t < kRows && VM_CACHE_ROWS > 0 && !used.count({a, r}); ++t)
        if (!used.count({a, t}) && line_of(a, t) == line_of(a, r)) {
          r5 = r, r6 = t;
          break;
        }
    if (r6 != 0) {
      used.insert({a, r5});
      used.insert({a, r6});
      issue(false, true, a, r5);
      for (int i = 0; i < 20; ++i) cycle(false, false);
      from = hbm.aw_log.size();
      issue(false, true, a, r6);
      for (int i = 0; i < 300; ++i) cycle(false, false);
      CHECK(aws_to(from, a, r5, &whole) == 1, "Z4: the written row was not evicted");
      ar = hbm.n_ar;
      issue(true, false, a, r5);
      await(a, 0);
      CHECK(hbm.n_ar == ar + 1, "Z4: the evicted row was not read from memory");
      ++steps;
    }

    // Z5: zero raised with a read in memory: that read answers from memory,
    // a read after it on chip. Dropped the cycle after a read: that read
    // still answers on chip, the next goes to memory.
    {
      hbm.lat_lo = hbm.lat_hi = 300;
      zero = false;
      cycle(false, false);
      const uint32_t r7 = fresh(a), r8 = fresh(top);
      ar = hbm.n_ar;
      issue(true, false, a, r7);
      for (int i = 0; i < 20; ++i) cycle(false, false);
      CHECK(pending[a] && hbm.n_ar == ar + 1, "Z5: with zero low the read did not go to memory");
      zero = true;
      // One read per instance outstanding: with one instance, after it.
      if (top == a) await(a, 0);
      issue(true, false, top, r8);
      await(top, 0);
      CHECK(quick(top) && hbm.n_ar == ar + 1, "Z5: zero raised, a known-zero read left the chip");
      if (top != a) CHECK(pending[a], "Z5: the read in memory answered early");
      await(a, 0);
      const uint32_t r9 = fresh(top), r10 = fresh(top);
      ar = hbm.n_ar;
      issue(true, false, top, r9);
      for (int i = 0; i < kIntake; ++i) cycle(false, false);
      zero = false;
      await(top, 0);
      CHECK(quick(top) && hbm.n_ar == ar, "Z5: a read taken with zero high went to memory");
      issue(true, false, top, r10);
      await(top, 0);
      CHECK(hbm.n_ar == ar + 1, "Z5: a read taken with zero low did not go to memory");
      hbm.lat_lo = hbm.lat_hi = 50;
      ++steps;
    }

    // Z6: with zero low, a row answered as zero earlier is read from memory
    // (a zero answer fills no line), and the host may change a row nobody
    // touched: it reads back the host's data.
    {
      for (int i = 0; i < 50; ++i) cycle(false, false);
      ar = hbm.n_ar;
      issue(true, false, a, 0);
      await(a, 0);
      CHECK(hbm.n_ar == ar + 1, "Z6: a known-zero answer filled a line");
      const uint32_t r11 = fresh(a);
      dma(a, r11);
      issue(true, false, a, r11);
      await(a, 0);
      CHECK(hbm.n_ar == ar + 2, "Z6: the host's row was not read from memory");
      ++steps;
    }

    // Z7: a reset forgets every write: with zero low the rows written before
    // it read back from memory; once the host clears the image and raises
    // zero, the same rows are known zero again, the last row of the last
    // instance (the clear's last word) among them.
    {
      issue(false, true, top, rtop);
      for (int i = 0; i < 50; ++i) cycle(false, false);
      reset_dut(2);
      while (draining) cycle(false, false);
      ar = hbm.n_ar;
      issue(true, false, a, rw);
      await(a, 0);
      CHECK(hbm.n_ar == ar + 1, "Z7: zero low after the reset, yet no read of memory");
      host_zero();
      used.clear();
      for (auto [i, r] : {std::pair<uint32_t, uint32_t>{a, rw}, {a, rp}, {top, rtop}}) {
        ar = hbm.n_ar;
        issue(true, false, i, r);
        await(i, 0);
        CHECK(quick(i) && hbm.n_ar == ar, "Z7: row %u of instance %u, written before the reset, is not known zero",
              r, i);
      }
      ++steps;
    }

    // Z9: a known-zero read taken the cycle before a one-cycle reset is not
    // answered (the TB's check: nothing during a reset or its drain).
    {
      const uint32_t r12 = fresh(a);
      issue(true, false, a, r12);
      reset_dut(1);
      while (draining) cycle(false, false);
      for (int i = 0; i < 20; ++i) cycle(false, false);
      host_zero();
      ++steps;
    }

    // Z8: where instances share lines, a write of a known-zero row whose line
    // a pair has pinned goes around the cache, whole, and takes its row out.
    if (kPin && VM_RDW_READ_FIRST != 0) {
      uint32_t p = kInstances, rp1 = 0, q = kInstances, rq = 0;
      for (uint32_t i = 1; i < kInstances && q == kInstances; ++i)
        for (uint32_t r = 1; r < kRows && q == kInstances; ++r) {
          if (used.count({i, r})) continue;
          for (uint32_t j = i + 1; j < kInstances && q == kInstances; ++j)
            for (uint32_t t = 1; t < kRows; ++t)
              if (!used.count({j, t}) && line_of(j, t) == line_of(i, r)) {
                p = i, rp1 = r, q = j, rq = t;
                break;
              }
        }
      CHECK(q != kInstances, "Z8: no two instances' rows share a line");
      if (q != kInstances) {
        used.insert({p, rp1});
        used.insert({q, rq});
        hbm.lat_lo = hbm.lat_hi = 400;
        hbm.late_sample = true;
        // The pair's read must go to memory: zero low for its cycle.
        zero = false;
        issue(true, true, p, rp1);
        for (int i = 0; i < kIntake; ++i) cycle(false, false);
        zero = true;
        for (int i = 0; i < 10; ++i) cycle(false, false);
        CHECK(pending[p], "Z8: the pinning read answered early");
        from = hbm.aw_log.size();
        issue(false, true, q, rq, false);
        for (int i = 0; i < 300; ++i) cycle(false, false);
        CHECK(pending[p], "Z8: the pinning read answered early");
        CHECK(aws_to(from, q, rq, &whole) == 1 && whole, "Z8: the write of the pinned line did not go around whole");
        await(p, 0);
        hbm.lat_lo = hbm.lat_hi = 50;
        hbm.late_sample = false;
        issue(true, false, q, rq);
        await(q, 0);
        ++steps;
      }
    }
    finish();
    printf("zdirected %d steps (cache %d, read-first %d, pins %d, %lu drain cycles): %s\n", steps,
           VM_CACHE_ROWS, VM_RDW_READ_FIRST, int(kPin), (unsigned long)drain_cycles,
           failures == before ? "ok" : "FAILED");
    delete dut;
    dut = nullptr;
    return failures - before;
  }

  // ---- Directed: region moves ----
  uint64_t old_addr(uint64_t base, uint32_t inst, uint32_t row) const {
    return base + ((uint64_t(row) * kInstances + inst) << kRowLog2);
  }

  // ---- Directed: known-zero blocks (KZ_SHIFT > 0) ----
  // A bit stands for 2^KZ_SHIFT consecutive image rows (row * INSTANCES +
  // inst). B1: a row of an untouched block reads zeros on chip. B2: once a
  // row of a block is written, another row of it goes to memory (the host's
  // zeros, the answer checked against the reference). B3: a partial write of
  // a third row there is not made whole (goes to memory partial, or merges
  // into nothing: around the cache). B4: a row of another untouched block is
  // still known zero.
  int run_kzb_directed(uint64_t seed) {
    static const Setup kZ = {"zbdirected", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 1.0, 0, 1, 0};
    const int before = failures;
    start(kZ, seed);
    hbm.rvalid_p = 1.0;
    auto block = [](uint32_t i, uint32_t r) { return (uint64_t(r) * kInstances + i) >> VM_KZ_SHIFT; };
    // Rows of block `b`, other than those given.
    auto in_block = [&](uint64_t b, std::set<std::pair<uint32_t, uint32_t>> no) {
      for (uint32_t r = 0; r < kRows; ++r)
        for (uint32_t i = 0; i < kInstances; ++i)
          if (block(i, r) == b && !no.count({i, r})) return std::make_pair(i, r);
      return std::make_pair(kInstances, kRows);
    };
    int steps = 0;
    uint64_t ar;
    const auto p0 = in_block(1, {});
    const auto p1 = in_block(1, {p0});
    const auto p2 = in_block(1, {p0, p1});
    // B1
    ar = hbm.n_ar;
    issue(true, false, p0.first, p0.second);
    await(p0.first, 0);
    CHECK(quick(p0.first) && hbm.n_ar == ar, "B1: a row of an untouched block left the chip");
    ++steps;
    // B2
    issue(false, true, p0.first, p0.second);
    for (int i = 0; i < 300; ++i) cycle(false, false);
    if (p1.first < kInstances) {
      ar = hbm.n_ar;
      issue(true, false, p1.first, p1.second);
      await(p1.first, 0);
      CHECK(hbm.n_ar == ar + 1, "B2: a row of a written block was answered as known zero");
      ++steps;
    }
    // B3
    if (p2.first < kInstances && VM_MW > 1) {
      const size_t from = hbm.aw_log.size();
      bool whole = true;
      issue(false, true, p2.first, p2.second, false);
      for (int i = 0; i < 300; ++i) cycle(false, false);
      CHECK(aws_to(from, p2.first, p2.second, &whole) == 1 && !whole,
            "B3: a partial write in a written block was made whole");
      issue(true, false, p2.first, p2.second);
      await(p2.first, 0);
      ++steps;
    }
    // B4
    const auto p3 = in_block(2, {});
    if (p3.first < kInstances) {
      ar = hbm.n_ar;
      issue(true, false, p3.first, p3.second);
      await(p3.first, 0);
      CHECK(quick(p3.first) && hbm.n_ar == ar, "B4: a row of another untouched block left the chip");
      ++steps;
    }
    finish();
    printf("zbdirected %d steps (block of %d rows): %s\n", steps, 1 << VM_KZ_SHIFT, failures == before ? "ok" : "FAILED");
    delete dut;
    dut = nullptr;
    return failures - before;
  }

  // ---- Directed: write throughput ----
  // Writes that go to memory (no host cache), back to back, against a memory
  // of fixed latency L that is always ready: a write holds its buffer slot
  // only until its last W beat, so up to WRITES_OUT wait for B at once and
  // the controller takes min(1, WRITES_OUT / L) writes a cycle (a slot until B
  // would allow QUEUE / L). Then reads of every row just written, while their
  // writes still wait for B, each answered with its write's data (the
  // reference): none reaches AR before its write's B.
  int run_wrate_directed(uint64_t seed) {
    static const Setup kW = {"wrate", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 400, 400, 1.0, 0, 0, 0};
    const int before = failures;
    start(kW, seed);
    hbm.rvalid_p = 1.0;
    const uint64_t span = 8000, w0 = n_writes, c0 = now;
    uint32_t k = 0;
    std::vector<std::pair<uint32_t, uint32_t>> last;
    while (now - c0 < span) {
      if (lead_ok(1)) {
        const uint32_t i = k % kInstances, r = (k / kInstances) % kRows;
        issue(false, true, i, r);
        last.push_back({i, r});
        if (last.size() > 64) last.erase(last.begin());
        ++k;
      } else {
        cycle(false, false);
      }
    }
    const double rate = double(n_writes - w0) / double(span);
    const double bound = std::min(1.0, double(VM_WRITES_OUT) / 400.0);
    // A queue near the simulator's room (10) admits a write only when nearly
    // empty: the rate is then the room's, not the memory's.
    if (VM_QUEUE >= 32)
      CHECK(rate >= 0.8 * bound, "wrate: %.3f writes a cycle, expected about %.3f", rate, bound);
    // Reads of the last rows written, in flight on AXI.
    uint64_t raw = 0;
    for (auto [i, r] : last) {
      while (!lead_ok(1) || pending[i]) cycle(false, false);
      if (wr_inflight.count(row_addr(i, r))) ++raw;
      issue(true, false, i, r);
    }
    // (Few writes on AXI keep the queue full of writes: reads then wait
    // until their B's are back.)
    if (VM_QUEUE >= 32 && VM_WRITES_OUT >= 32) CHECK(raw > 0, "wrate: no read met its write in flight");
    finish();
    printf("wrate %lu writes in %lu cycles: %.3f a cycle (WRITES_OUT %d, QUEUE %d, latency 400: bound %.3f); "
           "%lu reads of rows whose writes wait for B: %s\n",
           (unsigned long)(n_writes - w0), (unsigned long)span, rate, VM_WRITES_OUT, VM_QUEUE, bound,
           (unsigned long)raw, failures == before ? "ok" : "FAILED");
    delete dut;
    dut = nullptr;
    return failures - before;
  }

  int run_move_directed(uint64_t seed) {
    static const Setup kM = {"mdirected", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 50, 50, 1.0, 0, 0, 0};
    const int before = failures;
    start(kM, seed);
    hbm.rvalid_p = 1.0;
    int steps = 0;
    uint64_t ar;

    // M1: rows written whole sit dirty in their lines; a move writes each back
    // to the old region, and nothing to the new one. There they read the new
    // region's rows, from memory: no line of the old region answers.
    if (VM_CACHE_ROWS > 0) {
      const uint64_t b0 = g_base;
      std::vector<std::pair<uint32_t, uint32_t>> dirty;
      std::set<uint32_t> lines;
      for (uint32_t i = 0; i < kInstances && dirty.size() < 4; ++i)
        for (uint32_t r = 1; r < kRows && dirty.size() < 4; ++r)
          if (!lines.count(line_of(i, r))) {
            dirty.push_back({i, r});
            lines.insert(line_of(i, r));
          }
      const uint64_t aw = hbm.n_aw;
      for (auto [i, r] : dirty) issue(false, true, i, r);
      for (int c = 0; c < 50; ++c) cycle(false, false);
      CHECK(hbm.n_aw == aw, "M1: whole-row writes went to memory, not to their lines");
      const size_t from = hbm.aw_log.size();
      begin_move(kBase + kStride, false);
      finish_move(false, false);
      std::set<uint64_t> want, got;
      for (auto [i, r] : dirty) want.insert(old_addr(b0, i, r));
      for (size_t k = from; k < hbm.aw_log.size(); ++k) got.insert(hbm.aw_log[k]);
      CHECK(got == want && hbm.aw_log.size() - from == dirty.size(),
            "M1: the move wrote %zu rows back, want the %zu dirty ones at the old base", hbm.aw_log.size() - from,
            dirty.size());
      ar = hbm.n_ar;
      for (auto [i, r] : dirty) {
        issue(true, false, i, r);
        await(i, 0);
        CHECK(!quick(i), "M1: row %u of instance %u answered on chip after the move", r, i);
      }
      CHECK(hbm.n_ar == ar + dirty.size(), "M1: %lu reads of the new region, want %zu", (unsigned long)(hbm.n_ar - ar),
            dirty.size());
      ++steps;
    }

    // M2: reads in memory when the move comes, and requests within the lead
    // after it (a read, and a write of another instance), are served from the
    // old region and answered before the move ends; free is 0 all along. The
    // base's high word comes during the move. The written row is the old
    // region's: the new one's reads its own.
    {
      const uint32_t n = std::min<uint32_t>(kInstances, 3);
      const uint32_t rr = 20 % kRows, rw2 = 21 % kRows;
      hbm.lat_lo = hbm.lat_hi = 300;
      issue(true, false, 0, rr);
      for (int c = 0; c < 10; ++c) cycle(false, false);
      CHECK(pending[0] && !hbm.reads.empty(), "M2: the read is not in memory");
      const uint64_t b0 = g_base, taken = lead_taken, mc = move_cycles;
      begin_move(kBase + 2 * kStride, true);
      if (n > 1)
        issue(true, false, 1, rr);  // the move's own cycle
      else
        cycle(false, false);
      if (n > 2) issue(false, true, 2, rw2);
      CHECK(lead_taken == taken + (n > 1) + (n > 2), "M2: the requests within the lead were not taken");
      finish_move(false, false, false, 5);
      hbm.lat_lo = hbm.lat_hi = 50;
      CHECK(move_cycles - mc > uint64_t(250), "M2: the move ended before its reads (%lu cycles)",
            (unsigned long)(move_cycles - mc));
      if (n > 2) {
        CHECK(hbm.byte(old_addr(b0, 2, rw2)) == ref.get(old_addr(b0, 2, rw2))[0],
              "M2: the write within the lead did not reach the old region");
        ar = hbm.n_ar;
        issue(true, false, 2, rw2);
        await(2, 0);
        CHECK(hbm.n_ar == ar + 1, "M2: the new region's row did not come from memory");
      }
      ++steps;
    }

    // M3: with known-zero rows, the rows written before a move to the same
    // region (cleared by the host, `zero` raised) are known zero again: answered
    // on chip with no read of memory; and so are those of a new region.
    if (VM_KNOWN_ZERO) {
      host_zero();
      const uint32_t a = 0, r1 = 30 % kRows, r2 = 31 % kRows, r3 = 32 % kRows;
      issue(false, true, a, r1);
      issue(false, true, a, r2, false);
      issue(true, false, a, r3);
      await(a, 0);
      CHECK(quick(a), "M3: a known-zero row before the move left the chip");
      for (int c = 0; c < 50; ++c) cycle(false, false);
      for (uint64_t to : {g_base, kBase + 3 * kStride}) {
        begin_move(to, to != g_base);
        finish_move(true, true);
        ar = hbm.n_ar;
        for (uint32_t r : {r1, r2, r3}) {
          issue(true, false, a, r);
          await(a, 0);
          CHECK(quick(a), "M3: row %u, written before the move, not known zero after it (%lu cycles)", r,
                (unsigned long)rsp_lat[a]);
        }
        CHECK(hbm.n_ar == ar, "M3: %lu reads of memory for known-zero rows", (unsigned long)(hbm.n_ar - ar));
        // Written again (partially: made whole over zeros), it reads back.
        issue(false, true, a, r1, false);
        issue(false, true, a, r2);
        for (int c = 0; c < 50; ++c) cycle(false, false);
        for (uint32_t r : {r1, r2}) {
          issue(true, false, a, r);
          await(a, 0);
        }
      }
      ++steps;
    }

    // M4: a move to the same region, not cleared: the host may then change
    // rows the cache held (one written, dirty; one read, clean): they read the
    // host's data, from memory.
    if (VM_CACHE_ROWS > 0) {
      // Row rw of instance a written; row rr of instance b, on another line, read.
      const uint32_t a = 0, rw = 40 % kRows;
      uint32_t b = kInstances, rr = 0;
      for (uint32_t i = 0; i < kInstances && b == kInstances; ++i)
        for (uint32_t r = 41 % kRows; r < kRows; ++r)
          if (line_of(i, r) != line_of(a, rw)) {
            b = i, rr = r;
            break;
          }
      CHECK(b != kInstances, "M4: no two rows on different lines");
      // `zero` low: the read row fills its line.
      zero = false;
      cycle(false, false);
      issue(false, true, a, rw);
      issue(true, false, b, rr);
      await(b, 20);
      issue(true, false, b, rr);
      await(b, 0);
      CHECK(quick(b), "M4: the read row was not in the cache");
      issue(true, false, a, rw);
      await(a, 0);
      CHECK(quick(a), "M4: the written row was not in the cache");
      begin_move(g_base, false);
      finish_move(false, false);
      dma(a, rw);
      dma(b, rr);
      ar = hbm.n_ar;
      for (auto [i, r] : {std::pair<uint32_t, uint32_t>{a, rw}, {b, rr}}) {
        issue(true, false, i, r);
        await(i, 0);
        CHECK(!quick(i), "M4: row %u of instance %u answered from the cache after the move", r, i);
      }
      CHECK(hbm.n_ar == ar + 2, "M4: the host's rows were not read from memory");
      ++steps;
    }

    // M5: a reset during a move: its drain finishes the old region (the reads
    // dropped), and the new region is used after it.
    {
      hbm.lat_lo = hbm.lat_hi = 300;
      issue(true, false, 0, 50 % kRows);
      for (int c = 0; c < 10; ++c) cycle(false, false);
      begin_move(kBase + 4 * kStride, false);
      for (int c = 0; c < 5; ++c) cycle(false, false);
      reset_dut(2);
      while (draining) cycle(false, false);
      hbm.lat_lo = hbm.lat_hi = 50;
      CHECK(g_base == kBase + 4 * kStride && !moving_tb, "M5: the reset did not end the move");
      base_in = g_base;
      issue(false, true, 0, 51 % kRows);
      issue(true, false, 0, 51 % kRows);
      await(0, 20);
      ++steps;
    }
    // M6: a move with nothing in flight, and a write taken at the end of the
    // simulator's lead (the third cycle after it): the flush waits for it, and
    // it reaches the old region.
    {
      const uint64_t b0 = g_base;
      const uint32_t r = 60 % kRows;
      for (int c = 0; c < 50; ++c) cycle(false, false);
      begin_move(kBase + 5 * kStride, false);
      cycle(false, false);
      cycle(false, false);
      issue(false, true, 0, r);
      finish_move(false, false);
      const Row want = ref.get(old_addr(b0, 0, r));
      Row got(kBytes);
      for (int i = 0; i < kBytes; ++i) got[i] = hbm.byte(old_addr(b0, 0, r) + i);
      CHECK(clip(got) == want, "M6: the write taken in the lead did not reach the old region");
      ++steps;
    }
    finish();
    printf("mdirected %d steps (cache %d, known zero %d, %lu moves, %lu cycles in them): %s\n", steps, VM_CACHE_ROWS,
           VM_KNOWN_ZERO, (unsigned long)n_moves, (unsigned long)move_cycles, failures == before ? "ok" : "FAILED");
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
  printf("foldsim_vmem: IW %d AW %d DW %d MW %d RDW_READ_FIRST %d QUEUE %d AXI_ID_W %d OUTSTANDING %d INSTANCES %u "
         "CACHE_ROWS %d KNOWN_ZERO %d KZ_SHIFT %d ROW_BYTES %d WRITES_OUT %d\n",
         VM_IW, VM_AW, VM_DW, VM_MW, VM_RDW_READ_FIRST, VM_QUEUE, VM_AXI_ID_W, VM_OUTSTANDING, kInstances,
         VM_CACHE_ROWS, VM_KNOWN_ZERO, VM_KZ_SHIFT, 1 << kRowLog2, VM_WRITES_OUT);
  Tb tb;
  if (VM_CACHE_ROWS > 0 && (!only || !std::strcmp(only, "directed"))) tb.run_directed(1);
  if (VM_KNOWN_ZERO && VM_KZ_SHIFT == 0 && (!only || !std::strcmp(only, "zdirected"))) tb.run_kz_directed(1);
  if (VM_KNOWN_ZERO && VM_KZ_SHIFT > 0 && (!only || !std::strcmp(only, "zbdirected"))) tb.run_kzb_directed(1);
  if (!only || !std::strcmp(only, "mdirected")) tb.run_move_directed(1);
  if (VM_CACHE_ROWS == 0 && (!only || !std::strcmp(only, "wrate"))) tb.run_wrate_directed(1);
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
