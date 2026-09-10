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

#include "firesim_sim.h"

#include "bridges/clock.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <gmp.h>

#include "bridges/loadmem.h"
#include "bridges/peek_poke.h"
#include "core/simif.h"
#include "core/simulation.h"
#include "core/widget_registry.h"

// Provided by the simulator's entry translation unit.
extern int entry(int argc, char **argv);

namespace {

bool trace_enabled() {
  static const bool on = (getenv("VORTEX_FIRESIM_TRACE") != nullptr);
  return on;
}

// Timestamps are the point: they say whether time is spent advancing the target
// or in the caller between requests, which no amount of ordering information can.
inline double trace_now() {
  static const auto t0 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

#define FSIM_TRACE(...)                                                        \
  do {                                                                         \
    if (trace_enabled()) {                                                     \
      fprintf(stderr, "[firesim %8.3f] ", trace_now());                        \
      fprintf(stderr, __VA_ARGS__);                                            \
      fputc('\n', stderr);                                                     \
      fflush(stderr);                                                          \
    }                                                                          \
  } while (0)

// Formats a request label. Cheap enough at this rate to build unconditionally,
// and building it only when tracing would make the traced and untraced runs
// differ in what they serialize.
std::string label(const char *fmt, ...) {
  char buf[128];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  return std::string(buf);
}

// Requests are closures executed on the simulator thread. Handing over a
// closure rather than a tagged struct keeps the queue from having to grow a
// case for every new operation.
using request_t = std::function<void()>;

// Each request carries a name so a trace says which operation the target is in.
// Without it every entry reads the same and a stall cannot be attributed.
struct labelled_request_t {
  std::string label;
  request_t fn;
};

// Target DRAM is reached through the LoadMem widget, which moves one host
// memory beat at a time. Unaligned or partial edges are handled with a
// read-modify-write so callers can use arbitrary byte ranges.
//
// The widget's chunk is counted in 32-bit words, not bytes: read_mem and
// write_mem push exactly that many words through the data register. Treating
// it as a byte count uploads a quarter of each beat and advances the address
// by a quarter of a beat, which stores two live bytes per eight and drops
// every write that lands off a beat boundary.
void access_mem(uint64_t addr, uint64_t size, void *value, bool is_write);

struct channel_t {
  std::mutex mtx;
  std::condition_variable to_sim;
  std::condition_variable to_host;
  std::queue<labelled_request_t> pending;
  bool request_done = false;
  bool ready = false;
  bool shutdown = false;
};

class vortex_simulation_t;

// The simulator's own globals are never torn down, so these outlive the
// wrapper and are cleared only when the run loop exits.
channel_t *g_channel = nullptr;
vortex_simulation_t *g_sim = nullptr;

// A simulation whose run loop services requests instead of executing a fixed
// script. Everything the target does happens on this thread: the bridges are
// not thread-safe and must never be touched by the caller.
class vortex_simulation_t final : public simulation_t {
public:
  vortex_simulation_t(widget_registry_t &registry,
                      const std::vector<std::string> &args)
      : simulation_t(registry, args),
        peek_poke(registry.get_widget<peek_poke_t>()),
        loadmem(registry.get_widget<loadmem_t>()),
        clock(registry.get_widget<clockmodule_t>()) {}

  int simulation_run() override {
    g_sim = this;
    FSIM_TRACE("simulator thread entered the request loop");
    {
      std::lock_guard<std::mutex> lock(g_channel->mtx);
      g_channel->ready = true;
    }
    g_channel->to_host.notify_all();

    for (;;) {
      request_t req;
      std::string req_label;
      {
        std::unique_lock<std::mutex> lock(g_channel->mtx);
        g_channel->to_sim.wait(
            lock, [] { return g_channel->shutdown || !g_channel->pending.empty(); });
        if (g_channel->shutdown && g_channel->pending.empty()) {
          g_sim = nullptr;
          return 0;
        }
        req = std::move(g_channel->pending.front().fn);
        req_label = std::move(g_channel->pending.front().label);
        g_channel->pending.pop();
      }

      FSIM_TRACE("servicing %s", req_label.c_str());
      req();
      FSIM_TRACE("%s complete", req_label.c_str());

      {
        std::lock_guard<std::mutex> lock(g_channel->mtx);
        g_channel->request_done = true;
      }
      g_channel->to_host.notify_all();
    }
  }

  void step(uint32_t n) {
    peek_poke.step(n, true);
  }

  peek_poke_t &peek_poke;
  loadmem_t &loadmem;
  // True target cycles elapsed. step() requests cycles; this reports how many
  // the target actually advanced, which is the only way to tell a design that
  // is idle apart from a simulation that has stopped advancing it.
  clockmodule_t &clock;
};

} // namespace

// Replaces the TEST_MAIN-generated definition: the simulator is a library here,
// not a self-contained test.
std::unique_ptr<simulation_t>
create_simulation(simif_t &simif,
                  widget_registry_t &registry,
                  const std::vector<std::string> &args) {
  (void)simif;
  return std::make_unique<vortex_simulation_t>(registry, args);
}

namespace {

// Records each host-to-device upload so a real launch can be replayed in the
// harness under metasimulation, where the core's internals are visible. Without
// this the harness loads only the program image, so its kernel runs on zeros and
// exits in a few hundred cycles -- it reproduces the launch but not the
// workload, and so cannot reproduce a stall that needs real data to reach.
void dump_upload(uint64_t addr, uint64_t size, const void *value) {
  const char *dir = getenv("VORTEX_FIRESIM_DUMP_UPLOADS");
  if (dir == nullptr) {
    return;
  }
  static int index = 0;
  char path[1024];
  snprintf(path, sizeof(path), "%s/upload_%03d_0x%llx_%llu.bin", dir, index++,
           (unsigned long long)addr, (unsigned long long)size);
  FILE *f = fopen(path, "wb");
  if (f == nullptr) {
    fprintf(stderr, "[firesim] cannot write %s\n", path);
    return;
  }
  const size_t written = fwrite(value, 1, size, f);
  fclose(f);
  if (written != size) {
    fprintf(stderr, "[firesim] short write dumping %s\n", path);
    return;
  }
  fprintf(stderr, "[firesim] dumped upload -> %s\n", path);
}

// Rounds of batched stepping a DCR read waits for its answer, each round
// advancing 256 target cycles. A cache flush legitimately takes thousands of
// cycles; anything past this is a defect, and reporting it quickly beats
// waiting an hour to be sure.
static uint32_t dcr_poll_rounds() {
  static const uint32_t value = [] {
    if (const char *env = getenv("VORTEX_FIRESIM_DCR_POLL_ROUNDS")) {
      const long v = strtol(env, nullptr, 0);
      if (v > 0) {
        return static_cast<uint32_t>(v);
      }
    }
    // One round is one target cycle, because dcr_rsp_valid is a one-cycle pulse
    // that a batched step would miss. The bound is in cycles, not rounds, so it
    // has to be the full budget rather than a count of batches.
    return 131072u;
  }();
  return value;
}

// Whether a healthy completion also dumps counters. Off by default: each dump
// costs a dozen DCR reads, which is real time on this vehicle.
static bool report_on_idle() {
  static const bool value = (getenv("VORTEX_FIRESIM_REPORT_ON_IDLE") != nullptr);
  return value;
}

// Whether every write is read back and compared. Debug-only: it doubles MMIO
// traffic, which on this vehicle is the dominant cost.
static bool verify_writes() {
  static const bool value = (getenv("VORTEX_FIRESIM_VERIFY_WRITES") != nullptr);
  return value;
}

void access_mem(uint64_t addr, uint64_t size, void *value, bool is_write) {
  const size_t chunk =
      g_sim->loadmem.get_mem_data_chunk() * sizeof(uint32_t);
  auto *bytes = static_cast<uint8_t *>(value);

  mpz_t word;
  mpz_init(word);

  uint64_t done = 0;
  while (done < size) {
    const uint64_t cur = addr + done;
    const uint64_t base = (cur / chunk) * chunk;
    const uint64_t off = cur - base;
    const uint64_t n = std::min<uint64_t>(chunk - off, size - done);

    // A beat-aligned write of one or more whole beats goes out as a single
    // burst. Each write_mem costs three MMIO writes to arm the widget before
    // any data moves, so beat-at-a-time spends more than half its round trips
    // on addressing; one burst arms it once and streams the rest.
    if (is_write && off == 0 && (size - done) >= chunk) {
      const uint64_t span = ((size - done) / chunk) * chunk;
      mpz_import(word, span, -1, 1, 0, 0, bytes + done);
      g_sim->loadmem.write_mem_chunk(base, word, span);
      done += span;
      continue;
    }

    std::vector<uint8_t> buf(chunk, 0);
    const bool partial = (off != 0) || (n != chunk);
    if (!is_write || partial) {
      g_sim->loadmem.read_mem(base, word);
      size_t count = 0;
      mpz_export(buf.data(), &count, -1, 1, 0, 0, word);
    }

    if (is_write) {
      std::memcpy(buf.data() + off, bytes + done, n);
      mpz_import(word, chunk, -1, 1, 0, 0, buf.data());
      g_sim->loadmem.write_mem(base, word);
    } else {
      std::memcpy(bytes + done, buf.data() + off, n);
    }
    done += n;
  }

  mpz_clear(word);

  // Cross-path check: read back what was just written and compare. A write path
  // that is self-consistent can still disagree with what the target sees, and a
  // check that only round-trips one path cannot tell the difference -- that is
  // exactly how a broken upload stayed green for days. Off by default because
  // it doubles the MMIO cost of every transfer.
  if (is_write && verify_writes()) {
    std::vector<uint8_t> back(size, 0);
    access_mem(addr, size, back.data(), false);
    if (std::memcmp(back.data(), value, size) != 0) {
      size_t first = 0;
      while (first < size && back[first] == bytes[first]) {
        ++first;
      }
      fprintf(stderr,
              "[firesim] WRITE VERIFY FAILED at 0x%llx size=%llu: first bad "
              "byte at +%zu, wrote 0x%02x read 0x%02x\n",
              (unsigned long long)addr, (unsigned long long)size, first,
              bytes[first], back[first]);
    }
  }
}

} // namespace

namespace vortex {

// Largest number of target cycles a single poll may advance. Each poll costs a
// thread handoff and an MMIO round trip, so stepping only a few cycles spends
// all its time on host overhead and squanders the point of running the target
// in fabric. Tunable because that trade is workload-specific.
static uint32_t step_batch_ceiling() {
  static const uint32_t value = [] {
    if (const char *env = getenv("VORTEX_FIRESIM_STEP_BATCH")) {
      const long v = strtol(env, nullptr, 0);
      if (v > 0) {
        return static_cast<uint32_t>(v);
      }
    }
    return 100000u;
  }();
  return value;
}

// Smallest, used while the target is idle. `busy` is a level the caller watches
// for a rising edge, and a poll only observes its value at the end of the step:
// step far enough and a kernel can start and finish inside one batch, so the
// edge is never seen and the caller waits for a launch that already happened.
// Sampling finely while idle makes that impossible for any kernel longer than
// this, which is every real one.
static constexpr uint32_t kStepBatchFloor = 16;

class firesim_sim::Impl {
public:
  ~Impl() {
    if (thread_.joinable()) {
      {
        std::lock_guard<std::mutex> lock(channel_.mtx);
        channel_.shutdown = true;
      }
      channel_.to_sim.notify_all();
      thread_.join();
    }
    g_channel = nullptr;
  }

  int init(const char *bitstream) {
    g_channel = &channel_;

    argv_storage_.emplace_back("firesim");
    if (bitstream != nullptr && *bitstream != '\0') {
      argv_storage_.emplace_back(std::string("+binary_file=") + bitstream);
    }
    for (auto &s : argv_storage_) {
      argv_.push_back(const_cast<char *>(s.c_str()));
    }

    thread_ = std::thread(
        [this] { entry(static_cast<int>(argv_.size()), argv_.data()); });

    FSIM_TRACE("waiting for the simulator thread to come up");
    {
      std::unique_lock<std::mutex> lock(channel_.mtx);
      channel_.to_host.wait(lock, [this] { return channel_.ready; });
    }
    FSIM_TRACE("simulator thread is up; issuing reset");

    // Vortex samples reset for VX_CFG_RESET_DELAY cycles before it accepts any
    // request, so clear the control inputs and hold reset well past that.
    return submit("reset", [] {
      g_sim->peek_poke.poke("ctrl_start", 0, true);
      g_sim->peek_poke.poke("ctrl_dcr_req_valid", 0, true);
      g_sim->peek_poke.poke("reset", 1, true);
      g_sim->step(32);
      g_sim->peek_poke.poke("reset", 0, true);
      g_sim->step(1);
      // Diagnostic knob: shifts every subsequent target cycle by a fixed
      // amount. The bring-up harness runs this same launch to completion while
      // the runtime does not, and the only residual difference between them is
      // that the harness issues one extra DCR write at init -- worth two target
      // cycles. If the outcome depends on this offset, the defect is a
      // timing-sensitive race rather than anything about the data or protocol,
      // and that is worth knowing before looking any further.
      if (const char *env = getenv("VORTEX_FIRESIM_PRESTART_STEPS")) {
        const long n = strtol(env, nullptr, 0);
        if (n > 0) {
          fprintf(stderr, "[firesim] pre-start offset: %ld target cycles\n", n);
          g_sim->step(static_cast<uint32_t>(n));
        }
      }
    });
  }

  // Runs `fn` on the simulator thread and waits for it to finish. Requests are
  // serialized, so the target advances only in the order asked for.
  int submit(const std::string &op, request_t fn) {
    {
      std::lock_guard<std::mutex> lock(channel_.mtx);
      if (channel_.shutdown) {
        return -1;
      }
      channel_.request_done = false;
      channel_.pending.push({op, std::move(fn)});
    }
    channel_.to_sim.notify_all();
    FSIM_TRACE("%s enqueued, waiting", op.c_str());

    std::unique_lock<std::mutex> lock(channel_.mtx);
    channel_.to_host.wait(lock, [this] { return channel_.request_done; });
    FSIM_TRACE("%s acknowledged", op.c_str());
    return 0;
  }

private:
  channel_t channel_;
  std::thread thread_;
  std::vector<std::string> argv_storage_;
  std::vector<char *> argv_;
};

firesim_sim::firesim_sim()
    : impl_(new Impl()), step_batch_(kStepBatchFloor), busy_cycles_(0),
      stall_reported_(false) {}

// Target cycles of continuous busy after which the target is assumed wedged and
// its bus activity is reported. Well above any kernel this platform runs today
// (demo is ~7k cycles under rtlsim) so a legitimately long run is not flagged.
static uint64_t stall_report_cycles() {
  static const uint64_t value = [] {
    if (const char *env = getenv("VORTEX_FIRESIM_STALL_CYCLES")) {
      const long long v = atoll(env);
      if (v > 0) {
        return static_cast<uint64_t>(v);
      }
    }
    return static_cast<uint64_t>(500000);
  }();
  return value;
}

void firesim_sim::report_progress(const char *why) {
  uint64_t tcycle = 0;
  const int ret = impl_->submit("read target clock",
                                [&tcycle] { tcycle = g_sim->clock.tcycle(); });
  if (ret != 0) {
    return;
  }
  // Requested cycles versus target cycles actually elapsed. If these diverge the
  // simulation has stopped advancing the design, and every "the core is idle"
  // conclusion drawn from frozen counters would be an artefact of that rather
  // than anything the design did.
  fprintf(stderr, "[firesim] %s: requested=%llu tcycle=%llu\n", why,
          (unsigned long long)busy_cycles_, (unsigned long long)tcycle);
  report_core_counters();
}

// Dumps Vortex's own performance counters over the DCR read path.
//
// The AXI boundary cannot distinguish a core waiting forever on a response it
// never retired from a core with nothing left to do -- both look like an idle
// bus. These counters can: they say whether warps are still resident, whether
// they are parked at a barrier, and which pipeline is holding them.
//
// Requires the RTL to be built with PERF_ENABLE (see hw/syn/firesim/Makefile);
// without it VX_csr_data.sv ties the counters off and every read returns zero,
// so an all-zero line here means the instrument is absent, not that the core is
// idle. That distinction is exactly the kind this project has been bitten by,
// so it is called out rather than left to be inferred.
void firesim_sim::report_core_counters() {
  struct counter_t {
    const char *name;
    uint32_t csr;
  };
  static constexpr counter_t kCore[] = {
      {"sched_idle", 0xB03},  {"active_warps", 0xB04}, {"stalled_warps", 0xB05},
      {"issued_warps", 0xB06}, {"stall_fetch", 0xB08}, {"stall_ibuf", 0xB09},
      {"stall_scrb", 0xB0A},  {"stall_opds", 0xB0B},   {"stall_alu", 0xB0C},
      {"stall_lsu", 0xB0E},
      {"stall_sfu", 0xB0F},   {"stall_tcu", 0xB10},
      // Vortex's own per-unit dispatched-instruction counts. These are
      // independent of the wrapper's debug taps, so they arbitrate when a tap
      // and the per-warp pending counters disagree: their sum is the number of
      // instructions that reached an execute unit, measured by the design
      // rather than by this investigation's instrumentation.
      {"issued_threads", 0xB07},
      {"instr_alu", 0xB13},   {"instr_fpu", 0xB14},
      {"instr_lsu", 0xB15},   {"instr_sfu", 0xB16},
  };
  constexpr uint32_t kMpmBase = 0xB00;      // VX_CSR_MPM_BASE
  constexpr uint32_t kDcrMpmValue = 0x001;  // VX_DCR_BASE_MPM_VALUE
  constexpr uint32_t kClassCore = 1;        // VX_DCR_MPM_CLASS_CORE

  std::string line;
  bool all_zero = true;
  for (const auto &c : kCore) {
    const uint32_t tag =
        (kClassCore << 22) | ((c.csr - kMpmBase) << 16) | 0 /* core 0 */;
    uint32_t value = 0;
    if (this->dcr_read(kDcrMpmValue, tag, &value) != 0) {
      line += " ";
      line += c.name;
      line += "=?";
      continue;
    }
    all_zero = all_zero && (value == 0);
    line += " ";
    line += c.name;
    line += "=";
    line += std::to_string(value);
  }
  fprintf(stderr, "[firesim] core counters:%s%s\n", line.c_str(),
          all_zero ? "  (all zero -- was the RTL built with PERF_ENABLE?)" : "");

  // The dcache, separately. A load that allocates an MSHR whose fill never
  // finalizes would show exactly the core-side signature already measured --
  // one instruction outstanding, scoreboard blocked, no further memory traffic
  // -- and only these counters can tell that apart from the core simply having
  // stopped asking. `reads` minus `miss_r` bounds what has actually been
  // served.
  static constexpr counter_t kDcache[] = {
      {"reads", 0xB03},   {"writes", 0xB04},  {"miss_r", 0xB05},
      {"miss_w", 0xB06},  {"evicts", 0xB07},  {"bank_st", 0xB08},
      {"mshr_st", 0xB09},
  };
  constexpr uint32_t kClassDcache = 4;  // VX_DCR_MPM_CLASS_DCACHE
  std::string dline;
  for (const auto &c : kDcache) {
    const uint32_t tag = (kClassDcache << 22) | ((c.csr - kMpmBase) << 16) | 0;
    uint32_t value = 0;
    dline += " ";
    dline += c.name;
    dline += "=";
    dline += (this->dcr_read(kDcrMpmValue, tag, &value) == 0)
                 ? std::to_string(value)
                 : std::string("?");
  }
  fprintf(stderr, "[firesim] dcache counters:%s\n", dline.c_str());


}

firesim_sim::~firesim_sim() { delete impl_; }

int firesim_sim::init(const char *bitstream) { return impl_->init(bitstream); }

int firesim_sim::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->submit(label("dcr_write addr=0x%x data=0x%x", addr, value), [addr, value] {
    auto &pp = g_sim->peek_poke;
    pp.poke("ctrl_dcr_req_valid", 1, true);
    pp.poke("ctrl_dcr_req_rw", 1, true);
    pp.poke("ctrl_dcr_req_addr", addr, true);
    pp.poke("ctrl_dcr_req_data", value, true);
    g_sim->step(1);
    pp.poke("ctrl_dcr_req_valid", 0, true);
    g_sim->step(1);
  });
}

int firesim_sim::dcr_read(uint32_t addr, uint32_t tag, uint32_t *value) {
  uint32_t sampled = 0;
  bool answered = false;
  int ret = impl_->submit(label("dcr_read addr=0x%x", addr), [addr, tag, &sampled, &answered] {
    auto &pp = g_sim->peek_poke;
    pp.poke("ctrl_dcr_req_valid", 1, true);
    pp.poke("ctrl_dcr_req_rw", 0, true);
    pp.poke("ctrl_dcr_req_addr", addr, true);
    pp.poke("ctrl_dcr_req_data", tag, true);
    g_sim->step(1);
    pp.poke("ctrl_dcr_req_valid", 0, true);

    // dcr_rsp_valid is a one-cycle pulse passed straight out of the core, so
    // the poll has to advance one target cycle at a time: a batched step lands
    // past the pulse and the read reports unanswered on a target that replied.
    //
    // Bounded rather than open-ended: a target that never answers must fail the
    // read instead of hanging the runtime. The bound is generous because a
    // flush legitimately takes tens of thousands of cycles.
    //
    // The cost is one MMIO round trip per target cycle, which on the card is
    // tenths of a second for a flush and under emulation is minutes. Batched
    // polling needs the response latched at the wrapper; that is a target-side
    // change and a new bitstream, so it is left as a follow-up rather than
    // carried here as a driver-side workaround against ports the shipping
    // design does not have.
    const uint32_t kPollRounds = dcr_poll_rounds();
    for (uint32_t i = 0; i < kPollRounds; ++i) {
      g_sim->step(1);
      if (pp.peek("ctrl_dcr_rsp_valid", true) != 0) {
        sampled = pp.peek("ctrl_dcr_rsp_data", true);
        answered = true;
        break;
      }
    }
  });
  if (ret != 0) {
    return ret;
  }
  if (!answered) {
    // Report the flush gate state here rather than at the busy->idle
    // transition: the command processor issues the flush *after* the target
    // goes idle, so a dump taken then would show every sticky bit clear and say
    // nothing. This is the only point at which the gates have had a chance to
    // move.
    fprintf(stderr, "[firesim] DCR read addr=0x%x went unanswered\n", addr);

    return -1;
  }
  *value = sampled;
  return 0;
}

// Stamps a recognizable pattern over a region just before launch.
//
// The destination buffer is never written by the host, so on a write-allocate
// miss the cache fetches uninitialized DRAM. That makes "the line was overwritten
// by its fill" and "the line was never stored to" produce the same
// unrecognizable garbage. Filling it first tells them apart: if the value that
// comes back is this pattern, the fill landed on top of the store.
static void prefill_region() {
  const char *spec = getenv("VORTEX_FIRESIM_PREFILL");
  if (spec == nullptr) {
    return;
  }
  unsigned long long addr = 0, size = 0, value = 0;
  if (sscanf(spec, "%llx:%llu:%llx", &addr, &size, &value) != 3) {
    fprintf(stderr, "[firesim] PREFILL wants <hexaddr>:<size>:<hexval>\n");
    return;
  }
  std::vector<uint32_t> buf(size / sizeof(uint32_t),
                            static_cast<uint32_t>(value));
  access_mem(addr, buf.size() * sizeof(uint32_t), buf.data(), true);
  fprintf(stderr, "[firesim] prefilled 0x%llx +%llu with 0x%08x\n", addr, size,
          static_cast<uint32_t>(value));
}

int firesim_sim::start() {
  impl_->submit("prefill", [] { prefill_region(); });
  return impl_->submit("start", [] {
    auto &pp = g_sim->peek_poke;
    pp.poke("ctrl_start", 1, true);
    g_sim->step(1);
    pp.poke("ctrl_start", 0, true);
    g_sim->step(1);
  });
}

int firesim_sim::is_busy(bool *busy) {
  uint32_t sampled = 0;
  // Advance before sampling: the target only moves while it is being asked to,
  // so a caller that merely polls without stepping would spin forever.
  const uint32_t batch = step_batch_;
  int ret = impl_->submit(label("step %u + busy", batch), [&sampled, batch] {
    g_sim->step(batch);
    sampled = g_sim->peek_poke.peek("ctrl_busy", true);
  });
  if (ret != 0) {
    return ret;
  }
  *busy = (sampled != 0);

  // Once the target is known to be running there is no edge left to miss, so
  // grow towards the ceiling and give a long kernel the throughput it needs.
  // Going idle drops back to the floor, ready for the next launch.
  if (*busy) {
    const uint32_t ceiling = step_batch_ceiling();
    step_batch_ = (batch >= ceiling / 4) ? ceiling : batch * 4;
    busy_cycles_ += batch;
    // A target that stays busy far past any plausible run length is wedged.
    // Report what it did on the memory bus before it stopped making progress:
    // whether it ever issued a write separates a stalled pipeline from a
    // kernel that ran and never stored its results. Reported once, because the
    // point is the transition, not the polling.
    if (!stall_reported_ && busy_cycles_ >= stall_report_cycles()) {
      stall_reported_ = true;
      report_progress("target still busy");
    }
  } else {
    // Report on a healthy completion too, not only on a stall. Without this the
    // only counter dumps come from wedged runs, leaving nothing to compare a
    // good run against -- and "the kernel finished" and "the kernel gave up
    // early" look identical from outside. This lives here rather than in
    // ready_wait because the command processor polls is_busy directly through
    // its vortex_busy hook and never calls ready_wait at all.
    if (busy_cycles_ > 0 && report_on_idle()) {
      report_progress("target went idle");
    }
    step_batch_ = std::min(kStepBatchFloor, step_batch_ceiling());
    busy_cycles_ = 0;
    stall_reported_ = false;
  }
  return 0;
}

int firesim_sim::ready_wait(int64_t timeout_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  for (;;) {
    bool busy = true;
    int ret = this->is_busy(&busy);
    if (ret != 0) {
      return ret;
    }
    if (!busy) {
      return 0;
    }
    if (timeout_ms >= 0 && std::chrono::steady_clock::now() >= deadline) {
      return -1;
    }
  }
}

int firesim_sim::mem_write(uint64_t addr, uint64_t size, const void *value) {
  dump_upload(addr, size, value);
  return impl_->submit(
      label("mem_write addr=0x%lx size=%lu", (unsigned long)addr, (unsigned long)size),
      [addr, size, value] { access_mem(addr, size, const_cast<void *>(value), true); });
}

int firesim_sim::mem_read(uint64_t addr, uint64_t size, void *value) {
  return impl_->submit(
      label("mem_read addr=0x%lx size=%lu", (unsigned long)addr, (unsigned long)size),
      [addr, size, value] { access_mem(addr, size, value, false); });
}

}
