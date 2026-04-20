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

#include <common.h>
#include <vortex.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

class ProfilingMode {
public:
  ProfilingMode() : mpm_class_(0) {
    auto profiling_s = getenv("VORTEX_PROFILING");
    if (profiling_s) {
      mpm_class_ = std::atoi(profiling_s);
    }
  }

  ~ProfilingMode() {}

  int mpm_class() const {
    return mpm_class_;
  }

private:
  int mpm_class_;
};

int get_profiling_mode() {
  static ProfilingMode gProfilingMode;
  return gProfilingMode.mpm_class();
}

static inline double safe_div(double num, double den) {
  return (den == 0.0) ? 0.0 : (num / den);
}

static inline int calc_percent(uint64_t part, uint64_t total) {
  return (total == 0) ? 0 : (int)std::lround(safe_div((double)part, (double)total) * 100.0);
}

static inline int calc_ratio(uint64_t misses, uint64_t accesses) {
  if (accesses == 0)
    return 0;
  double miss_rate = safe_div((double)misses, (double)accesses);
  return (int)std::lround((1.0 - miss_rate) * 100.0);
}

static inline int calc_utility(uint64_t useful, uint64_t stalls) {
  return calc_percent(useful, useful + stalls);
}

static inline double bytes_to_GB(uint64_t bytes) {
  return (double)bytes / 1e9;
}

static inline void perf_print_core(FILE *stream, int core_id, const char *fmt, ...) {
  std::va_list args;
  va_start(args, fmt);
  std::fprintf(stream, "PERF: core%d: ", core_id);
  std::vfprintf(stream, fmt, args);
  std::fprintf(stream, "\n");
  va_end(args);
}

static inline void perf_print(FILE *stream, const char *fmt, ...) {
  std::va_list args;
  va_start(args, fmt);
  std::fprintf(stream, "PERF: ");
  std::vfprintf(stream, fmt, args);
  std::fprintf(stream, "\n");
  va_end(args);
}

// -----------------------------------------------------------------------------
// Data structs
// -----------------------------------------------------------------------------

struct CoreCounters {
  uint64_t cycles = 0;
  uint64_t instrs = 0;

  // scheduler
  uint64_t sched_idle = 0;
  uint64_t active_warps = 0;
  uint64_t stalled_warps = 0;
  uint64_t issued_warps = 0;
  uint64_t issued_threads = 0;

  // pipeline stalls
  uint64_t stall_fetch = 0;
  uint64_t stall_ibuf = 0;
  uint64_t stall_scrb = 0;
  uint64_t stall_opds = 0;
  uint64_t stall_alu = 0;
  uint64_t stall_fpu = 0;
  uint64_t stall_lsu = 0;
  uint64_t stall_sfu = 0;
  uint64_t stall_tcu = 0;

  // workload mix
  uint64_t instr_alu = 0;
  uint64_t instr_fpu = 0;
  uint64_t instr_lsu = 0;
  uint64_t instr_sfu = 0;
  uint64_t instr_tcu = 0;

  // branches
  uint64_t branches = 0;
  uint64_t divergence = 0;

  // memory (front-end + LSU)
  uint64_t ifetches = 0;
  uint64_t ifetch_lt = 0;
  uint64_t loads = 0;
  uint64_t load_lt = 0;
  uint64_t stores = 0;

  // memory
  uint64_t mem_reads = 0;
  uint64_t mem_writes = 0;
};

struct CacheCounters {
  uint64_t reads = 0;
  uint64_t writes = 0;
  uint64_t miss_r = 0;
  uint64_t miss_w = 0;
  uint64_t bank_st = 0;
  uint64_t mshr_st = 0;
};

struct Metric {
  const char *label;
  int value;
  bool enabled;
};

static void print_metric_list(FILE *stream, const char *category, int core_id, const std::vector<Metric> &metrics) {
  std::stringstream ss;
  ss << category << ": ";
  bool first = true;
  for (const auto &m : metrics) {
    if (m.enabled) {
      if (!first)
        ss << ", ";
      ss << m.label << "=" << m.value << "%";
      first = false;
    }
  }

  if (core_id >= 0) {
    perf_print_core(stream, core_id, "%s", ss.str().c_str());
  } else {
    perf_print(stream, "%s", ss.str().c_str());
  }
}

extern int vx_dump_perf(vx_device_h hdevice, FILE *stream) {
  if (nullptr == stream)
    stream = stdout;

  uint64_t num_cores = 0;
  uint64_t num_clusters = 0;
  uint64_t socket_size = 0;
  uint64_t num_warps = 0;
  uint64_t num_threads = 0;
  uint64_t issue_width = 0;
  uint64_t isa_flags = 0;
  uint64_t clock_mhz = 0;
  uint64_t peak_mem_bw_MBps = 0;

  const auto mpm_class = get_profiling_mode();

  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_CORES, &num_cores), { return err; });
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_CLUSTERS, &num_clusters), { return err; });
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_SOCKET_SIZE, &socket_size), { return err; });
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_WARPS, &num_warps), { return err; });
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_THREADS, &num_threads), { return err; });
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_ISSUE_WIDTH, &issue_width), { return err; });
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_ISA_FLAGS, &isa_flags), { return err; });
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_CLOCK_RATE, &clock_mhz), { return err; });
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_PEAK_MEM_BW, &peak_mem_bw_MBps), { return err; });

  const bool icache_en = (isa_flags & VX_ISA_EXT_ICACHE) != 0;
  const bool dcache_en = (isa_flags & VX_ISA_EXT_DCACHE) != 0;
  const bool l2_en = (isa_flags & VX_ISA_EXT_L2CACHE) != 0;
  const bool l3_en = (isa_flags & VX_ISA_EXT_L3CACHE) != 0;
  const bool lmem_en = (isa_flags & VX_ISA_EXT_LMEM) != 0;

  const bool fpu_en = (isa_flags & VX_ISA_STD_F) != 0;
  const bool tcu_en = (isa_flags & VX_ISA_EXT_TCU) != 0;

  std::vector<CoreCounters> cores(num_cores);
  uint64_t total_instrs = 0;
  uint64_t max_cycles = 0;

  // 1. Per-Core Loop (Always runs to get basics)
  for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
    auto &c = cores[core_id];

    CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MCYCLE, core_id, &c.cycles), { return err; });
    CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MINSTRET, core_id, &c.instrs), { return err; });

    total_instrs += c.instrs;
    max_cycles = std::max<uint64_t>(max_cycles, c.cycles);
  }

  switch (mpm_class) {
  case VX_DCR_MPM_CLASS_BASE:
    break;

  case VX_DCR_MPM_CLASS_CORE: {
    CoreCounters tot;

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
      auto &c = cores[core_id];

      // Query Core Specifics
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_SCHED_IDLE, core_id, &c.sched_idle), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_ACTIVE_WARPS, core_id, &c.active_warps), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STALLED_WARPS, core_id, &c.stalled_warps), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_ISSUED_WARPS, core_id, &c.issued_warps), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_ISSUED_THREADS, core_id, &c.issued_threads), { return err; });

      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STALL_FETCH, core_id, &c.stall_fetch), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STALL_IBUF, core_id, &c.stall_ibuf), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STALL_SCRB, core_id, &c.stall_scrb), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STALL_OPDS, core_id, &c.stall_opds), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STALL_ALU, core_id, &c.stall_alu), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STALL_FPU, core_id, &c.stall_fpu), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STALL_LSU, core_id, &c.stall_lsu), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STALL_SFU, core_id, &c.stall_sfu), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STALL_TCU, core_id, &c.stall_tcu), { return err; });

      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_INSTR_ALU, core_id, &c.instr_alu), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_INSTR_FPU, core_id, &c.instr_fpu), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_INSTR_LSU, core_id, &c.instr_lsu), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_INSTR_SFU, core_id, &c.instr_sfu), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_INSTR_TCU, core_id, &c.instr_tcu), { return err; });

      // Branches
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_BRANCHES, core_id, &c.branches), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DIVERGENCE, core_id, &c.divergence), { return err; });

      // Memory (front-end + LSU)
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_IFETCHES, core_id, &c.ifetches), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_IFETCH_LT, core_id, &c.ifetch_lt), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_LOADS, core_id, &c.loads), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_LOAD_LT, core_id, &c.load_lt), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_STORES, core_id, &c.stores), { return err; });

      if (num_cores > 1) {
        // Per-Core report
        const uint64_t cycles = c.cycles;
        const uint64_t cycles_wide = cycles * issue_width;
        const double ipc = safe_div((double)c.instrs, (double)cycles);

        // Scheduler report
        const int idle_pct = calc_percent(c.sched_idle, cycles);
        const double avg_occ = safe_div((double)c.active_warps, (double)cycles);
        const int occ_pct = (int)std::lround(safe_div(avg_occ, (double)num_warps) * 100.0);
        const double warp_eff = safe_div((double)c.issued_threads, (double)c.issued_warps);
        const int warp_eff_pct = (int)std::lround(safe_div(warp_eff, (double)num_threads) * 100.0);
        perf_print_core(stream, core_id, "scheduler: idle=%d%%, occupancy=%.1f (%d%%), simt_util=%.1f (%d%%)",
                        idle_pct, avg_occ, occ_pct, warp_eff, warp_eff_pct);

        // Pipeline stalls report
        std::vector<Metric> stall_metrics = {
          {"fetch", calc_percent(c.stall_fetch, cycles), true},
          {"ibuf", calc_percent(c.stall_ibuf, cycles), true},
          {"scrb", calc_percent(c.stall_scrb, cycles_wide), true},
          {"opds", calc_percent(c.stall_opds, cycles_wide), true},
          {"alu", calc_percent(c.stall_alu, cycles_wide), true},
          {"lsu", calc_percent(c.stall_lsu, cycles_wide), true},
          {"sfu", calc_percent(c.stall_sfu, cycles_wide), true},
          {"fpu", calc_percent(c.stall_fpu, cycles_wide), fpu_en},
          {"tcu", calc_percent(c.stall_tcu, cycles_wide), tcu_en}
        };
        print_metric_list(stream, "stalls", core_id, stall_metrics);

        // Instruction mix report
        std::vector<Metric> mix_metrics = {
          {"alu", calc_percent(c.instr_alu, c.instrs), true},
          {"lsu", calc_percent(c.instr_lsu, c.instrs), true},
          {"sfu", calc_percent(c.instr_sfu, c.instrs), true},
          {"fpu", calc_percent(c.instr_fpu, c.instrs), fpu_en},
          {"tcu", calc_percent(c.instr_tcu, c.instrs), tcu_en}
        };
        print_metric_list(stream, "inst_mix", core_id, mix_metrics);

        // Branch report
        int div_pct = calc_percent(c.divergence, c.branches);
        perf_print_core(stream, core_id, "branches: total=%" PRIu64 ", divergent=%" PRIu64 " (rate=%d%%)",
                        c.branches, c.divergence, div_pct);

        // Memory report
        const double ifetch_avg_lt = safe_div((double)c.ifetch_lt, (double)c.ifetches);
        const double load_avg_lt = safe_div((double)c.load_lt, (double)c.loads);
        perf_print_core(stream, core_id, "memory: ifetch_lat=%.2f, load_lat=%.2f, loads=%" PRIu64 ", stores=%" PRIu64,
                        ifetch_avg_lt, load_avg_lt, c.loads, c.stores);

        perf_print_core(stream, core_id, "instrs=%" PRIu64 ", cycles=%" PRIu64 ", IPC=%.3f", c.instrs, c.cycles, ipc);
      }

      // Accumulate
      tot.cycles += c.cycles;
      tot.instrs += c.instrs;
      tot.sched_idle += c.sched_idle;
      tot.active_warps += c.active_warps;
      tot.stalled_warps += c.stalled_warps;
      tot.issued_warps += c.issued_warps;
      tot.issued_threads += c.issued_threads;

      tot.stall_fetch += c.stall_fetch;
      tot.stall_ibuf += c.stall_ibuf;
      tot.stall_scrb += c.stall_scrb;
      tot.stall_opds += c.stall_opds;
      tot.stall_alu += c.stall_alu;
      tot.stall_fpu += c.stall_fpu;
      tot.stall_lsu += c.stall_lsu;
      tot.stall_sfu += c.stall_sfu;
      tot.stall_tcu += c.stall_tcu;

      tot.instr_alu += c.instr_alu;
      tot.instr_fpu += c.instr_fpu;
      tot.instr_lsu += c.instr_lsu;
      tot.instr_sfu += c.instr_sfu;
      tot.instr_tcu += c.instr_tcu;

      tot.branches += c.branches;
      tot.divergence += c.divergence;

      tot.ifetches += c.ifetches;
      tot.ifetch_lt += c.ifetch_lt;
      tot.loads += c.loads;
      tot.load_lt += c.load_lt;
      tot.stores += c.stores;
    }

    // Query global MPM counters
    CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_MEM_READS, 0, &tot.mem_reads), { return err; });
    CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_MEM_WRITES, 0, &tot.mem_writes), { return err; });

    // Core Summary
    uint64_t tot_cycles_wide = tot.cycles * issue_width;

    // Scheduler report
    const int tot_idle_pct = calc_percent(tot.sched_idle, tot.cycles);
    const double tot_avg_occ = safe_div((double)tot.active_warps, (double)tot.cycles);
    const int tot_occ_pct = (int)std::lround(safe_div(tot_avg_occ, (double)num_warps) * 100.0);
    const double tot_warp_eff = safe_div((double)tot.issued_threads, (double)tot.issued_warps);
    const int tot_warp_eff_pct = (int)std::lround(safe_div(tot_warp_eff, (double)num_threads) * 100.0);
    perf_print(stream, "scheduler: idle=%d%%, occupancy=%.1f (%d%%), simt_util=%.1f (%d%%)",
               tot_idle_pct, tot_avg_occ, tot_occ_pct, tot_warp_eff, tot_warp_eff_pct);

    // Pipeline stalls report
    std::vector<Metric> global_stalls = {
      {"fetch", calc_percent(tot.stall_fetch, tot.cycles), true},
      {"ibuf", calc_percent(tot.stall_ibuf, tot.cycles), true},
      {"scrb", calc_percent(tot.stall_scrb, tot_cycles_wide), true},
      {"opds", calc_percent(tot.stall_opds, tot_cycles_wide), true},
      {"alu", calc_percent(tot.stall_alu, tot_cycles_wide), true},
      {"lsu", calc_percent(tot.stall_lsu, tot_cycles_wide), true},
      {"sfu", calc_percent(tot.stall_sfu, tot_cycles_wide), true},
      {"fpu", calc_percent(tot.stall_fpu, tot_cycles_wide), fpu_en},
      {"tcu", calc_percent(tot.stall_tcu, tot_cycles_wide), tcu_en}
    };
    print_metric_list(stream, "stalls", -1, global_stalls);

    // Instruction mix report
    std::vector<Metric> global_mix = {
      {"alu", calc_percent(tot.instr_alu, tot.instrs), true},
      {"lsu", calc_percent(tot.instr_lsu, tot.instrs), true},
      {"sfu", calc_percent(tot.instr_sfu, tot.instrs), true},
      {"fpu", calc_percent(tot.instr_fpu, tot.instrs), fpu_en},
      {"tcu", calc_percent(tot.instr_tcu, tot.instrs), tcu_en}
    };
    print_metric_list(stream, "inst_mix", -1, global_mix);

    // Branch report
    int tot_div_pct = calc_percent(tot.divergence, tot.branches);
    perf_print(stream, "branches: total=%" PRIu64 ", divergent=%" PRIu64 " (rate=%d%%)",
               tot.branches, tot.divergence, tot_div_pct);

    // Memory report
    const double tot_ifetch_avg_lt = safe_div((double)tot.ifetch_lt, (double)tot.ifetches);
    const double tot_load_avg_lt = safe_div((double)tot.load_lt, (double)tot.loads);
    uint64_t read_bytes = tot.mem_reads * CACHE_BLOCK_SIZE;
    uint64_t write_bytes = tot.mem_writes * CACHE_BLOCK_SIZE;
    perf_print(stream, "memory: ifetch_lat=%.2f, load_lat=%.2f, loads=%" PRIu64 ", stores=%" PRIu64 ", read_bytes=%" PRIu64 ", write_bytes=%" PRIu64,
               tot_ifetch_avg_lt, tot_load_avg_lt, tot.loads, tot.stores, read_bytes, write_bytes);
  } break;

  case VX_DCR_MPM_CLASS_MEM: {
    const uint64_t num_sockets = (num_cores + socket_size - 1) / socket_size;
    const uint64_t cores_per_cluster = (num_cores + num_clusters - 1) / num_clusters;

    CacheCounters icache_tot, dcache_tot, l2_tot, l3_tot;
    CacheCounters lmem_tot;
    uint64_t coalescer_miss_tot = 0;

    // Per-Core Local Memory & Coalescer (Print Core First)
    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
      if (lmem_en) {
        uint64_t r = 0, w = 0, bst = 0;
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_LMEM_READS, core_id, &r), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_LMEM_WRITES, core_id, &w), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_LMEM_BANK_ST, core_id, &bst), { return err; });

        lmem_tot.reads += r;
        lmem_tot.writes += w;
        lmem_tot.bank_st += bst;
        perf_print_core(stream, core_id, "lmem: reqs=%" PRIu64 ", bank_stalls=%" PRIu64 " (utility=%d%%)",
                        r + w, bst, calc_utility(r + w, bst));
      }

      if (dcache_en) {
        uint64_t cm = 0;
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_COALESCER_MISS, core_id, &cm), { return err; });
        coalescer_miss_tot += cm;
        perf_print_core(stream, core_id, "coalescer: misses=%" PRIu64, cm);
      }
    }

    // Per-Socket L1
    for (uint32_t s = 0; s < num_sockets; ++s) {
      uint32_t rep_core = s * (uint32_t)socket_size;
      if (rep_core >= num_cores)
        continue;

      if (icache_en) {
        uint64_t r = 0, m = 0, st = 0;
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_ICACHE_READS, rep_core, &r), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_ICACHE_MISS_R, rep_core, &m), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_ICACHE_MSHR_ST, rep_core, &st), { return err; });

        icache_tot.reads += r;
        icache_tot.miss_r += m;
        icache_tot.mshr_st += st;

        perf_print_core(stream, rep_core, "icache: reads=%" PRIu64 ", miss=%" PRIu64 " (hit=%d%%), mshr_st=%" PRIu64 " (utility=%d%%)",
                        r, m, calc_ratio(m, r), st, calc_utility(m, st));
      }
      if (dcache_en) {
        uint64_t r = 0, w = 0, mr = 0, mw = 0, bst = 0, mst = 0;
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DCACHE_READS, rep_core, &r), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DCACHE_WRITES, rep_core, &w), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DCACHE_MISS_R, rep_core, &mr), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DCACHE_MISS_W, rep_core, &mw), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DCACHE_BANK_ST, rep_core, &bst), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DCACHE_MSHR_ST, rep_core, &mst), { return err; });

        dcache_tot.reads += r;
        dcache_tot.writes += w;
        dcache_tot.miss_r += mr;
        dcache_tot.miss_w += mw;
        dcache_tot.bank_st += bst;
        dcache_tot.mshr_st += mst;

        perf_print_core(stream, rep_core, "dcache: reqs=%" PRIu64 ", miss_r=%" PRIu64 " (hit=%d%%), miss_w=%" PRIu64 " (hit=%d%%), bank_st=%" PRIu64 " (utility=%d%%)",
                        r + w, mr, calc_ratio(mr, r), mw, calc_ratio(mw, w), bst, calc_utility(r + w, bst));
      }
    }

    // Per-Cluster L2
    if (l2_en) {
      for (uint32_t c = 0; c < num_clusters; ++c) {
        uint32_t rep_core = c * (uint32_t)cores_per_cluster;
        if (rep_core >= num_cores)
          continue;
        uint64_t r = 0, w = 0, mr = 0, mw = 0, bst = 0, mst = 0;
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L2CACHE_READS, rep_core, &r), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L2CACHE_WRITES, rep_core, &w), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L2CACHE_MISS_R, rep_core, &mr), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L2CACHE_MISS_W, rep_core, &mw), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L2CACHE_BANK_ST, rep_core, &bst), { return err; });
        CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L2CACHE_MSHR_ST, rep_core, &mst), { return err; });

        l2_tot.reads += r;
        l2_tot.writes += w;
        l2_tot.miss_r += mr;
        l2_tot.miss_w += mw;
        l2_tot.bank_st += bst;
        l2_tot.mshr_st += mst;

        perf_print_core(stream, rep_core, "l2cache: reqs=%" PRIu64 ", miss_r=%" PRIu64 " (hit=%d%%), miss_w=%" PRIu64 " (hit=%d%%)",
                        r + w, mr, calc_ratio(mr, r), mw, calc_ratio(mw, w));
      }
    }

    // Global L3
    if (l3_en) {
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L3CACHE_READS, 0, &l3_tot.reads), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L3CACHE_WRITES, 0, &l3_tot.writes), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L3CACHE_MISS_R, 0, &l3_tot.miss_r), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L3CACHE_MISS_W, 0, &l3_tot.miss_w), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L3CACHE_BANK_ST, 0, &l3_tot.bank_st), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_L3CACHE_MSHR_ST, 0, &l3_tot.mshr_st), { return err; });

      perf_print(stream, "l3cache: reqs=%" PRIu64 ", miss_r=%" PRIu64 " (hit=%d%%), miss_w=%" PRIu64 " (hit=%d%%)",
                 l3_tot.reads + l3_tot.writes, l3_tot.miss_r, calc_ratio(l3_tot.miss_r, l3_tot.reads),
                 l3_tot.miss_w, calc_ratio(l3_tot.miss_w, l3_tot.writes));
    }

    // Global DRAM
    {
      uint64_t r = 0, w = 0, lat = 0, bst = 0;
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_MEM_READS, 0, &r), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_MEM_WRITES, 0, &w), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_MEM_LT, 0, &lat), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_MEM_BANK_ST, 0, &bst), { return err; });

      double avg_lat = safe_div((double)lat, (double)r);
      perf_print(stream, "memory: reqs=%" PRIu64 " (r=%" PRIu64 ", w=%" PRIu64 "), lat=%.1f cyc, bank_st=%" PRIu64 " (utility=%d%%)",
                 r + w, r, w, avg_lat, bst, calc_utility(r + w, bst));
    }
  } break;

  case VX_DCR_MPM_CLASS_DXA: {
    // DXA counters are cluster-level; query representative core per cluster.
    uint64_t cores_per_cluster = (num_cores + num_clusters - 1) / num_clusters;
    uint64_t tot_transfers = 0, tot_gmem_reads = 0, tot_gmem_dedup = 0;
    uint64_t tot_lmem_writes = 0, tot_gmem_lt = 0;
    for (uint32_t c = 0; c < num_clusters; ++c) {
      uint32_t rep_core = (uint32_t)(c * cores_per_cluster);
      if (rep_core >= num_cores)
        break;
      uint64_t transfers = 0, gmem_reads = 0, gmem_dedup = 0, lmem_writes = 0, gmem_lt = 0;
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DXA_TRANSFERS,  rep_core, &transfers),  { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DXA_GMEM_READS, rep_core, &gmem_reads), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DXA_GMEM_DEDUP, rep_core, &gmem_dedup), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DXA_LMEM_WRITES,rep_core, &lmem_writes),{ return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_DXA_GMEM_LT,    rep_core, &gmem_lt),    { return err; });
      tot_transfers  += transfers;
      tot_gmem_reads += gmem_reads;
      tot_gmem_dedup += gmem_dedup;
      tot_lmem_writes+= lmem_writes;
      tot_gmem_lt    += gmem_lt;
      double avg_lat = safe_div((double)gmem_lt, (double)transfers);
      int dedup_pct  = calc_percent(gmem_dedup, gmem_reads + gmem_dedup);
      perf_print_core(stream, rep_core, "dxa: transfers=%" PRIu64 ", gmem_reads=%" PRIu64 ", gmem_dedup=%" PRIu64 " (rate=%d%%), lmem_writes=%" PRIu64 ", avg_gmem_lat=%.1f",
                      transfers, gmem_reads, gmem_dedup, dedup_pct, lmem_writes, avg_lat);
    }
    {
      double avg_lat = safe_div((double)tot_gmem_lt, (double)tot_transfers);
      int dedup_pct  = calc_percent(tot_gmem_dedup, tot_gmem_reads + tot_gmem_dedup);
      perf_print(stream, "dxa: transfers=%" PRIu64 ", gmem_reads=%" PRIu64 ", gmem_dedup=%" PRIu64 " (rate=%d%%), lmem_writes=%" PRIu64 ", avg_gmem_lat=%.1f",
                 tot_transfers, tot_gmem_reads, tot_gmem_dedup, dedup_pct, tot_lmem_writes, avg_lat);
    }
  } break;

  case VX_DCR_MPM_CLASS_TCU: {
    if (!tcu_en) {
      perf_print(stream, "TCU not supported");
      break;
    }
    uint64_t tot_tbuf_stalls = 0, tot_tbuf_cache_hits = 0, tot_lmem_reads = 0;
    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
      uint64_t tbuf_stalls = 0, tbuf_cache_hits = 0, lmem_reads = 0;
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_TCU_TBUF_STALLS, core_id, &tbuf_stalls),  { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_TCU_TBUF_CACHE_HITS,  core_id, &tbuf_cache_hits), { return err; });
      CHECK_ERR(vx_mpm_query(hdevice, mpm_class, VX_CSR_MPM_TCU_LMEM_READS,    core_id, &lmem_reads),   { return err; });
      perf_print_core(stream, core_id,
        "tcu: tbuf_stalls=%" PRIu64 ", tbuf_cache_hits=%" PRIu64 ", lmem_reads=%" PRIu64,
        tbuf_stalls, tbuf_cache_hits, lmem_reads);
      tot_tbuf_stalls  += tbuf_stalls;
      tot_tbuf_cache_hits += tbuf_cache_hits;
      tot_lmem_reads   += lmem_reads;
    }
    perf_print(stream,
      "tcu: total_tbuf_stalls=%" PRIu64 ", total_tbuf_cache_hits=%" PRIu64 ", total_lmem_reads=%" PRIu64,
      tot_tbuf_stalls, tot_tbuf_cache_hits, tot_lmem_reads);
  } break;

  default:
    fprintf(stream, "Error: invalid profiling class: %d)", mpm_class);
    return -1;
  }

  double global_ipc = safe_div((double)total_instrs, (double)max_cycles);
  perf_print(stream, "instrs=%" PRIu64 ", cycles=%" PRIu64 ", IPC=%.3f", total_instrs, max_cycles, global_ipc);

  std::fflush(stream);
  return 0;
}