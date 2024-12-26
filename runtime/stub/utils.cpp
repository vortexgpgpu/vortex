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

#include <iostream>
#include <fstream>
#include <list>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <vortex.h>
#include <assert.h>

class ProfilingMode {
public:
  ProfilingMode() : perf_class_(0) {
    auto profiling_s = getenv("VORTEX_PROFILING");
    if (profiling_s) {
      perf_class_ = std::atoi(profiling_s);
    }
  }

  ~ProfilingMode() {}

  int perf_class() const {
    return perf_class_;
  }

private:
  int perf_class_;
};

int get_profiling_mode() {
  static ProfilingMode gProfilingMode;
  return gProfilingMode.perf_class();
}

extern int vx_upload_kernel_bytes(vx_device_h hdevice, const void* content, uint64_t size, vx_buffer_h* hbuffer) {
  if (nullptr == hdevice || nullptr == content || size <= 8 || nullptr == hbuffer)
    return -1;

  auto bytes = reinterpret_cast<const uint64_t*>(content);

  auto min_vma = *bytes++;
  auto max_vma = *bytes++;
  auto bin_size = size - 2 * 8;
  auto runtime_size = (max_vma - min_vma);

  vx_buffer_h _hbuffer;
  CHECK_ERR(vx_mem_reserve(hdevice, min_vma, runtime_size, 0, &_hbuffer), {
    return err;
  });

  // mask binary region as read-only
  CHECK_ERR(vx_mem_access(_hbuffer, 0, bin_size, VX_MEM_READ), {
    vx_mem_free(_hbuffer);
    return err;
  });

  // mark global variables region as read-write
  CHECK_ERR(vx_mem_access(_hbuffer, bin_size, runtime_size - bin_size, VX_MEM_READ_WRITE), {
    vx_mem_free(_hbuffer);
    return err;
  });

  CHECK_ERR(vx_copy_to_dev(_hbuffer, bytes, 0, bin_size), {
    vx_mem_free(_hbuffer);
    return err;
  });

  *hbuffer = _hbuffer;

  return 0;
}

extern int vx_upload_kernel_file(vx_device_h hdevice, const char* filename, vx_buffer_h* hbuffer) {
  if (nullptr == hdevice || nullptr == filename || nullptr == hbuffer)
    return -1;

  std::ifstream ifs(filename);
  if (!ifs) {
    std::cout << "error: " << filename << " not found" << std::endl;
    return -1;
  }

  // read file content
  ifs.seekg(0, ifs.end);
  auto size = ifs.tellg();
  std::vector<char> content(size);
  ifs.seekg(0, ifs.beg);
  ifs.read(content.data(), size);

  // upload buffer
  CHECK_ERR(vx_upload_kernel_bytes(hdevice, content.data(), size, hbuffer), {
    return err;
  });

  return 0;
}

extern int vx_upload_bytes(vx_device_h hdevice, const void* content, uint64_t size, vx_buffer_h* hbuffer) {
  if (nullptr == hdevice || nullptr == content || 0 == size || nullptr == hbuffer)
    return -1;

  vx_buffer_h _hbuffer;

  CHECK_ERR(vx_mem_alloc(hdevice, size, VX_MEM_READ, &_hbuffer), {
    return err;
  });

  CHECK_ERR(vx_copy_to_dev(_hbuffer, content, 0, size), {
    vx_mem_free(_hbuffer);
    return err;
  });

  *hbuffer = _hbuffer;

  return 0;
}

extern int vx_upload_file(vx_device_h hdevice, const char* filename, vx_buffer_h* hbuffer) {
  if (nullptr == hdevice || nullptr == filename || nullptr == hbuffer)
    return -1;

  std::ifstream ifs(filename);
  if (!ifs) {
    std::cout << "error: " << filename << " not found" << std::endl;
    return -1;
  }

  // read file content
  ifs.seekg(0, ifs.end);
  auto size = ifs.tellg();
  std::vector<char> content(size);
  ifs.seekg(0, ifs.beg);
  ifs.read(content.data(), size);

  // upload buffer
  CHECK_ERR(vx_upload_bytes(hdevice, content.data(), size, hbuffer), {
    return err;
  });

  return 0;
}

///////////////////////////////////////////////////////////////////////////////

extern int vx_dump_perf(vx_device_h hdevice, FILE* stream) {
  uint64_t total_instrs = 0;
  uint64_t total_cycles = 0;
  uint64_t max_cycles = 0;

  auto calcRatio = [&](uint64_t part, uint64_t total)->int {
    if (total == 0)
      return 0;
    return int((1.0 - (double(part) / double(total))) * 100);
  };

  auto caclAverage = [&](uint64_t part, uint64_t total)->double {
    if (total == 0)
      return 0;
    return double(part) / double(total);
  };

  auto calcAvgPercent = [&](uint64_t part, uint64_t total)->int {
    return int(caclAverage(part, total) * 100);
  };

  // PERF: pipeline stalls
  uint64_t sched_idles = 0;
  uint64_t sched_stalls = 0;
  uint64_t ibuffer_stalls = 0;
  uint64_t scrb_stalls = 0;
  uint64_t opds_stalls = 0;
  uint64_t scrb_alu = 0;
  uint64_t scrb_fpu = 0;
  uint64_t scrb_lsu = 0;
  uint64_t scrb_csrs = 0;
  uint64_t scrb_wctl = 0;
  uint64_t ifetches = 0;
  uint64_t loads = 0;
  uint64_t stores = 0;
  uint64_t ifetch_lat = 0;
  uint64_t load_lat   = 0;
  // PERF: l2cache
  uint64_t l2cache_reads = 0;
  uint64_t l2cache_writes = 0;
  uint64_t l2cache_read_misses = 0;
  uint64_t l2cache_write_misses = 0;
  uint64_t l2cache_bank_stalls = 0;
  uint64_t l2cache_mshr_stalls = 0;
  // PERF: l3cache
  uint64_t l3cache_reads = 0;
  uint64_t l3cache_writes = 0;
  uint64_t l3cache_read_misses = 0;
  uint64_t l3cache_write_misses = 0;
  uint64_t l3cache_bank_stalls = 0;
  uint64_t l3cache_mshr_stalls = 0;
  // PERF: memory
  uint64_t mem_reads = 0;
  uint64_t mem_writes = 0;
  uint64_t mem_lat = 0;
  uint64_t mem_bank_stalls = 0;

  uint64_t num_cores;
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_CORES, &num_cores), {
    return err;
  });

  uint64_t isa_flags;
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_ISA_FLAGS, &isa_flags), {
    return err;
  });

  uint64_t num_mem_bank_ports;
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_MEM_BANKS, &num_mem_bank_ports), {
    return err;
  });

  bool icache_enable  = isa_flags & VX_ISA_EXT_ICACHE;
  bool dcache_enable  = isa_flags & VX_ISA_EXT_DCACHE;
  bool l2cache_enable = isa_flags & VX_ISA_EXT_L2CACHE;
  bool l3cache_enable = isa_flags & VX_ISA_EXT_L3CACHE;
  bool lmem_enable    = isa_flags & VX_ISA_EXT_LMEM;

  auto perf_class = get_profiling_mode();

  for (unsigned core_id = 0; core_id < num_cores; ++core_id) {
    uint64_t cycles_per_core;
    CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MCYCLE, core_id, &cycles_per_core), {
      return err;
    });

    uint64_t instrs_per_core;
    CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MINSTRET, core_id, &instrs_per_core), {
      return err;
    });

    switch (perf_class) {
    case VX_DCR_MPM_CLASS_CORE: {
      // PERF: pipeline
      // scheduler idles
      {
        uint64_t sched_idles_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_SCHED_ID, core_id, &sched_idles_per_core), {
          return err;
        });
        if (num_cores > 1) {
          int idles_percent_per_core = calcAvgPercent(sched_idles_per_core, cycles_per_core);
          fprintf(stream, "PERF: core%d: scheduler idle=%ld (%d%%)\n", core_id, sched_idles_per_core, idles_percent_per_core);
        }
        sched_idles += sched_idles_per_core;
      }
      // scheduler stalls
      {
        uint64_t sched_stalls_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_SCHED_ST, core_id, &sched_stalls_per_core), {
          return err;
        });
        if (num_cores > 1) {
          int stalls_percent_per_core = calcAvgPercent(sched_stalls_per_core, cycles_per_core);
          fprintf(stream, "PERF: core%d: scheduler stalls=%ld (%d%%)\n", core_id, sched_stalls_per_core, stalls_percent_per_core);
        }
        sched_stalls += sched_stalls_per_core;
      }
      // ibuffer stalls
      {
        uint64_t ibuffer_stalls_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_IBUF_ST, core_id, &ibuffer_stalls_per_core), {
          return err;
        });
        if (num_cores > 1) {
          int ibuffer_percent_per_core = calcAvgPercent(ibuffer_stalls_per_core, cycles_per_core);
          fprintf(stream, "PERF: core%d: ibuffer stalls=%ld (%d%%)\n", core_id, ibuffer_stalls_per_core, ibuffer_percent_per_core);
        }
        ibuffer_stalls += ibuffer_stalls_per_core;
      }
      // scoreboard stalls
      {
        uint64_t scrb_stalls_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_SCRB_ST, core_id, &scrb_stalls_per_core), {
          return err;
        });
        uint64_t scrb_alu_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_SCRB_ALU, core_id, &scrb_alu_per_core), {
          return err;
        });
        uint64_t scrb_fpu_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_SCRB_FPU, core_id, &scrb_fpu_per_core), {
          return err;
        });
        uint64_t scrb_lsu_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_SCRB_LSU, core_id, &scrb_lsu_per_core), {
          return err;
        });
        uint64_t scrb_csrs_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_SCRB_CSRS, core_id, &scrb_csrs_per_core), {
          return err;
        });
        uint64_t scrb_wctl_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_SCRB_WCTL, core_id, &scrb_wctl_per_core), {
          return err;
        });
        scrb_alu += scrb_alu_per_core;
        scrb_fpu += scrb_fpu_per_core;
        scrb_lsu += scrb_lsu_per_core;
        scrb_csrs += scrb_csrs_per_core;
        scrb_wctl += scrb_wctl_per_core;
        if (num_cores > 1) {
          uint64_t scrb_total = scrb_alu_per_core + scrb_fpu_per_core + scrb_lsu_per_core + scrb_csrs_per_core + scrb_wctl_per_core;
          int scrb_percent_per_core = calcAvgPercent(scrb_stalls_per_core, cycles_per_core);
          fprintf(stream, "PERF: core%d: scoreboard stalls=%ld (%d%%) (alu=%d%%, fpu=%d%%, lsu=%d%%, csrs=%d%%, wctl=%d%%)\n"
          , core_id
          , scrb_stalls_per_core
          , scrb_percent_per_core
          , calcAvgPercent(scrb_alu_per_core, scrb_total)
          , calcAvgPercent(scrb_fpu_per_core, scrb_total)
          , calcAvgPercent(scrb_lsu_per_core, scrb_total)
          , calcAvgPercent(scrb_csrs_per_core, scrb_total)
          , calcAvgPercent(scrb_wctl_per_core, scrb_total)
          );
        }
        scrb_stalls += scrb_stalls_per_core;
      }
      // operands stalls
      {
        uint64_t opds_stalls_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_OPDS_ST, core_id, &opds_stalls_per_core), {
          return err;
        });
        if (num_cores > 1) {
          int opds_percent_per_core = calcAvgPercent(opds_stalls_per_core, cycles_per_core);
          fprintf(stream, "PERF: core%d: operands stalls=%ld (%d%%)\n", core_id, opds_stalls_per_core, opds_percent_per_core);
        }
        opds_stalls += opds_stalls_per_core;
      }
      // PERF: memory
      // ifetches
      {
        uint64_t ifetches_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_IFETCHES, core_id, &ifetches_per_core), {
          return err;
        });
        if (num_cores > 1) fprintf(stream, "PERF: core%d: ifetches=%ld\n", core_id, ifetches_per_core);
        ifetches += ifetches_per_core;

        uint64_t ifetch_lat_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_IFETCH_LT, core_id, &ifetch_lat_per_core), {
          return err;
        });
        if (num_cores > 1) {
          int mem_avg_lat = caclAverage(ifetch_lat_per_core, ifetches_per_core);
          fprintf(stream, "PERF: core%d: ifetch latency=%d cycles\n", core_id, mem_avg_lat);
        }
        ifetch_lat += ifetch_lat_per_core;
      }
      // loads
      {
        uint64_t loads_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_LOADS, core_id, &loads_per_core), {
          return err;
        });
        if (num_cores > 1) fprintf(stream, "PERF: core%d: loads=%ld\n", core_id, loads_per_core);
        loads += loads_per_core;

        uint64_t load_lat_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_LOAD_LT, core_id, &load_lat_per_core), {
          return err;
        });
        if (num_cores > 1) {
          int mem_avg_lat = caclAverage(load_lat_per_core, loads_per_core);
          fprintf(stream, "PERF: core%d: load latency=%d cycles\n", core_id, mem_avg_lat);
        }
        load_lat += load_lat_per_core;
      }
      // stores
      {
        uint64_t stores_per_core;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_STORES, core_id, &stores_per_core), {
          return err;
        });
        if (num_cores > 1) fprintf(stream, "PERF: core%d: stores=%ld\n", core_id, stores_per_core);
        stores += stores_per_core;
      }
    } break;
    case VX_DCR_MPM_CLASS_MEM: {
      if (lmem_enable) {
        // PERF: lmem
        uint64_t lmem_reads;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_LMEM_READS, core_id, &lmem_reads), {
          return err;
        });
        uint64_t lmem_writes;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_LMEM_WRITES, core_id, &lmem_writes), {
          return err;
        });
        uint64_t lmem_bank_stalls;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_LMEM_BANK_ST, core_id, &lmem_bank_stalls), {
          return err;
        });
        int lmem_bank_utilization = calcAvgPercent(lmem_reads + lmem_writes, lmem_reads + lmem_writes + lmem_bank_stalls);
        fprintf(stream, "PERF: core%d: lmem reads=%ld\n", core_id, lmem_reads);
        fprintf(stream, "PERF: core%d: lmem writes=%ld\n", core_id, lmem_writes);
        fprintf(stream, "PERF: core%d: lmem bank stalls=%ld (utilization=%d%%)\n", core_id, lmem_bank_stalls, lmem_bank_utilization);
      }

      if (icache_enable) {
        // PERF: Icache
        uint64_t icache_reads;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_ICACHE_READS, core_id, &icache_reads), {
          return err;
        });
        uint64_t icache_read_misses;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_ICACHE_MISS_R, core_id, &icache_read_misses), {
          return err;
        });
        uint64_t icache_mshr_stalls;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_ICACHE_MSHR_ST, core_id, &icache_mshr_stalls), {
          return err;
        });
        int icache_read_hit_ratio = calcRatio(icache_read_misses, icache_reads);
        int mshr_utilization = calcAvgPercent(icache_read_misses, icache_read_misses + icache_mshr_stalls);
        fprintf(stream, "PERF: core%d: icache reads=%ld\n", core_id, icache_reads);
        fprintf(stream, "PERF: core%d: icache read misses=%ld (hit ratio=%d%%)\n", core_id, icache_read_misses, icache_read_hit_ratio);
        fprintf(stream, "PERF: core%d: icache mshr stalls=%ld (utilization=%d%%)\n", core_id, icache_mshr_stalls, mshr_utilization);
      }

      uint64_t dcache_requests_per_core = 0;

      if (dcache_enable) {
        // PERF: Dcache
        uint64_t dcache_reads;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_DCACHE_READS, core_id, &dcache_reads), {
          return err;
        });
        uint64_t dcache_writes;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_DCACHE_WRITES, core_id, &dcache_writes), {
          return err;
        });
        dcache_requests_per_core += dcache_reads + dcache_writes;
        uint64_t dcache_read_misses;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_DCACHE_MISS_R, core_id, &dcache_read_misses), {
          return err;
        });
        uint64_t dcache_write_misses;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_DCACHE_MISS_W, core_id, &dcache_write_misses), {
          return err;
        });
        uint64_t dcache_bank_stalls;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_DCACHE_BANK_ST, core_id, &dcache_bank_stalls), {
          return err;
        });
        uint64_t dcache_mshr_stalls;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_DCACHE_MSHR_ST, core_id, &dcache_mshr_stalls), {
          return err;
        });
        int dcache_read_hit_ratio = calcRatio(dcache_read_misses, dcache_reads);
        int dcache_write_hit_ratio = calcRatio(dcache_write_misses, dcache_writes);
        int dcache_bank_utilization = calcAvgPercent(dcache_reads + dcache_writes, dcache_reads + dcache_writes + dcache_bank_stalls);
        int mshr_utilization = calcAvgPercent(dcache_read_misses + dcache_write_misses, dcache_read_misses + dcache_write_misses + dcache_mshr_stalls);
        fprintf(stream, "PERF: core%d: dcache reads=%ld\n", core_id, dcache_reads);
        fprintf(stream, "PERF: core%d: dcache writes=%ld\n", core_id, dcache_writes);
        fprintf(stream, "PERF: core%d: dcache read misses=%ld (hit ratio=%d%%)\n", core_id, dcache_read_misses, dcache_read_hit_ratio);
        fprintf(stream, "PERF: core%d: dcache write misses=%ld (hit ratio=%d%%)\n", core_id, dcache_write_misses, dcache_write_hit_ratio);
        fprintf(stream, "PERF: core%d: dcache bank stalls=%ld (utilization=%d%%)\n", core_id, dcache_bank_stalls, dcache_bank_utilization);
        fprintf(stream, "PERF: core%d: dcache mshr stalls=%ld (utilization=%d%%)\n", core_id, dcache_mshr_stalls, mshr_utilization);
      }

      // PERF: coalescer
      uint64_t coalescer_misses;
      CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_COALESCER_MISS, core_id, &coalescer_misses), {
        return err;
      });
      int coalescer_utilization = calcAvgPercent(dcache_requests_per_core - coalescer_misses, dcache_requests_per_core);
      fprintf(stream, "PERF: core%d: coalescer misses=%ld (hit ratio=%d%%)\n", core_id, coalescer_misses, coalescer_utilization);

      if (l2cache_enable) {
        // PERF: L2cache
        uint64_t tmp;
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L2CACHE_READS, core_id, &tmp), {
          return err;
        });
        l2cache_reads += tmp;

        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L2CACHE_WRITES, core_id, &tmp), {
          return err;
        });
        l2cache_writes += tmp;

        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L2CACHE_MISS_R, core_id, &tmp), {
          return err;
        });
        l2cache_read_misses += tmp;

        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L2CACHE_MISS_W, core_id, &tmp), {
          return err;
        });
        l2cache_write_misses += tmp;

        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L2CACHE_BANK_ST, core_id, &tmp), {
          return err;
        });
        l2cache_bank_stalls += tmp;

        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L2CACHE_MSHR_ST, core_id, &tmp), {
          return err;
        });
        l2cache_mshr_stalls += tmp;
      }
      if (0 == core_id) {
        if (l3cache_enable) {
          // PERF: L3cache
          CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L3CACHE_READS, core_id, &l3cache_reads), {
            return err;
          });
          CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L3CACHE_WRITES, core_id, &l3cache_writes), {
            return err;
          });
          CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L3CACHE_MISS_R, core_id, &l3cache_read_misses), {
            return err;
          });
          CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L3CACHE_MISS_W, core_id, &l3cache_write_misses), {
            return err;
          });
          CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L3CACHE_BANK_ST, core_id, &l3cache_bank_stalls), {
            return err;
          });
          CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_L3CACHE_MSHR_ST, core_id, &l3cache_mshr_stalls), {
            return err;
          });
        }
        // PERF: memory
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_MEM_READS, core_id, &mem_reads), {
          return err;
        });
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_MEM_WRITES, core_id, &mem_writes), {
          return err;
        });
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_MEM_LT, core_id, &mem_lat), {
          return err;
        });
        CHECK_ERR(vx_mpm_query(hdevice, VX_CSR_MPM_MEM_BANK_ST, core_id, &mem_bank_stalls), {
          return err;
        });
      }
    } break;
    default:
      break;
    }

    float IPC = caclAverage(instrs_per_core, cycles_per_core);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: instrs=%ld, cycles=%ld, IPC=%f\n", core_id, instrs_per_core, cycles_per_core, IPC);
    total_instrs += instrs_per_core;
    total_cycles += cycles_per_core;
    max_cycles = std::max<uint64_t>(cycles_per_core, max_cycles);
  }

  switch (perf_class) {
  case VX_DCR_MPM_CLASS_CORE: {
    int sched_idles_percent = calcAvgPercent(sched_idles, total_cycles);
    int sched_stalls_percent = calcAvgPercent(sched_stalls, total_cycles);
    int ibuffer_percent = calcAvgPercent(ibuffer_stalls, total_cycles);
    int scrb_percent = calcAvgPercent(scrb_stalls, total_cycles);
    int opds_percent = calcAvgPercent(opds_stalls, total_cycles);
    int ifetch_avg_lat = caclAverage(ifetch_lat, ifetches);
    int load_avg_lat = caclAverage(load_lat, loads);
    uint64_t scrb_total = scrb_alu + scrb_fpu + scrb_lsu + scrb_csrs + scrb_wctl;
    fprintf(stream, "PERF: scheduler idle=%ld (%d%%)\n", sched_idles, sched_idles_percent);
    fprintf(stream, "PERF: scheduler stalls=%ld (%d%%)\n", sched_stalls, sched_stalls_percent);
    fprintf(stream, "PERF: ibuffer stalls=%ld (%d%%)\n", ibuffer_stalls, ibuffer_percent);
    fprintf(stream, "PERF: scoreboard stalls=%ld (%d%%) (alu=%d%%, fpu=%d%%, lsu=%d%%, csrs=%d%%, wctl=%d%%)\n"
      , scrb_stalls
      , scrb_percent
      , calcAvgPercent(scrb_alu, scrb_total)
      , calcAvgPercent(scrb_fpu, scrb_total)
      , calcAvgPercent(scrb_lsu, scrb_total)
      , calcAvgPercent(scrb_csrs, scrb_total)
      , calcAvgPercent(scrb_wctl, scrb_total)
    );
    fprintf(stream, "PERF: operands stalls=%ld (%d%%)\n", opds_stalls, opds_percent);
    fprintf(stream, "PERF: ifetches=%ld\n", ifetches);
    fprintf(stream, "PERF: loads=%ld\n", loads);
    fprintf(stream, "PERF: stores=%ld\n", stores);
    fprintf(stream, "PERF: ifetch latency=%d cycles\n", ifetch_avg_lat);
    fprintf(stream, "PERF: load latency=%d cycles\n", load_avg_lat);
  } break;
  case VX_DCR_MPM_CLASS_MEM: {
    if (l2cache_enable) {
      l2cache_reads /= num_cores;
      l2cache_writes /= num_cores;
      l2cache_read_misses /= num_cores;
      l2cache_write_misses /= num_cores;
      l2cache_bank_stalls /= num_cores;
      l2cache_mshr_stalls /= num_cores;
      int read_hit_ratio = calcRatio(l2cache_read_misses, l2cache_reads);
      int write_hit_ratio = calcRatio(l2cache_write_misses, l2cache_writes);
      int bank_utilization = calcAvgPercent(l2cache_reads + l2cache_writes, l2cache_reads + l2cache_writes + l2cache_bank_stalls);
      int mshr_utilization = calcAvgPercent(l2cache_read_misses + l2cache_write_misses, l2cache_read_misses + l2cache_write_misses + l2cache_mshr_stalls);
      fprintf(stream, "PERF: l2cache reads=%ld\n", l2cache_reads);
      fprintf(stream, "PERF: l2cache writes=%ld\n", l2cache_writes);
      fprintf(stream, "PERF: l2cache read misses=%ld (hit ratio=%d%%)\n", l2cache_read_misses, read_hit_ratio);
      fprintf(stream, "PERF: l2cache write misses=%ld (hit ratio=%d%%)\n", l2cache_write_misses, write_hit_ratio);
      fprintf(stream, "PERF: l2cache bank stalls=%ld (utilization=%d%%)\n", l2cache_bank_stalls, bank_utilization);
      fprintf(stream, "PERF: l2cache mshr stalls=%ld (utilization=%d%%)\n", l2cache_mshr_stalls, mshr_utilization);
    }

    if (l3cache_enable) {
      int read_hit_ratio = calcRatio(l3cache_read_misses, l3cache_reads);
      int write_hit_ratio = calcRatio(l3cache_write_misses, l3cache_writes);
      int bank_utilization = calcAvgPercent(l3cache_reads + l3cache_writes, l3cache_reads + l3cache_writes + l3cache_bank_stalls);
      int mshr_utilization = calcAvgPercent(l3cache_read_misses + l3cache_write_misses, l3cache_read_misses + l3cache_write_misses + l3cache_mshr_stalls);
      fprintf(stream, "PERF: l3cache reads=%ld\n", l3cache_reads);
      fprintf(stream, "PERF: l3cache writes=%ld\n", l3cache_writes);
      fprintf(stream, "PERF: l3cache read misses=%ld (hit ratio=%d%%)\n", l3cache_read_misses, read_hit_ratio);
      fprintf(stream, "PERF: l3cache write misses=%ld (hit ratio=%d%%)\n", l3cache_write_misses, write_hit_ratio);
      fprintf(stream, "PERF: l3cache bank stalls=%ld (utilization=%d%%)\n", l3cache_bank_stalls, bank_utilization);
      fprintf(stream, "PERF: l3cache mshr stalls=%ld (utilization=%d%%)\n", l3cache_mshr_stalls, mshr_utilization);
    }

    {
      uint64_t mem_requests = mem_reads + mem_writes;
      int mem_avg_lat = caclAverage(mem_lat, mem_reads);
      int mem_bank_utilization = calcAvgPercent(mem_requests, mem_requests + mem_bank_stalls);
      fprintf(stream, "PERF: memory requests=%ld (reads=%ld, writes=%ld)\n", mem_requests, mem_reads, mem_writes);
      fprintf(stream, "PERF: memory latency=%d cycles\n", mem_avg_lat);
      fprintf(stream, "PERF: memory bank stalls=%ld (utilization=%d%%)\n", mem_bank_stalls, mem_bank_utilization);
    }
  } break;
  default:
    break;
  }

  float IPC = caclAverage(total_instrs, max_cycles);
  fprintf(stream, "PERF: instrs=%ld, cycles=%ld, IPC=%f\n", total_instrs, max_cycles, IPC);

  fflush(stream);

  return 0;
}

int vx_check_occupancy(vx_device_h hdevice, uint32_t group_size, uint32_t* max_localmem) {
   // check group size
  uint64_t warps_per_core, threads_per_warp;
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_WARPS, &warps_per_core), {
    return err;
  });
  CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_NUM_THREADS, &threads_per_warp), {
    return err;
  });
  uint32_t threads_per_core = warps_per_core * threads_per_warp;
  if (group_size > threads_per_core) {
    printf("Error: cannot schedule kernel with group_size > threads_per_core (%d,%d)\n", group_size, threads_per_core);
    return -1;
  }

  // calculate groups occupancy
  int warps_per_group = (group_size + threads_per_warp-1) / threads_per_warp;
  int groups_per_core = warps_per_core / warps_per_group;

  // check local memory capacity
  if (max_localmem) {
    uint64_t local_mem_size;
    CHECK_ERR(vx_dev_caps(hdevice, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size), {
      return err;
    });
    *max_localmem = local_mem_size / groups_per_core;
  }

  return 0;
}