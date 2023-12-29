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

#include "utils.h"
#include <iostream>
#include <fstream>
#include <list>
#include <cstring>
#include <vector>
#include <vortex.h>
#include <assert.h>

#define RT_CHECK(_expr, _cleanup)                               \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     _cleanup                                                   \
   } while (false)

uint64_t aligned_size(uint64_t size, uint64_t alignment) {        
    assert(0 == (alignment & (alignment - 1)));
    return (size + alignment - 1) & ~(alignment - 1);
}

bool is_aligned(uint64_t addr, uint64_t alignment) {
    assert(0 == (alignment & (alignment - 1)));
    return 0 == (addr & (alignment - 1));
}

///////////////////////////////////////////////////////////////////////////////

class AutoPerfDump {
public:
    AutoPerfDump() : perf_class_(0) {}

    ~AutoPerfDump() {
      for (auto hdevice : hdevices_) {
        vx_dump_perf(hdevice, stdout);
      }
    }

    void add_device(vx_device_h hdevice) {
      auto perf_class_s = getenv("PERF_CLASS");
      if (perf_class_s) {
        perf_class_ = std::atoi(perf_class_s);
        vx_dcr_write(hdevice, VX_DCR_BASE_MPM_CLASS, perf_class_);
      }
      hdevices_.push_back(hdevice);
    }

    void remove_device(vx_device_h hdevice) {
      hdevices_.remove(hdevice);
      vx_dump_perf(hdevice, stdout);
    }

    int get_perf_class() const {
      return perf_class_;
    }
    
private:
    std::list<vx_device_h> hdevices_;
    int perf_class_;
};

#ifdef DUMP_PERF_STATS
AutoPerfDump gAutoPerfDump;
#endif

void perf_add_device(vx_device_h hdevice) {
#ifdef DUMP_PERF_STATS
  gAutoPerfDump.add_device(hdevice);
#else
  (void)hdevice;
#endif
}

void perf_remove_device(vx_device_h hdevice) {
#ifdef DUMP_PERF_STATS
  gAutoPerfDump.remove_device(hdevice);
#else
  (void)hdevice;
#endif
}

///////////////////////////////////////////////////////////////////////////////

extern int vx_upload_kernel_bytes(vx_device_h hdevice, const void* content, uint64_t size) {
  int err = 0;

  if (NULL == content || 0 == size)
    return -1;

  uint64_t kernel_base_addr;
  err = vx_dev_caps(hdevice, VX_CAPS_KERNEL_BASE_ADDR, &kernel_base_addr);
  if (err != 0)
    return err;

  return vx_copy_to_dev(hdevice, kernel_base_addr, content, size);
}

extern int vx_upload_kernel_file(vx_device_h hdevice, const char* filename) {
  std::ifstream ifs(filename);
  if (!ifs) {
    std::cout << "error: " << filename << " not found" << std::endl;
    return -1;
  }

  // read file content
  ifs.seekg(0, ifs.end);
  auto size = ifs.tellg();
  auto content = new char [size];   
  ifs.seekg(0, ifs.beg);
  ifs.read(content, size);

  // upload
  int err = vx_upload_kernel_bytes(hdevice, content, size);

  // release buffer
  delete[] content;

  return err;
}

///////////////////////////////////////////////////////////////////////////////

void DeviceConfig::write(uint32_t addr, uint32_t value) {
  data_[addr] = value;
}

uint32_t DeviceConfig::read(uint32_t addr) const {
  if (0 == data_.count(addr)) {
    printf("Error: DeviceConfig::read(%d) failed\n", addr);
  }
  return data_.at(addr);
}

int dcr_initialize(vx_device_h hdevice) {
  const uint64_t startup_addr(STARTUP_ADDR);
  RT_CHECK(vx_dcr_write(hdevice, VX_DCR_BASE_STARTUP_ADDR0, startup_addr & 0xffffffff), {
    return -1;
  });

  RT_CHECK(vx_dcr_write(hdevice, VX_DCR_BASE_STARTUP_ADDR1, startup_addr >> 32), {
    return -1;
  });

  RT_CHECK(vx_dcr_write(hdevice, VX_DCR_BASE_MPM_CLASS, 0), {
    return -1;
  });
  
  return 0;
}

///////////////////////////////////////////////////////////////////////////////

static uint64_t get_csr_64(const void* ptr, int addr) {
  auto w_ptr = reinterpret_cast<const uint32_t*>(ptr);
  uint32_t value_lo = w_ptr[addr - VX_CSR_MPM_BASE];
  uint32_t value_hi = w_ptr[addr - VX_CSR_MPM_BASE + 32];
  return (uint64_t(value_hi) << 32) | value_lo;
}

extern int vx_dump_perf(vx_device_h hdevice, FILE* stream) {
  int ret = 0;

  uint64_t total_instrs = 0;
  uint64_t total_cycles = 0;
  uint64_t max_cycles = 0;

#ifdef PERF_ENABLE

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

  auto perf_class = gAutoPerfDump.get_perf_class();

  // PERF: pipeline stalls
  uint64_t sched_idles = 0;
  uint64_t sched_stalls = 0;
  uint64_t ibuffer_stalls = 0;
  uint64_t scrb_stalls = 0;
  uint64_t scrb_alu = 0;
  uint64_t scrb_fpu = 0;
  uint64_t scrb_lsu = 0;
  uint64_t scrb_sfu = 0;
  uint64_t scrb_wctl = 0;
  uint64_t scrb_csrs = 0;
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
#endif

  uint64_t num_cores;
  ret = vx_dev_caps(hdevice, VX_CAPS_NUM_CORES, &num_cores);
  if (ret != 0)
    return ret;

#ifdef PERF_ENABLE
  uint64_t isa_flags;
  ret = vx_dev_caps(hdevice, VX_CAPS_ISA_FLAGS, &isa_flags);
  if (ret != 0)
    return ret;

  bool icache_enable  = isa_flags & VX_ISA_EXT_ICACHE;
  bool dcache_enable  = isa_flags & VX_ISA_EXT_DCACHE;
  bool l2cache_enable = isa_flags & VX_ISA_EXT_L2CACHE;
  bool l3cache_enable = isa_flags & VX_ISA_EXT_L3CACHE;
  bool smem_enable    = isa_flags & VX_ISA_EXT_SMEM;
#endif

  std::vector<uint8_t> staging_buf(64* sizeof(uint32_t));

  for (unsigned core_id = 0; core_id < num_cores; ++core_id) {
    uint64_t mpm_mem_addr = IO_CSR_ADDR + core_id * staging_buf.size();    
    ret = vx_copy_from_dev(hdevice, staging_buf.data(), mpm_mem_addr, staging_buf.size());
    if (ret != 0)
      return ret;

    uint64_t cycles_per_core = get_csr_64(staging_buf.data(), VX_CSR_MCYCLE);
    uint64_t instrs_per_core = get_csr_64(staging_buf.data(), VX_CSR_MINSTRET);    

  #ifdef PERF_ENABLE
    switch (perf_class) {
    case VX_DCR_MPM_CLASS_CORE: {
      // PERF: pipeline    
      // scheduler idles
      {
        uint64_t sched_idles_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCHED_ID);        
        if (num_cores > 1) {
          int idles_percent_per_core = calcAvgPercent(sched_idles_per_core, cycles_per_core);
          fprintf(stream, "PERF: core%d: scheduler idle=%ld (%d%%)\n", core_id, sched_idles_per_core, idles_percent_per_core);
        }
        sched_idles += sched_idles_per_core;
      }
      // scheduler stalls
      {
        uint64_t sched_stalls_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCHED_ST);        
        if (num_cores > 1) {
          int stalls_percent_per_core = calcAvgPercent(sched_stalls_per_core, cycles_per_core);
          fprintf(stream, "PERF: core%d: scheduler stalls=%ld (%d%%)\n", core_id, sched_stalls_per_core, stalls_percent_per_core);
        }
        sched_stalls += sched_stalls_per_core;
      }
      // ibuffer_stalls
      {
        uint64_t ibuffer_stalls_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_IBUF_ST);        
        if (num_cores > 1) {
          int ibuffer_percent_per_core = calcAvgPercent(ibuffer_stalls_per_core, cycles_per_core);
          fprintf(stream, "PERF: core%d: ibuffer stalls=%ld (%d%%)\n", core_id, ibuffer_stalls_per_core, ibuffer_percent_per_core);
        }
        ibuffer_stalls += ibuffer_stalls_per_core;
      }
      // issue_stalls
      {
        uint64_t scrb_stalls_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCRB_ST);
        uint64_t scrb_alu_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCRB_ALU);
        uint64_t scrb_fpu_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCRB_FPU);
        uint64_t scrb_lsu_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCRB_LSU);
        uint64_t scrb_sfu_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCRB_SFU);        
        scrb_alu += scrb_alu_per_core;
        scrb_fpu += scrb_fpu_per_core;
        scrb_lsu += scrb_lsu_per_core;
        scrb_sfu += scrb_sfu_per_core;      
        if (num_cores > 1) {
          uint64_t scrb_total = scrb_alu_per_core + scrb_fpu_per_core + scrb_lsu_per_core + scrb_sfu_per_core;
          fprintf(stream, "PERF: core%d: issue stalls=%ld (alu=%d%%, fpu=%d%%, lsu=%d%%, sfu=%d%%)\n", core_id, scrb_stalls_per_core, 
          calcAvgPercent(scrb_alu_per_core, scrb_total), 
          calcAvgPercent(scrb_fpu_per_core, scrb_total),
          calcAvgPercent(scrb_lsu_per_core, scrb_total),
          calcAvgPercent(scrb_sfu_per_core, scrb_total));
        }
        scrb_stalls += scrb_stalls_per_core;
      }
      // sfu_stalls
      {
        uint64_t scrb_sfu_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCRB_SFU);  
        uint64_t scrb_wctl_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCRB_WCTL);
        uint64_t scrb_csrs_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCRB_CSRS);
        if (num_cores > 1) {
          uint64_t sfu_total = scrb_wctl_per_core + scrb_csrs_per_core;
          fprintf(stream, "PERF: core%d: sfu stalls=%ld (scrs=%d%%, wctl=%d%%)\n"
            , core_id
            , scrb_sfu_per_core            
            , calcAvgPercent(scrb_csrs_per_core, sfu_total)
            , calcAvgPercent(scrb_wctl_per_core, sfu_total)
          );
        }
        scrb_wctl += scrb_wctl_per_core;
        scrb_csrs += scrb_csrs_per_core;
      }
      // PERF: memory
      // ifetches
      {
        uint64_t ifetches_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_IFETCHES);
        if (num_cores > 1) fprintf(stream, "PERF: core%d: ifetches=%ld\n", core_id, ifetches_per_core);
        ifetches += ifetches_per_core;

        uint64_t ifetch_lat_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_IFETCH_LT);        
        if (num_cores > 1) {
          int mem_avg_lat = caclAverage(ifetch_lat_per_core, ifetches_per_core);
          fprintf(stream, "PERF: core%d: ifetch latency=%d cycles\n", core_id, mem_avg_lat);
        }
        ifetch_lat += ifetch_lat_per_core;
      }
      // loads
      {
        uint64_t loads_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_LOADS);
        if (num_cores > 1) fprintf(stream, "PERF: core%d: loads=%ld\n", core_id, loads_per_core);
        loads += loads_per_core;

        uint64_t load_lat_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_LOAD_LT);        
        if (num_cores > 1) {
          int mem_avg_lat = caclAverage(load_lat_per_core, loads_per_core);
          fprintf(stream, "PERF: core%d: load latency=%d cycles\n", core_id, mem_avg_lat);
        }
        load_lat += load_lat_per_core;
      }
      // stores
      {
        uint64_t stores_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_STORES);
        if (num_cores > 1) fprintf(stream, "PERF: core%d: stores=%ld\n", core_id, stores_per_core);
        stores += stores_per_core;
      }
    } break;
    case VX_DCR_MPM_CLASS_MEM: { 
      if (smem_enable) {
        // PERF: smem
        uint64_t smem_reads = get_csr_64(staging_buf.data(), VX_CSR_MPM_SMEM_READS);
        uint64_t smem_writes = get_csr_64(staging_buf.data(), VX_CSR_MPM_SMEM_WRITES);
        uint64_t smem_bank_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_SMEM_BANK_ST);
        int smem_bank_utilization = calcAvgPercent(smem_reads + smem_writes, smem_reads + smem_writes + smem_bank_stalls);
        fprintf(stream, "PERF: core%d: smem reads=%ld\n", core_id, smem_reads);
        fprintf(stream, "PERF: core%d: smem writes=%ld\n", core_id, smem_writes); 
        fprintf(stream, "PERF: core%d: smem bank stalls=%ld (utilization=%d%%)\n", core_id, smem_bank_stalls, smem_bank_utilization);
      }

      if (icache_enable) {
        // PERF: Icache
        uint64_t icache_reads = get_csr_64(staging_buf.data(), VX_CSR_MPM_ICACHE_READS);
        uint64_t icache_read_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_ICACHE_MISS_R);
        uint64_t icache_mshr_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_ICACHE_MSHR_ST);
        int icache_read_hit_ratio = calcRatio(icache_read_misses, icache_reads); 
        int mshr_utilization = calcAvgPercent(icache_read_misses, icache_read_misses + icache_mshr_stalls);
        fprintf(stream, "PERF: core%d: icache reads=%ld\n", core_id, icache_reads);
        fprintf(stream, "PERF: core%d: icache read misses=%ld (hit ratio=%d%%)\n", core_id, icache_read_misses, icache_read_hit_ratio);
        fprintf(stream, "PERF: core%d: icache mshr stalls=%ld (utilization=%d%%)\n", core_id, icache_mshr_stalls, mshr_utilization);
      }
      
      if (dcache_enable) {
        // PERF: Dcache
        uint64_t dcache_reads = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_READS);
        uint64_t dcache_writes = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_WRITES);
        uint64_t dcache_read_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_MISS_R);
        uint64_t dcache_write_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_MISS_W);
        uint64_t dcache_bank_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_BANK_ST);
        uint64_t dcache_mshr_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_MSHR_ST);
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

      if (l2cache_enable) {
        // PERF: L2cache
        l2cache_reads += get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_READS);
        l2cache_writes += get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_WRITES);
        l2cache_read_misses += get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_MISS_R);
        l2cache_write_misses += get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_MISS_W);
        l2cache_bank_stalls += get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_BANK_ST);
        l2cache_mshr_stalls += get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_MSHR_ST);
      }

      if (0 == core_id) {      
        if (l3cache_enable) {
          // PERF: L3cache
          l3cache_reads = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_READS);
          l3cache_writes = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_WRITES);
          l3cache_read_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_MISS_R);
          l3cache_write_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_MISS_W);
          l3cache_bank_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_BANK_ST);
          l3cache_mshr_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_MSHR_ST);
        }
      
        // PERF: memory
        mem_reads  = get_csr_64(staging_buf.data(), VX_CSR_MPM_MEM_READS);
        mem_writes = get_csr_64(staging_buf.data(), VX_CSR_MPM_MEM_WRITES);
        mem_lat    = get_csr_64(staging_buf.data(), VX_CSR_MPM_MEM_LT);
      }
    } break;
    default:
      break;
    }
  #endif 

    float IPC = (float)(double(instrs_per_core) / double(cycles_per_core));
    if (num_cores > 1) fprintf(stream, "PERF: core%d: instrs=%ld, cycles=%ld, IPC=%f\n", core_id, instrs_per_core, cycles_per_core, IPC);            
    total_instrs += instrs_per_core;
    total_cycles += cycles_per_core;
    max_cycles = std::max<uint64_t>(cycles_per_core, max_cycles);
  }
      
#ifdef PERF_ENABLE
  switch (perf_class) {
  case VX_DCR_MPM_CLASS_CORE: {    
    int sched_idles_percent = calcAvgPercent(sched_idles, total_cycles);
    int sched_stalls_percent = calcAvgPercent(sched_stalls, total_cycles);
    int ibuffer_percent = calcAvgPercent(ibuffer_stalls, total_cycles);
    int ifetch_avg_lat = (int)(double(ifetch_lat) / double(ifetches));
    int load_avg_lat = (int)(double(load_lat) / double(loads));
    uint64_t scrb_total = scrb_alu + scrb_fpu + scrb_lsu + scrb_sfu;
    uint64_t sfu_total = scrb_wctl + scrb_csrs;
    fprintf(stream, "PERF: scheduler idle=%ld (%d%%)\n", sched_idles, sched_idles_percent);
    fprintf(stream, "PERF: scheduler stalls=%ld (%d%%)\n", sched_stalls, sched_stalls_percent);
    fprintf(stream, "PERF: ibuffer stalls=%ld (%d%%)\n", ibuffer_stalls, ibuffer_percent);
    fprintf(stream, "PERF: issue stalls=%ld (alu=%d%%, fpu=%d%%, lsu=%d%%, sfu=%d%%)\n", scrb_stalls,
      calcAvgPercent(scrb_alu, scrb_total), 
      calcAvgPercent(scrb_fpu, scrb_total),
      calcAvgPercent(scrb_lsu, scrb_total),
      calcAvgPercent(scrb_sfu, scrb_total));    
    fprintf(stream, "PERF: sfu stalls=%ld (scrs=%d%%, wctl=%d%%)\n"
      , scrb_sfu      
      , calcAvgPercent(scrb_csrs, sfu_total)
      , calcAvgPercent(scrb_wctl, sfu_total)
    );
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

    int mem_avg_lat = caclAverage(mem_lat, mem_reads);   
    fprintf(stream, "PERF: memory requests=%ld (reads=%ld, writes=%ld)\n", (mem_reads + mem_writes), mem_reads, mem_writes);
    fprintf(stream, "PERF: memory latency=%d cycles\n", mem_avg_lat);
  } break;
  default:
    break;
  }
#endif 
  
  float IPC = (float)(double(total_instrs) / double(max_cycles));
  fprintf(stream, "PERF: instrs=%ld, cycles=%ld, IPC=%f\n", total_instrs, max_cycles, IPC);  

  fflush(stream);

  return 0;
}

extern int vx_perf_counter(vx_device_h hdevice, int counter, int core_id, uint64_t* value) {
  int ret = 0;
  uint64_t num_cores;
  ret = vx_dev_caps(hdevice, VX_CAPS_NUM_CORES, &num_cores);
  if (ret != 0)
    return ret;

  if (core_id >= (int)num_cores) {
    std::cout << "error: core_id out of range" << std::endl;
    return -1;
  }

  std::vector<uint8_t> staging_buf(64 * sizeof(uint32_t));

  uint64_t _value = 0;
  
  unsigned i = 0;
  if (core_id != -1) {
    i = core_id;
    num_cores = core_id + 1;
  }
      
  for (i = 0; i < num_cores; ++i) {
    uint64_t mpm_mem_addr = IO_CSR_ADDR + i * staging_buf.size();    
    ret = vx_copy_from_dev(hdevice, staging_buf.data(), mpm_mem_addr, staging_buf.size());
    if (ret != 0)
      return ret;

    auto per_core_value = get_csr_64(staging_buf.data(), counter);     
    if (counter == VX_CSR_MCYCLE) {
      _value = std::max<uint64_t>(per_core_value, _value);
    } else {
      _value += per_core_value;
    }    
  }

  // output
  *value = _value;

  return 0;
}
