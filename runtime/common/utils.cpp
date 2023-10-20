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

  uint64_t instrs = 0;
  uint64_t cycles = 0;

#ifdef PERF_ENABLE   
  auto perf_class = gAutoPerfDump.get_perf_class();

  // PERF: pipeline stalls
  uint64_t ibuffer_stalls = 0;
  uint64_t scoreboard_stalls = 0;
  uint64_t lsu_stalls = 0;
  uint64_t fpu_stalls = 0;
  uint64_t alu_stalls = 0;
  uint64_t sfu_stalls = 0;  
  uint64_t ifetches = 0;
  uint64_t loads = 0;
  uint64_t stores = 0;
  uint64_t ifetch_lat = 0;
  uint64_t load_lat   = 0;
  // PERF: Icache 
  uint64_t icache_reads = 0;
  uint64_t icache_read_misses = 0;
  // PERF: Dcache 
  uint64_t dcache_reads = 0;
  uint64_t dcache_writes = 0;
  uint64_t dcache_read_misses = 0;
  uint64_t dcache_write_misses = 0;
  uint64_t dcache_bank_stalls = 0; 
  uint64_t dcache_mshr_stalls = 0;
  // PERF: shared memory
  uint64_t smem_reads = 0;
  uint64_t smem_writes = 0;
  uint64_t smem_bank_stalls = 0;
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

  std::vector<uint8_t> staging_buf(64* sizeof(uint32_t));
      
  for (unsigned core_id = 0; core_id < num_cores; ++core_id) {    
    uint64_t mpm_mem_addr = IO_CSR_ADDR + core_id * staging_buf.size();    
    ret = vx_copy_from_dev(hdevice, staging_buf.data(), mpm_mem_addr, staging_buf.size());
    if (ret != 0)
      return ret;

    uint64_t instrs_per_core = get_csr_64(staging_buf.data(), VX_CSR_MINSTRET);
    uint64_t cycles_per_core = get_csr_64(staging_buf.data(), VX_CSR_MCYCLE);
    float IPC = (float)(double(instrs_per_core) / double(cycles_per_core));
    if (num_cores > 1) fprintf(stream, "PERF: core%d: instrs=%ld, cycles=%ld, IPC=%f\n", core_id, instrs_per_core, cycles_per_core, IPC);            
    instrs += instrs_per_core;
    cycles = std::max<uint64_t>(cycles_per_core, cycles);

  #ifdef PERF_ENABLE
    switch (perf_class) {
    case VX_DCR_MPM_CLASS_CORE: {
      // PERF: pipeline    
      // ibuffer_stall
      uint64_t ibuffer_stalls_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_IBUF_ST);
      if (num_cores > 1) fprintf(stream, "PERF: core%d: ibuffer stalls=%ld\n", core_id, ibuffer_stalls_per_core);
      ibuffer_stalls += ibuffer_stalls_per_core;
      // scoreboard_stall
      uint64_t scoreboard_stalls_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SCRB_ST);
      if (num_cores > 1) fprintf(stream, "PERF: core%d: scoreboard stalls=%ld\n", core_id, scoreboard_stalls_per_core);
      scoreboard_stalls += scoreboard_stalls_per_core;
      // alu_stall
      uint64_t alu_stalls_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_ALU_ST);
      if (num_cores > 1) fprintf(stream, "PERF: core%d: alu unit stalls=%ld\n", core_id, alu_stalls_per_core);
      alu_stalls += alu_stalls_per_core;      
      // lsu_stall
      uint64_t lsu_stalls_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_LSU_ST);
      if (num_cores > 1) fprintf(stream, "PERF: core%d: lsu unit stalls=%ld\n", core_id, lsu_stalls_per_core);
      lsu_stalls += lsu_stalls_per_core;
      // fpu_stall
      uint64_t fpu_stalls_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_FPU_ST);
      if (num_cores > 1) fprintf(stream, "PERF: core%d: fpu unit stalls=%ld\n", core_id, fpu_stalls_per_core);
      fpu_stalls += fpu_stalls_per_core;      
      // sfu_stall
      uint64_t sfu_stalls_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_SFU_ST);
      if (num_cores > 1) fprintf(stream, "PERF: core%d: sfu unit stalls=%ld\n", core_id, sfu_stalls_per_core);
      sfu_stalls += sfu_stalls_per_core;
      // PERF: memory
      // ifetches
      uint64_t ifetches_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_LOADS);
      if (num_cores > 1) fprintf(stream, "PERF: core%d: ifetches=%ld\n", core_id, ifetches_per_core);
      ifetches += ifetches_per_core;
      // loads
      uint64_t loads_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_LOADS);
      if (num_cores > 1) fprintf(stream, "PERF: core%d: loads=%ld\n", core_id, loads_per_core);
      loads += loads_per_core;
      // stores
      uint64_t stores_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_STORES);
      if (num_cores > 1) fprintf(stream, "PERF: core%d: stores=%ld\n", core_id, stores_per_core);
      stores += stores_per_core;
      // ifetch latency
      uint64_t ifetch_lat_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_IFETCH_LAT);
      if (num_cores > 1) {
        int mem_avg_lat = (int)(double(ifetch_lat_per_core) / double(ifetches_per_core));
        fprintf(stream, "PERF: core%d: ifetch latency=%d cycles\n", core_id, mem_avg_lat);
      }
      ifetch_lat += ifetch_lat_per_core;
      // load latency
      uint64_t load_lat_per_core = get_csr_64(staging_buf.data(), VX_CSR_MPM_LOAD_LAT);
      if (num_cores > 1) {
        int mem_avg_lat = (int)(double(load_lat_per_core) / double(loads_per_core));
        fprintf(stream, "PERF: core%d: load latency=%d cycles\n", core_id, mem_avg_lat);
      }
      load_lat += load_lat_per_core;      
    } break;
    case VX_DCR_MPM_CLASS_MEM: {      
      if (0 == core_id) {
        // PERF: Icache
        icache_reads = get_csr_64(staging_buf.data(), VX_CSR_MPM_ICACHE_READS);
        icache_read_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_ICACHE_MISS_R);
      
        // PERF: Dcache
        dcache_reads = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_READS);
        dcache_writes = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_WRITES);
        dcache_read_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_MISS_R);
        dcache_write_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_MISS_W);
        dcache_bank_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_BANK_ST);
        dcache_mshr_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_DCACHE_MSHR_ST);
      
        // PERF: smem
        smem_reads = get_csr_64(staging_buf.data(), VX_CSR_MPM_SMEM_READS);
        smem_writes = get_csr_64(staging_buf.data(), VX_CSR_MPM_SMEM_WRITES);
        smem_bank_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_SMEM_BANK_ST);
      
        // PERF: L2cache
        l2cache_reads = get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_READS);
        l2cache_writes = get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_WRITES);
        l2cache_read_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_MISS_R);
        l2cache_write_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_MISS_W);
        l2cache_bank_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_BANK_ST);
        l2cache_mshr_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_L2CACHE_MSHR_ST);
      
        // PERF: L3cache
        l3cache_reads = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_READS);
        l3cache_writes = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_WRITES);
        l3cache_read_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_MISS_R);
        l3cache_write_misses = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_MISS_W);
        l3cache_bank_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_BANK_ST);
        l3cache_mshr_stalls = get_csr_64(staging_buf.data(), VX_CSR_MPM_L3CACHE_MSHR_ST);
      
        // PERF: memory
        mem_reads  = get_csr_64(staging_buf.data(), VX_CSR_MPM_MEM_READS);
        mem_writes = get_csr_64(staging_buf.data(), VX_CSR_MPM_MEM_WRITES);
        mem_lat    = get_csr_64(staging_buf.data(), VX_CSR_MPM_MEM_LAT);
      }
    } break;
    default:
      break;
    }
  #endif
  }  
  
  float IPC = (float)(double(instrs) / double(cycles));
  fprintf(stream, "PERF: instrs=%ld, cycles=%ld, IPC=%f\n", instrs, cycles, IPC);    
      
#ifdef PERF_ENABLE
  switch (perf_class) {
  case VX_DCR_MPM_CLASS_CORE: {    
    int ifetch_avg_lat = (int)(double(ifetch_lat) / double(ifetches));
    int load_avg_lat = (int)(double(load_lat) / double(loads));
    fprintf(stream, "PERF: ibuffer stalls=%ld\n", ibuffer_stalls);
    fprintf(stream, "PERF: scoreboard stalls=%ld\n", scoreboard_stalls);
    fprintf(stream, "PERF: alu unit stalls=%ld\n", alu_stalls);
    fprintf(stream, "PERF: lsu unit stalls=%ld\n", lsu_stalls);
    fprintf(stream, "PERF: fpu unit stalls=%ld\n", fpu_stalls);
    fprintf(stream, "PERF: sfu unit stalls=%ld\n", sfu_stalls);
    fprintf(stream, "PERF: ifetches=%ld\n", ifetches);
    fprintf(stream, "PERF: loads=%ld\n", loads);
    fprintf(stream, "PERF: stores=%ld\n", stores);    
    fprintf(stream, "PERF: ifetch latency=%d cycles\n", ifetch_avg_lat);
    fprintf(stream, "PERF: load latency=%d cycles\n", load_avg_lat);
    
  } break;  
  case VX_DCR_MPM_CLASS_MEM: {
    int icache_read_hit_ratio = (int)((1.0 - (double(icache_read_misses) / double(icache_reads))) * 100);    
    int dcache_read_hit_ratio = (int)((1.0 - (double(dcache_read_misses) / double(dcache_reads))) * 100);
    int dcache_write_hit_ratio = (int)((1.0 - (double(dcache_write_misses) / double(dcache_writes))) * 100);
    int dcache_bank_utilization = (int)((double(dcache_reads + dcache_writes) / double(dcache_reads + dcache_writes + dcache_bank_stalls)) * 100);    
    int l2cache_read_hit_ratio = (int)((1.0 - (double(l2cache_read_misses) / double(l2cache_reads))) * 100);
    int l2cache_write_hit_ratio = (int)((1.0 - (double(l2cache_write_misses) / double(l2cache_writes))) * 100);
    int l2cache_bank_utilization = (int)((double(l2cache_reads + l2cache_writes) / double(l2cache_reads + l2cache_writes + l2cache_bank_stalls)) * 100);    
    int l3cache_read_hit_ratio = (int)((1.0 - (double(l3cache_read_misses) / double(l3cache_reads))) * 100);
    int l3cache_write_hit_ratio = (int)((1.0 - (double(l3cache_write_misses) / double(l3cache_writes))) * 100);
    int l3cache_bank_utilization = (int)((double(l3cache_reads + l3cache_writes) / double(l3cache_reads + l3cache_writes + l3cache_bank_stalls)) * 100);    
    int smem_bank_utilization = (int)((double(smem_reads + smem_writes) / double(smem_reads + smem_writes + smem_bank_stalls)) * 100);    
    int mem_avg_lat = (int)(double(mem_lat) / double(mem_reads));    
    fprintf(stream, "PERF: icache reads=%ld\n", icache_reads);
    fprintf(stream, "PERF: icache read misses=%ld (hit ratio=%d%%)\n", icache_read_misses, icache_read_hit_ratio);
    fprintf(stream, "PERF: dcache reads=%ld\n", dcache_reads);
    fprintf(stream, "PERF: dcache writes=%ld\n", dcache_writes);
    fprintf(stream, "PERF: dcache read misses=%ld (hit ratio=%d%%)\n", dcache_read_misses, dcache_read_hit_ratio);
    fprintf(stream, "PERF: dcache write misses=%ld (hit ratio=%d%%)\n", dcache_write_misses, dcache_write_hit_ratio);  
    fprintf(stream, "PERF: dcache bank stalls=%ld (utilization=%d%%)\n", dcache_bank_stalls, dcache_bank_utilization);
    fprintf(stream, "PERF: dcache mshr stalls=%ld\n", dcache_mshr_stalls);
    fprintf(stream, "PERF: smem reads=%ld\n", smem_reads);
    fprintf(stream, "PERF: smem writes=%ld\n", smem_writes); 
    fprintf(stream, "PERF: smem bank stalls=%ld (utilization=%d%%)\n", smem_bank_stalls, smem_bank_utilization);
    fprintf(stream, "PERF: l2cache reads=%ld\n", l2cache_reads);
    fprintf(stream, "PERF: l2cache writes=%ld\n", l2cache_writes);
    fprintf(stream, "PERF: l2cache read misses=%ld (hit ratio=%d%%)\n", l2cache_read_misses, l2cache_read_hit_ratio);
    fprintf(stream, "PERF: l2cache write misses=%ld (hit ratio=%d%%)\n", l2cache_write_misses, l2cache_write_hit_ratio);  
    fprintf(stream, "PERF: l2cache bank stalls=%ld (utilization=%d%%)\n", l2cache_bank_stalls, l2cache_bank_utilization);
    fprintf(stream, "PERF: l2cache mshr stalls=%ld\n", l2cache_mshr_stalls);
    fprintf(stream, "PERF: l3cache reads=%ld\n", l3cache_reads);
    fprintf(stream, "PERF: l3cache writes=%ld\n", l3cache_writes);
    fprintf(stream, "PERF: l3cache read misses=%ld (hit ratio=%d%%)\n", l3cache_read_misses, l3cache_read_hit_ratio);
    fprintf(stream, "PERF: l3cache write misses=%ld (hit ratio=%d%%)\n", l3cache_write_misses, l3cache_write_hit_ratio);  
    fprintf(stream, "PERF: l3cache bank stalls=%ld (utilization=%d%%)\n", l3cache_bank_stalls, l3cache_bank_utilization);
    fprintf(stream, "PERF: l3cache mshr stalls=%ld\n", l3cache_mshr_stalls);
    fprintf(stream, "PERF: memory requests=%ld (reads=%ld, writes=%ld)\n", (mem_reads + mem_writes), mem_reads, mem_writes);
    fprintf(stream, "PERF: memory latency=%d cycles\n", mem_avg_lat);
  } break;
  default:
    break;
  }
#endif

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
