#include <iostream>
#include <fstream>
#include <cstring>
#include <vortex.h>
#include <VX_config.h>

extern int vx_upload_kernel_bytes(vx_device_h device, const void* content, size_t size) {
  int err = 0;

  if (NULL == content || 0 == size)
    return -1;

  uint32_t buffer_transfer_size = 65536;
  unsigned kernel_base_addr;
  err = vx_dev_caps(device, VX_CAPS_KERNEL_BASE_ADDR, &kernel_base_addr);
  if (err != 0)
    return -1;

  // allocate device buffer
  vx_buffer_h buffer;
  err = vx_alloc_shared_mem(device, buffer_transfer_size, &buffer);
  if (err != 0)
    return -1; 

  // get buffer address
  auto buf_ptr = (uint8_t*)vx_host_ptr(buffer);

  //
  // upload content
  //

  size_t offset = 0;
  while (offset < size) {
    auto chunk_size = std::min<size_t>(buffer_transfer_size, size - offset);
    std::memcpy(buf_ptr, (uint8_t*)content + offset, chunk_size);

    /*printf("***  Upload Kernel to 0x%0x: data=", kernel_base_addr + offset);
    for (int i = 0, n = ((chunk_size+7)/8); i < n; ++i) {
      printf("%08x", ((uint64_t*)((uint8_t*)content + offset))[n-1-i]);
    }
    printf("\n");*/

    err = vx_copy_to_dev(buffer, kernel_base_addr + offset, chunk_size, 0);
    if (err != 0) {
      vx_buf_release(buffer);
      return err;
    }
    offset += chunk_size;
  }

  vx_buf_release(buffer);

  return 0;
}

extern int vx_upload_kernel_file(vx_device_h device, const char* filename) {
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
  int err = vx_upload_kernel_bytes(device, content, size);

  // release buffer
  delete[] content;

  return err;
}

/*static uint32_t get_csr_32(const uint32_t* buffer, int addr) {
  uint32_t value_lo = buffer[addr - CSR_MPM_BASE];
  return value_lo;
}*/

static uint64_t get_csr_64(const uint32_t* buffer, int addr) {
  uint32_t value_lo = buffer[addr - CSR_MPM_BASE];
  uint32_t value_hi = buffer[addr - CSR_MPM_BASE + 32];
  return (uint64_t(value_hi) << 32) | value_lo;
}

extern int vx_dump_perf(vx_device_h device, FILE* stream) {
  int ret = 0;

  uint64_t instrs = 0;
  uint64_t cycles = 0;

#ifdef PERF_ENABLE    
  // PERF: pipeline stalls
  uint64_t ibuffer_stalls = 0;
  uint64_t scoreboard_stalls = 0;
  uint64_t lsu_stalls = 0;
  uint64_t fpu_stalls = 0;
  uint64_t csr_stalls = 0;
  uint64_t alu_stalls = 0;
  uint64_t gpu_stalls = 0;
  // PERF: Icache 
  uint64_t icache_reads = 0;
  uint64_t icache_read_misses = 0;
  uint64_t icache_pipe_stalls = 0;
  uint64_t icache_rsp_stalls = 0;
  // PERF: Dcache 
  uint64_t dcache_reads = 0;
  uint64_t dcache_writes = 0;
  uint64_t dcache_read_misses = 0;
  uint64_t dcache_write_misses = 0;
  uint64_t dcache_bank_stalls = 0;  
  uint64_t dcache_mshr_stalls = 0;
  uint64_t dcache_pipe_stalls = 0;
  uint64_t dcache_rsp_stalls = 0;  
  // PERF: SMEM
  uint64_t smem_reads = 0;
  uint64_t smem_writes = 0;
  uint64_t smem_bank_stalls = 0;
  // PERF: memory
  uint64_t mem_reads = 0;
  uint64_t mem_writes = 0;
  uint64_t mem_stalls = 0;
  uint64_t mem_lat = 0;
#endif     

  unsigned num_cores;
  ret = vx_dev_caps(device, VX_CAPS_MAX_CORES, &num_cores);
  if (ret != 0)
    return ret;

  vx_buffer_h staging_buf;
  ret = vx_alloc_shared_mem(device, 64 * sizeof(uint32_t), &staging_buf);
  if (ret != 0)
    return ret;

  auto staging_ptr = (uint32_t*)vx_host_ptr(staging_buf);
      
  for (unsigned core_id = 0; core_id < num_cores; ++core_id) {
    ret = vx_copy_from_dev(staging_buf, IO_CSR_ADDR + 64 * sizeof(uint32_t) * core_id, 64 * sizeof(uint32_t), 0);
    if (ret != 0) {
      vx_buf_release(staging_buf);
      return ret;
    }

    uint64_t instrs_per_core = get_csr_64(staging_ptr, CSR_MINSTRET);
    uint64_t cycles_per_core = get_csr_64(staging_ptr, CSR_MCYCLE);
    float IPC = (float)(double(instrs_per_core) / double(cycles_per_core));
    if (num_cores > 1) fprintf(stream, "PERF: core%d: instrs=%ld, cycles=%ld, IPC=%f\n", core_id, instrs_per_core, cycles_per_core, IPC);            
    instrs += instrs_per_core;
    cycles = std::max<uint64_t>(cycles_per_core, cycles);

  #ifdef PERF_ENABLE
    // PERF: pipeline    
    // ibuffer_stall
    uint64_t ibuffer_stalls_per_core = get_csr_64(staging_ptr, CSR_MPM_IBUF_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: ibuffer stalls=%ld\n", core_id, ibuffer_stalls_per_core);
    ibuffer_stalls += ibuffer_stalls_per_core;
    // scoreboard_stall
    uint64_t scoreboard_stalls_per_core = get_csr_64(staging_ptr, CSR_MPM_SCRB_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: scoreboard stalls=%ld\n", core_id, scoreboard_stalls_per_core);
    scoreboard_stalls += scoreboard_stalls_per_core;
    // alu_stall
    uint64_t alu_stalls_per_core = get_csr_64(staging_ptr, CSR_MPM_ALU_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: alu unit stalls=%ld\n", core_id, alu_stalls_per_core);
    alu_stalls += alu_stalls_per_core;      
    // lsu_stall
    uint64_t lsu_stalls_per_core = get_csr_64(staging_ptr, CSR_MPM_LSU_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: lsu unit stalls=%ld\n", core_id, lsu_stalls_per_core);
    lsu_stalls += lsu_stalls_per_core;
    // csr_stall
    uint64_t csr_stalls_per_core = get_csr_64(staging_ptr, CSR_MPM_CSR_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: csr unit stalls=%ld\n", core_id, csr_stalls_per_core);
    csr_stalls += csr_stalls_per_core;    
    // fpu_stall
    uint64_t fpu_stalls_per_core = get_csr_64(staging_ptr, CSR_MPM_FPU_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: fpu unit stalls=%ld\n", core_id, fpu_stalls_per_core);
    fpu_stalls += fpu_stalls_per_core;      
    // gpu_stall
    uint64_t gpu_stalls_per_core = get_csr_64(staging_ptr, CSR_MPM_GPU_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: gpu unit stalls=%ld\n", core_id, gpu_stalls_per_core);
    gpu_stalls += gpu_stalls_per_core;  

    // PERF: Icache
    // total reads
    uint64_t icache_reads_per_core = get_csr_64(staging_ptr, CSR_MPM_ICACHE_READS);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: icache reads=%ld\n", core_id, icache_reads_per_core);
    icache_reads += icache_reads_per_core;
    // read misses
    uint64_t icache_miss_r_per_core = get_csr_64(staging_ptr, CSR_MPM_ICACHE_MISS_R);
    int icache_read_hit_ratio = (int)((1.0 - (double(icache_miss_r_per_core) / double(icache_reads_per_core))) * 100);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: icache read misses=%ld (hit ratio=%d%%)\n", core_id, icache_miss_r_per_core, icache_read_hit_ratio);
    icache_read_misses += icache_miss_r_per_core;
    // pipeline stalls
    uint64_t icache_pipe_st_per_core = get_csr_64(staging_ptr, CSR_MPM_ICACHE_PIPE_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: icache pipeline stalls=%ld\n", core_id, icache_pipe_st_per_core);
    icache_pipe_stalls += icache_pipe_st_per_core;
    // response stalls
    uint64_t icache_crsp_st_per_core = get_csr_64(staging_ptr, CSR_MPM_ICACHE_CRSP_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: icache reponse stalls=%ld\n", core_id, icache_crsp_st_per_core);
    icache_rsp_stalls += icache_crsp_st_per_core;

    // PERF: Dcache
    // total reads
    uint64_t dcache_reads_per_core = get_csr_64(staging_ptr, CSR_MPM_DCACHE_READS);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: dcache reads=%ld\n", core_id, dcache_reads_per_core);
    dcache_reads += dcache_reads_per_core;
    // total write
    uint64_t dcache_writes_per_core = get_csr_64(staging_ptr, CSR_MPM_DCACHE_WRITES);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: dcache writes=%ld\n", core_id, dcache_writes_per_core);
    dcache_writes += dcache_writes_per_core;
    // read misses
    uint64_t dcache_miss_r_per_core = get_csr_64(staging_ptr, CSR_MPM_DCACHE_MISS_R);
    int dcache_read_hit_ratio = (int)((1.0 - (double(dcache_miss_r_per_core) / double(dcache_reads_per_core))) * 100);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: dcache read misses=%ld (hit ratio=%d%%)\n", core_id, dcache_miss_r_per_core, dcache_read_hit_ratio);
    dcache_read_misses += dcache_miss_r_per_core;
    // read misses
    uint64_t dcache_miss_w_per_core = get_csr_64(staging_ptr, CSR_MPM_DCACHE_MISS_W);
    int dcache_write_hit_ratio = (int)((1.0 - (double(dcache_miss_w_per_core) / double(dcache_writes_per_core))) * 100);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: dcache write misses=%ld (hit ratio=%d%%)\n", core_id, dcache_miss_w_per_core, dcache_write_hit_ratio);
    dcache_write_misses += dcache_miss_w_per_core;
    // bank_stalls
    uint64_t dcache_bank_st_per_core = get_csr_64(staging_ptr, CSR_MPM_DCACHE_BANK_ST);
    int dcache_bank_utilization = (int)((double(dcache_reads_per_core + dcache_writes_per_core) / double(dcache_reads_per_core + dcache_writes_per_core + dcache_bank_st_per_core)) * 100);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: dcache bank stalls=%ld (utilization=%d%%)\n", core_id, dcache_bank_st_per_core, dcache_bank_utilization);
    dcache_bank_stalls += dcache_bank_st_per_core;
    // mshr_stalls
    uint64_t dcache_mshr_st_per_core = get_csr_64(staging_ptr, CSR_MPM_DCACHE_MSHR_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: dcache mshr stalls=%ld\n", core_id, dcache_mshr_st_per_core);
    dcache_mshr_stalls += dcache_mshr_st_per_core; 
     // pipeline stalls
    uint64_t dcache_pipe_st_per_core = get_csr_64(staging_ptr, CSR_MPM_DCACHE_PIPE_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: dcache pipeline stalls=%ld\n", core_id, dcache_pipe_st_per_core);
    dcache_pipe_stalls += dcache_pipe_st_per_core;
   // response stalls
    uint64_t dcache_crsp_st_per_core = get_csr_64(staging_ptr, CSR_MPM_DCACHE_CRSP_ST);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: dcache reponse stalls=%ld\n", core_id, dcache_crsp_st_per_core);
    dcache_rsp_stalls += dcache_crsp_st_per_core;

    // PERF: SMEM
    // total reads
    uint64_t smem_reads_per_core = get_csr_64(staging_ptr, CSR_MPM_SMEM_READS);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: smem reads=%ld\n", core_id, smem_reads_per_core);
    smem_reads += smem_reads_per_core;
    // total write
    uint64_t smem_writes_per_core = get_csr_64(staging_ptr, CSR_MPM_SMEM_WRITES);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: smem writes=%ld\n", core_id, smem_writes_per_core);
    smem_writes += smem_writes_per_core;
    // bank_stalls
    uint64_t smem_bank_st_per_core = get_csr_64(staging_ptr, CSR_MPM_SMEM_BANK_ST);
    int smem_bank_utilization = (int)((double(smem_reads_per_core + smem_writes_per_core) / double(smem_reads_per_core + smem_writes_per_core + smem_bank_st_per_core)) * 100);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: smem bank stalls=%ld (utilization=%d%%)\n", core_id, smem_bank_st_per_core, smem_bank_utilization);
    smem_bank_stalls += smem_bank_st_per_core;

    // PERF: memory
    uint64_t mem_reads_per_core  = get_csr_64(staging_ptr, CSR_MPM_MEM_READS);
    uint64_t mem_writes_per_core = get_csr_64(staging_ptr, CSR_MPM_MEM_WRITES);
    uint64_t mem_stalls_per_core = get_csr_64(staging_ptr, CSR_MPM_MEM_ST);
    uint64_t mem_lat_per_core    = get_csr_64(staging_ptr, CSR_MPM_MEM_LAT);      
    int mem_utilization = (int)((double(mem_reads_per_core + mem_writes_per_core) / double(mem_reads_per_core + mem_writes_per_core + mem_stalls_per_core)) * 100);
    int mem_avg_lat = (int)(double(mem_lat_per_core) / double(mem_reads_per_core));       
    if (num_cores > 1) fprintf(stream, "PERF: core%d: memory requests=%ld (reads=%ld, writes=%ld)\n", core_id, (mem_reads_per_core + mem_writes_per_core), mem_reads_per_core, mem_writes_per_core);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: memory stalls=%ld (utilization=%d%%)\n", core_id, mem_stalls_per_core, mem_utilization);
    if (num_cores > 1) fprintf(stream, "PERF: core%d: memory average latency=%d cycles\n", core_id, mem_avg_lat);
    mem_reads  += mem_reads_per_core;
    mem_writes += mem_writes_per_core;
    mem_stalls += mem_stalls_per_core;
    mem_lat    += mem_lat_per_core;    
  #endif
  }  
  
  float IPC = (float)(double(instrs) / double(cycles));
  fprintf(stream, "PERF: instrs=%ld, cycles=%ld, IPC=%f\n", instrs, cycles, IPC);    
      
#ifdef PERF_ENABLE
  int icache_read_hit_ratio = (int)((1.0 - (double(icache_read_misses) / double(icache_reads))) * 100);
  int dcache_read_hit_ratio = (int)((1.0 - (double(dcache_read_misses) / double(dcache_reads))) * 100);
  int dcache_write_hit_ratio = (int)((1.0 - (double(dcache_write_misses) / double(dcache_writes))) * 100);
  int dcache_bank_utilization = (int)((double(dcache_reads + dcache_writes) / double(dcache_reads + dcache_writes + dcache_bank_stalls)) * 100);
  int smem_bank_utilization = (int)((double(smem_reads + smem_writes) / double(smem_reads + smem_writes + smem_bank_stalls)) * 100);
  int mem_utilization = (int)((double(mem_reads + mem_writes) / double(mem_reads + mem_writes + mem_stalls)) * 100);
  int mem_avg_lat = (int)(double(mem_lat) / double(mem_reads));
  fprintf(stream, "PERF: ibuffer stalls=%ld\n", ibuffer_stalls);
  fprintf(stream, "PERF: scoreboard stalls=%ld\n", scoreboard_stalls);
  fprintf(stream, "PERF: alu unit stalls=%ld\n", alu_stalls);
  fprintf(stream, "PERF: lsu unit stalls=%ld\n", lsu_stalls);
  fprintf(stream, "PERF: csr unit stalls=%ld\n", csr_stalls);
  fprintf(stream, "PERF: fpu unit stalls=%ld\n", fpu_stalls);
  fprintf(stream, "PERF: gpu unit stalls=%ld\n", gpu_stalls);
  fprintf(stream, "PERF: icache reads=%ld\n", icache_reads);
  fprintf(stream, "PERF: icache read misses=%ld (hit ratio=%d%%)\n", icache_read_misses, icache_read_hit_ratio);
  fprintf(stream, "PERF: icache pipeline stalls=%ld\n", icache_pipe_stalls);  
  fprintf(stream, "PERF: icache reponse stalls=%ld\n", icache_rsp_stalls);
  fprintf(stream, "PERF: dcache reads=%ld\n", dcache_reads);
  fprintf(stream, "PERF: dcache writes=%ld\n", dcache_writes);
  fprintf(stream, "PERF: dcache read misses=%ld (hit ratio=%d%%)\n", dcache_read_misses, dcache_read_hit_ratio);
  fprintf(stream, "PERF: dcache write misses=%ld (hit ratio=%d%%)\n", dcache_write_misses, dcache_write_hit_ratio);  
  fprintf(stream, "PERF: dcache bank stalls=%ld (utilization=%d%%)\n", dcache_bank_stalls, dcache_bank_utilization);
  fprintf(stream, "PERF: dcache mshr stalls=%ld\n", dcache_mshr_stalls);
  fprintf(stream, "PERF: dcache pipeline stalls=%ld\n", dcache_pipe_stalls);
  fprintf(stream, "PERF: dcache reponse stalls=%ld\n", dcache_rsp_stalls);
  fprintf(stream, "PERF: smem reads=%ld\n", smem_reads);
  fprintf(stream, "PERF: smem writes=%ld\n", smem_writes); 
  fprintf(stream, "PERF: smem bank stalls=%ld (utilization=%d%%)\n", smem_bank_stalls, smem_bank_utilization);
  fprintf(stream, "PERF: memory requests=%ld (reads=%ld, writes=%ld)\n", (mem_reads + mem_writes), mem_reads, mem_writes);
  fprintf(stream, "PERF: memory stalls=%ld (utilization=%d%%)\n", mem_stalls, mem_utilization);
  fprintf(stream, "PERF: memory average latency=%d cycles\n", mem_avg_lat);
#endif

  // release allocated resources
  vx_buf_release(staging_buf);

  return ret;
}