#pragma once

#include "VVortex.h"
#include "VVortex__Syms.h"
#include "verilated.h"

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

#include <VX_config.h>
#include "ram.h"

#include <ostream>
#include <vector>

#define ENABLE_DRAM_STALLS
#define DRAM_LATENCY 100
#define DRAM_RQ_SIZE 16
#define DRAM_STALLS_MODULO 16

typedef struct {
  int cycles_left;  
  std::array<uint8_t, GLOBAL_BLOCK_SIZE> block;
  unsigned tag;
} dram_req_t;

class Simulator {
public:
  
  Simulator();
  virtual ~Simulator();

  void load_bin(const char* program_file);
  void load_ihex(const char* program_file);
  
  bool is_busy();  

  void reset();
  void step();
  void wait(uint32_t cycles);
  void flush_caches(uint32_t mem_addr, uint32_t size);  

  void attach_ram(RAM* ram);

  bool run();  
  void print_stats(std::ostream& out);

private:  

  void eval();  

  void eval_dram_bus();
  void eval_io_bus();
  void eval_csr_bus();
  void eval_snp_bus();
  
  std::vector<dram_req_t> dram_rsp_vec_;
  int dram_rsp_active_;
  
  bool snp_req_active_;
  uint32_t snp_req_size_;
  uint32_t pending_snp_reqs_;

  RAM *ram_;
  VVortex *vortex_;
#ifdef VCD_OUTPUT
  VerilatedVcdC *trace_;
#endif
};