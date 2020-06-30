#pragma once

#include "VVX_priority_encoder.h"
#include "VVX_priority_encoder__Syms.h"
#include "verilated.h"

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

//#include <VX_config.h>

#include <ostream>
#include <vector>

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


  bool run();  
  void print_stats(std::ostream& out);

private:  

  void eval();  

  void eval_dram_bus();
  void eval_io_bus();
  void eval_snp_bus();

  uint32_t snp_req_active_;
  uint32_t snp_req_size_;
  uint32_t pending_snp_reqs_;

  VVX_priority_encoder *vortex_;
#ifdef VCD_OUTPUT
  VerilatedVcdC *trace_;
#endif
};
