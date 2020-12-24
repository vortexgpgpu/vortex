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
#include <list>
#include <vector>
#include <sstream> 
#include <unordered_map>

class Simulator {
public:
  
  Simulator();
  virtual ~Simulator();

  void attach_ram(RAM* ram);

  void load_bin(const char* program_file);
  void load_ihex(const char* program_file);
  
  bool is_busy() const;

  bool csr_req_active() const;

  void reset();
  void step();
  void wait(uint32_t cycles);
  
  void set_csr(int core_id, int addr, unsigned value);
  void get_csr(int core_id, int addr, unsigned *value);

  void run();  
  int get_last_wb_value(int reg) const;  

  void print_stats(std::ostream& out);

private:  

  typedef struct {
    int cycles_left;  
    std::array<uint8_t, GLOBAL_BLOCK_SIZE> block;
    uint32_t addr;
    uint32_t tag;
  } dram_req_t;

  std::unordered_map<int, std::stringstream> print_bufs_;

  void eval();  

  void eval_dram_bus();
  void eval_io_bus();
  void eval_csr_bus();
  
  std::list<dram_req_t> dram_rsp_vec_;
  bool dram_rsp_active_;

  bool dram_rsp_ready_;  
  bool csr_req_ready_;
  bool csr_req_active_;
  uint32_t* csr_rsp_value_;

  RAM *ram_;
  VVortex *vortex_;
#ifdef VCD_OUTPUT
  VerilatedVcdC *trace_;
#endif
};