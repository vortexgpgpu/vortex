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

  void reset();
  void step();
  void wait(uint32_t cycles);

  void run();  

  int get_last_wb_value(int reg) const;  
  
  bool get_ebreak() const;

  void print_stats(std::ostream& out);

private:  

  typedef struct {
    int cycles_left;  
    std::array<uint8_t, MEM_BLOCK_SIZE> block;
    uint32_t addr;
    uint64_t tag;
  } mem_req_t;

  std::unordered_map<int, std::stringstream> print_bufs_;

  void eval();  

  void eval_mem_bus();

  std::list<mem_req_t> mem_rsp_vec_;
  bool mem_rsp_active_;

  bool mem_rsp_ready_;

  RAM *ram_;
  VVortex *vortex_;
#ifdef VCD_OUTPUT
  VerilatedVcdC *trace_;
#endif
};