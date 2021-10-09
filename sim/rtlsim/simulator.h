#pragma once

#include <VX_config.h>
#include <ostream>
#include <list>
#include <vector>
#include <sstream> 
#include <unordered_map>

#ifndef MEMORY_BANKS
  #ifdef PLATFORM_PARAM_LOCAL_MEMORY_BANKS
    #define MEMORY_BANKS PLATFORM_PARAM_LOCAL_MEMORY_BANKS
  #else
    #define MEMORY_BANKS 2
  #endif
#endif

namespace vortex {

class VL_OBJ;
class RAM;

class Simulator {
public:
  
  Simulator();
  virtual ~Simulator();

  void attach_ram(RAM* ram);

  bool is_busy() const;

  void reset();
  void step();
  void wait(uint32_t cycles);

  int run();

  void print_stats(std::ostream& out);

private:  

  typedef struct {    
    int cycles_left;  
    std::array<uint8_t, MEM_BLOCK_SIZE> block;
    uint64_t addr;
    uint64_t tag;
    bool write;
  } mem_req_t;

  std::unordered_map<int, std::stringstream> print_bufs_;

  void eval();  
  
#ifdef AXI_BUS
  void reset_axi_bus();  
  void eval_axi_bus(bool clk); 
#else
  void reset_mem_bus();  
  void eval_mem_bus(bool clk);
#endif

  int get_last_wb_value(int reg) const;  
  
  bool get_ebreak() const;

  std::list<mem_req_t> mem_rsp_vec_ [MEMORY_BANKS];
  uint32_t last_mem_rsp_bank_;

  bool mem_rd_rsp_active_;
  bool mem_rd_rsp_ready_;

  bool mem_wr_rsp_active_;
  bool mem_wr_rsp_ready_;

  RAM *ram_;

  VL_OBJ* vl_obj_;
};

}