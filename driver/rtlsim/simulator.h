#pragma once

#include "Vvortex_afu_sim.h"
#include "Vvortex_afu_sim__Syms.h"
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
  uint8_t *data;
  unsigned tag;
} dram_req_t;

class Simulator {
public:
  
  Simulator();
  virtual ~Simulator();

  void reset();
  
  void step();  

  int mmio_read(uint64_t addr, uint64_t* value);

  int mmio_write(uint64_t addr, uint64_t value);
  
private:  

  void eval(); 

  void avs_driver();

  void ccip_driver(); 
  
  std::vector<dram_req_t> dram_rsp_vec_;

  RAM ram_;
  Vvortex_afu_sim *vortex_;
  

#ifdef VCD_OUTPUT
  VerilatedVcdC *trace_;
#endif
};