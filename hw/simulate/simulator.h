#pragma once

#include "VVortex_Socket.h"
#include "VVortex_Socket__Syms.h"
#include "verilated.h"

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

#include "VX_config.h"
#include "ram.h"

#include <ostream>
#include <vector>

#define ENABLE_DRAM_STALLS
#define DRAM_LATENCY 100
#define DRAM_RQ_SIZE 16
#define DRAM_STALLS_MODULO 16
#define PIPELINE_FLUSH_LATENCY 300

typedef struct {
  int cycles_left;  
  unsigned *data;
  unsigned tag;
} dram_req_t;

class Simulator {
public:
  
  Simulator(RAM *ram);
  virtual ~Simulator();

  bool is_busy();  
  void reset();
  void step();
  void wait(uint32_t cycles);
  void flush_caches(uint32_t mem_addr, uint32_t size);  
  bool run();  
  void print_stats(std::ostream& out);

private:  

  void eval();  
  void dbus_driver();
  void io_driver();
  
  std::vector<dram_req_t> dram_rsp_vec_;

  RAM *ram_;
  VVortex_Socket *vortex_;
#ifdef VCD_OUTPUT
  VerilatedVcdC *trace_;
#endif
};