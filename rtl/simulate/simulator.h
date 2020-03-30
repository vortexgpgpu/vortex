#pragma once

#ifdef USE_MULTICORE
#include "VVortex_SOC.h"
#else
#include "VVortex.h"
#endif
#include "VVortex__Syms.h"
#include "verilated.h"

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

#include "VX_define.h"
#include "ram.h"

#include <ostream>
#include <vector>

//#define ENABLE_DRAM_STALLS
#define DRAM_LATENCY 200
#define DRAM_RQ_SIZE 16
#define DRAM_STALLS_MODULO 16
#define PIPELINE_FLUSH_LATENCY 300

typedef struct {
  int cycles_left;
  int data_length;
  unsigned base_addr;
  unsigned *data;
} dram_req_t;

class Simulator {
public:
  
  Simulator(RAM *ram);
  virtual ~Simulator();

  bool is_busy();  
  void reset();
  void step();
  void flush_caches(uint32_t mem_addr, uint32_t size);  
  bool run();
  void print_stats(std::ostream& out);

private:  

#ifndef USE_MULTICORE
  void ibus_driver();
#endif

  void dbus_driver();
  void io_handler();  
  void send_snoops(uint32_t mem_addr, uint32_t size);
  void wait(uint32_t cycles);

  uint64_t total_cycles_;
  bool dram_stalled_;
  bool I_dram_stalled_;
  std::vector<dram_req_t> dram_req_vec_;
  std::vector<dram_req_t> I_dram_req_vec_;
  RAM *ram_;
#ifdef USE_MULTICORE
  VVortex_SOC *vortex_;
#else
  VVortex *vortex_;
#endif
#ifdef VCD_OUTPUT
  VerilatedVcdC *trace_;
#endif
};