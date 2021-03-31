#include "vl_simulator.h"
#include "VVX_tex_sampler.h"
#include <iostream>

#define MAX_TICKS 20
#define NUM_THREADS

#define CHECK(x)                                  \
   do {                                           \
     if (x)                                       \
       break;                                     \
     std::cout << "FAILED: " << #x << std::endl;  \
	   std::abort();			                          \
   } while (false)

uint64_t ticks = 0;

double sc_time_stamp() { 
  return ticks;
}

using Device = VVX_tex_sampler;

int main(int argc, char **argv) {
  // Initialize Verilators variables
  Verilated::commandArgs(argc, argv);

  vl_simulator<Device> sim;

  // run test
  ticks = sim.reset(0);
  while (ticks < MAX_TICKS) {
    switch (ticks) {
    case 0:
      std::cout << "cycle 1" << std::endl;
      // input values
      sim->req_valid = 1;
      sim->req_wid = 3;
      sim->req_tmask = 11;
      sim->req_PC = 0x0505;
      sim->req_wb = 1;
      sim->req_filter = 0;
      sim->req_format = 3;
      sim->rsp_ready = 1;

      break;

    case 2:
      std::cout << "cycle 2" << std::endl;
      sim->req_valid = 1;
      sim->req_wid = 0;
      sim->req_tmask = 15;
      sim->req_PC = 0x0515;
      sim->req_wb = 1;
      sim->req_filter = 0;
      sim->req_format = 2; //rgba4
      vl_setw(sim->req_texels, 0x1234abcd, 0x1234abcd, 0x1234abcd, 0x1234abcd,
                               0xabcd1234, 0xabcd1234, 0xabcd1234, 0xabcd1234, 
                               0xfaafbaab, 0xfaafbaab, 0xfaafbaab, 0xfaafbaab, 
                               0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef);
      sim->rsp_ready = 1;

      break;
    case 4:
      //bilerp input r8b8g8a8
      std::cout << "cycle 3: point sampling check" << std::endl;

      sim->req_valid = 1;
      sim->rsp_ready = 1;
      sim->req_wid = 0;
      sim->req_tmask = 15;
      sim->req_PC = 0x0519;
      // sim->req_rd = req_rd;   
      sim->req_wb = 1;
      sim->req_filter = 1;
      sim->req_format = 0;
      /*
      1u ------- 0u              1v ------- 1v          tex0 ------- tex1
      |         |                |          |            |            |
      |         |                |          |            |            |
      |         |                |          |            |            |
      1u ------- 0u              0v ------- 0v          tex2 ------- tex3
      */
      vl_packsetw(sim->req_u, 20, 0x0080, 0x0000, 0x0080, 0x0040); // 1/2, 0, 1/2, 1/4
      vl_packsetw(sim->req_v, 20, 0x0000, 0x0080, 0x0080, 0x0040); // 0, 1/2, 1/2, 1/4
      // vl_setw(sim->req_texels, 0xffff, 0xfffa, 0xfffb, 0xfffc, 0xfffd, 0xafff, 0xbfff, 0xcfff, 0xdfff, 0xabcd, 0xdfdd, 0xeabf, 0xaaaa, 0xbbbb, 0xcccc, 0xdddd);
      // vl_setw(sim->req_texels, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000);
      vl_setw(sim->req_texels, 0xffffffff, 0x00000000, 0xffffffff, 0x00000000,
                               0x00000000, 0xffffffff, 0x00000000, 0x00000000,
                               0xffffffff, 0x00000000, 0xffffffff, 0x00000000,
                               0xffffffff, 0x00000000, 0xffffffff, 0x00000000);

      // point sampling output check
      CHECK(!vl_cmpw(sim->rsp_data, 0x0a0b0c0d, 0x01020304, 0x0b0a0a0b, 0x0b0e0e0f));
      
      break;

    case 6:

      sim->req_valid = 1;
      sim->rsp_ready = 1;
      // bilerp sampling output check


      std::cout << "cycle 4: bilerp r8g8b8a8 check" << std::endl;
      CHECK(!vl_cmpw(sim->rsp_data, 0x7f7f7f7f, 0x7f7f7f7f, 0x7f7f7f7f, 0x3f3f3f3f));

    }

    // advance clock
    ticks = sim.step(ticks, 2);
  }

  std::cout << "PASSED!" << std::endl;
  std::cout << "Simulation time: " << std::dec << ticks/2 << " cycles" << std::endl;

  return 0;
}