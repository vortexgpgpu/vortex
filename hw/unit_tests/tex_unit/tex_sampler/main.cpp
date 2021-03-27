#include "vl_simulator.h"
#include "VVX_tex_sampler.h"
#include <iostream>

#define MAX_TICKS 20
#define NUM_THREADS



    // // outputs
    // bool          req_ready;
    // bool          rsp_valid;
    // unsigned int  rsp_wid;
    // unsigned int  rsp_tmask;
    // unsigned int  rsp_PC;
    // unsigned int  rsp_rd;   
    // bool          rsp_wb;
    // unsigned int  rsp_data[NUM_THREADS];


    // if (input != input_map.end()){
    //   sim->req_valid = input->req_valid;
    //   sim->req_wid = input->req_wid;
    //   sim->req_tmask = input->req_tmask;
    //   sim->req_PC = input->req_PC;
    //   sim->req_rd = input->req_rd;   
    //   sim->req_wb = input->req_wb;
    //   sim->req_filter = input->req_filter;
    //   sim->req_format = input->req_format;
    //   // sim->req_u = input->req_u[NUM_THREADS];
    //   // sim->req_v = input->req_v[NUM_THREADS];
    //   vl_setw(sim->req_texels, input->req_texels)
    //   // sim->req_texels = input->req_texels[NUM_THREADS][4];
    //   sim->rsp_ready = input->rsp_ready;
    // } else{
    //   std::cout << "Warning! No Input on Cycle " << cycle << std::endl;       
    // }

    // if(output != output_map.end()){
    //   CHECK(sim->req_ready == output->req_ready);
    //   CHECK(sim->rsp_valid == output->rsp_valid);
    //   CHECK(sim->rsp_wid == output->rsp_wid);
    //   CHECK(sim->rsp_tmask == output->rsp_tmask);
    //   CHECK(sim->rsp_PC == output->rsp_PC);
    //   CHECK(sim->rsp_rd == output->rsp_rd);   
    //   CHECK(sim->rsp_wb == output->rsp_wb);
    //   CHECK(vl_cmpw(sim->rsp_data, output->rsp_data));
    // }

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
      sim->req_format = 3;
      vl_setw(sim->req_texels, 0xffff, 0xfffa, 0xfffb, 0xfffc,
                               0xfffd, 0xafff, 0xbfff, 0xcfff, 
                               0xdfff, 0xabcd, 0xdfdd, 0xeabf, 
                               0xaaaa, 0xbbbb, 0xcccc, 0xdddd);
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
      sim->req_format = 3;
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
      CHECK(!vl_cmpw(sim->rsp_data, 0xffff, 0xfffd, 0xdfff, 0xaaaa));
      
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