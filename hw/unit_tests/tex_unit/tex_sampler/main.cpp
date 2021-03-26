#include "vl_simulator.h"
#include "VVX_tex_sampler.h"
#include <iostream>
#include <map>

#define MAX_TICKS 20
#define MAX_UNIT_CYCLES 5
#define NUM_THREADS

#define CHECK(x)                                  \
   do {                                           \
     if (x)                                       \
       break;                                     \
     std::cout << "FAILED: " << #x << std::endl;  \
	   std::abort();			                          \
   } while (false)

uint64_t ticks = 0;

// using Device = VVX_tex_sampler;

template <typename T> 
class testbench
{
private:
  vl_simulator<T> sim;
  std::map<int, struct Input> input_map;
  std::map<int, struct Output> output_map;

public:

  struct UnitTest {
    bool use_reset;
    unsigned int num_cycles;
    bool use_cmodel;
    struct Output outputs[MAX_UNIT_CYCLES];
    struct Input inputs[MAX_UNIT_CYCLES];
    unsigned int num_output_check;
    unsigned int check_output_cycle[MAX_UNIT_CYCLES];
  }

  struct Input {
    bool          req_valid;
    unsigned int  req_wid;
    unsigned int  req_tmask;
    unsigned int  req_PC;
    unsigned int  req_rd;   
    unsigned int  req_wb;
    unsigned int  req_filter;
    unsigned int  req_format;
    unsigned int  req_u[NUM_THREADS];
    unsigned int  req_v[NUM_THREADS];
    unsigned int  req_texels[NUM_THREADS][4];
    bool          rsp_ready;
  }

  struct Output {
    int           output_cycle;
    // outputs
    bool          req_ready;
    bool          rsp_valid;
    unsigned int  rsp_wid;
    unsigned int  rsp_tmask;
    unsigned int  rsp_PC;
    unsigned int  rsp_rd;   
    bool          rsp_wb;
    unsigned int  rsp_data[NUM_THREADS];
  }

  testbench(/* args */){

  }
  
  ~testbench(){
  }

  void unittest_Cmodel(struct UnitTest * test){
    int cycles = test->num_cycles;
    int num_outputs = test->num_output_check;

    // struct Input* inputs = new (struct Input)[cycles]; 
    struct Output* outputs = new (struct Output)[num_outputs];

    // implement c model and assign outputs to struct
    
    if (test->inputs[0]->req_filter == 0){
      for (int i = 0; i < NUM_THREADS; i++)
        outputs[0]->rsp_data[0] = test->inputs->req_texels[i][0];
    } else {
      // for (int i = 0; i < NUM_THREADS; i++){
      //   uint32_t low[4], high[4];
      //   for (int j = 0; j < 4; j++){
      //     low[j] = test->inputs->req_texels[i][j] & 0x00ff00ff;
      //     high[j] = (test->inputs->req_texels[i][j] >> 8) & 0x00ff00ff;
      //   }

      // }
    }
    outputs[0]->output_cycle = 1;
    test->num_cycles = 1;
    test->outputs = &outputs;

  }

  void generate_test_vectors(struct UnitTest * tests, int num_tests, bool is_pipe){
    // for all unit tests create output test vectors (w w/o c-model)
    int prev_test_cycle = 0; 

    for (int i = 0; i < num_tests; i++)
    {
      int op_counter = 0;
      int ip_counter = 0;

      int test_cycle = 0;
      int last_ip_cycle = 0;

      struct UnitTest curr_test = tests[i];

      if (curr_test->use_cmodel){
        unittest_Cmodel(&curr_test);
      }

      for (int j = 0; j < curr_test->num_cycles; j++)
      {
        if (curr_test->inputs[ip_counter]->input_cycle == test_cycle){
          input_map.insert(std::make_pair(prev_test_cycle + test_cycle, curr_test->inputs[j]));
          last_ip_cycle = prev_test_cycle + test_cycle;
          ip_counter++;
        }

        if (curr_test->outputs[op_counter]->output_cycle == test_cycle){
          output_map.insert(std::make_pair(prev_test_cycle + test_cycle, curr_test->outputs[op_counter]));
          op_counter++;
        }

        test_cycle++;
      }

      if(!is_pipe){
        prev_test_cycle += (test_cycle - 1);
      }
      else{
        prev_test_cycle = last_ip_cycle + 1;
      }
      
    }
    
  }

  void run(){

    ticks = sim.reset(0);
    int cycle = 0;

    while (ticks < MAX_TICKS) {

      auto input = input_map.find(cycle);
      auto output = output_map.find(cycle);

      if (input != input_map.end()){
        sim->req_valid = input->req_valid;
        sim->req_wid = input->req_wid;
        sim->req_tmask = input->req_tmask;
        sim->req_PC = input->req_PC;
        sim->req_rd = input->req_rd;   
        sim->req_wb = input->req_wb;
        sim->req_filter = input->req_filter;
        sim->req_format = input->req_format;
        // sim->req_u = input->req_u[NUM_THREADS];
        // sim->req_v = input->req_v[NUM_THREADS];
        vl_setw(sim->req_texels, input->req_texels)
        // sim->req_texels = input->req_texels[NUM_THREADS][4];
        sim->rsp_ready = input->rsp_ready;
      } else{
        std::cout << "Warning! No Input on Cycle " << cycle << std::endl;       
      }

      if(output != output_map.end()){
        CHECK(sim->req_ready == output->req_ready);
        CHECK(sim->rsp_valid == output->rsp_valid);
        CHECK(sim->rsp_wid == output->rsp_wid);
        CHECK(sim->rsp_tmask == output->rsp_tmask);
        CHECK(sim->rsp_PC == output->rsp_PC);
        CHECK(sim->rsp_rd == output->rsp_rd);   
        CHECK(sim->rsp_wb == output->rsp_wb);
        CHECK(vl_cmpw(sim->rsp_data, output->rsp_data));
      }

      cycle++;
      ticks = sim.step(ticks,2);
    }
  }

  std::cout << "PASSED!" << std::endl;
  std::cout << "Simulation time: " << std::dec << ticks/2 << " cycles" << std::endl;

};


double sc_time_stamp() { 
  return ticks;
}

int main(int argc, char **argv) {
  // Initialize Verilators variables
  Verilated::commandArgs(argc, argv);

  testbench<VVX_tex_sampler> sampler_testbench;

  sampler_testbench.generate_test_vectors(tests, 1, 0);
  sampler_test_bench.run();


  return 0;
}