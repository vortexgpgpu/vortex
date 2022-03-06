#include "vl_simulator.h"
#include "VVX_tex_sampler.h"
#include <iostream>
#include <map>

#define MAX_TICKS        20
#define MAX_UNIT_CYCLES  5
#define NUM_THREADS      4

#define CHECK(x)                                  \
   do {                                           \
     if (x)                                       \
       break;                                     \
     std::cout << "FAILED: " << #x << std::endl;  \
	   std::abort();			                          \
   } while (false)

static uint64_t timestamp = 0;
static bool trace_enabled = false;
static uint64_t trace_start_time = 0;
static uint64_t trace_stop_time = -1ull;

double sc_time_stamp () { 
  return timestamp;
}

bool sim_trace_enabled () {
  if (timestamp >= trace_start_time 
   && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable (bool enable) {
  trace_enabled = enable;
}

template <typename T> 
class TestBench {
private:
  
  struct Input {
    int           input_cycle;
    bool          req_valid;
    //unsigned int  req_wid;
    unsigned int  req_tmask;
    //unsigned int  req_PC;
    //unsigned int  req_rd;   
    //unsigned int  req_wb;
    unsigned int  req_filter;
    unsigned int  req_format;
    //unsigned int  req_u[NUM_THREADS];
    //unsigned int  req_v[NUM_THREADS];
    unsigned int  req_texels[NUM_THREADS][4];
    bool          rsp_ready;
  };

  struct Output {
    int           output_cycle;
    // outputs
    bool          rsp_valid;
    bool          req_ready;    
    //unsigned int  rsp_wid;
    unsigned int  rsp_tmask;
    //unsigned int  rsp_PC;
    //unsigned int  rsp_rd;   
    //bool          rsp_wb;
    uint32_t rsp_data[NUM_THREADS];
  };  

  vl_simulator<T> sim;
  std::map<uint64_t, Input> input_map;
  std::map<uint64_t, Output> output_map;

public:

  struct UnitTest {
    bool use_reset;
    unsigned int num_cycles;
    bool use_cmodel;
    Output outputs[MAX_UNIT_CYCLES];
    Input inputs[MAX_UNIT_CYCLES];
    unsigned int num_output_check;
    unsigned int check_output_cycle[MAX_UNIT_CYCLES];
  };

  TestBench(/* args */) {
    //--
  }
  
  ~TestBench() {
    //--
  }

  void unittest_Cmodel(UnitTest* test) {
    int cycles = test->num_cycles;
    int num_outputs = test->num_output_check;

    auto outputs = new Output[num_outputs];

    // implement c model and assign outputs to struct
    
    if (test->inputs[0].req_filter == 0) {
      for (int i = 0; i < NUM_THREADS; i++) {
        outputs[0].rsp_data[0] = test->inputs[0].req_texels[i][0];
      }
    } else {
      // for (int i = 0; i < NUM_THREADS; i++){
      //   uint32_t low[4], high[4];
      //   for (int j = 0; j < 4; j++){
      //     low[j] = test->inputs->req_texels[i][j] & 0x00ff00ff;
      //     high[j] = (test->inputs->req_texels[i][j] >> 8) & 0x00ff00ff;
      // }
    }

    outputs[0].output_cycle = 1;
    test->num_cycles = 1;
    //test->outputs = outputs;
  }

  void generate_test_vectors(UnitTest* tests, int num_tests, bool is_pipe) {
    // for all unit tests create output test vectors (w w/o c-model)
    int prev_test_cycle = 0; 

    for (int i = 0; i < num_tests; i++) {
      int op_counter = 0;
      int ip_counter = 0;

      int test_cycle = 0;
      int last_ip_cycle = 0;

      auto& curr_test = tests[i];

      if (curr_test.use_cmodel) {
        unittest_Cmodel(&curr_test);
      }

      for (int j = 0; j < curr_test.num_cycles; j++) {
        if (curr_test.inputs[ip_counter].input_cycle == test_cycle) {
          input_map.insert(std::make_pair(prev_test_cycle + test_cycle, curr_test.inputs[j]));
          last_ip_cycle = prev_test_cycle + test_cycle;
          ip_counter++;
        }

        if (curr_test.outputs[op_counter].output_cycle == test_cycle) {
          output_map.insert(std::make_pair(prev_test_cycle + test_cycle, curr_test.outputs[op_counter]));
          op_counter++;
        }

        ++test_cycle;
      }

      if (!is_pipe) {
        prev_test_cycle += (test_cycle - 1);
      } else {
        prev_test_cycle = last_ip_cycle + 1;
      }      
    }    
  }

  void run() {

    timestamp = sim.reset(0);
    uint64_t cycle = 0;

    while (timestamp < MAX_TICKS) {

      auto input = input_map.find(cycle);
      auto output = output_map.find(cycle);

      if (input != input_map.end()) {
        sim->req_valid = input->second.req_valid;
        sim->req_tmask = input->second.req_tmask;
        //sim->req_PC = input->second.req_PC;
        //sim->req_rd = input->second.req_rd;   
        //sim->req_wb = input->second.req_wb;
        //sim->req_filter = input->second.req_filter;
        sim->req_format = input->second.req_format;
        // sim->req_u = input->second.req_u[NUM_THREADS];
        // sim->req_v = input->second.req_v[NUM_THREADS];
        //vl_setw(sim->req_texels, input->req_texels);
        // sim->req_texels = input->second.req_texels[NUM_THREADS][4];
        sim->rsp_ready = input->second.rsp_ready;
      } else {
        std::cout << "Warning! No Input on Cycle " << cycle << std::endl;       
      }

      if (output != output_map.end()) {        
        CHECK(sim->rsp_valid == output->second.rsp_valid);
        CHECK(sim->req_ready == output->second.req_ready);
        //CHECK(sim->rsp_wid == output->second.rsp_wid);
        CHECK(sim->rsp_tmask == output->second.rsp_tmask);
        //CHECK(sim->rsp_PC == output->second.rsp_PC);
        //CHECK(sim->rsp_rd == output->second.rsp_rd);   
        //CHECK(sim->rsp_wb == output->second.rsp_wb);
        //CHECK(vl_cmpw(sim->rsp_data, output->second.rsp_data));
      }

      ++cycle;

      timestamp = sim.step(timestamp, 2);
    }

    std::cout << "PASSED!" << std::endl;
    std::cout << "Simulation time: " << std::dec << timestamp/2 << " cycles" << std::endl;
  }  
};

int main(int argc, char **argv) {
  // Initialize Verilators variables
  Verilated::commandArgs(argc, argv);

  TestBench<VVX_tex_sampler> sampler_testbench;

  TestBench<VVX_tex_sampler>::UnitTest tests = {0};

  sampler_testbench.generate_test_vectors(&tests, 1, 0);
  sampler_testbench.run();

  return 0;
}