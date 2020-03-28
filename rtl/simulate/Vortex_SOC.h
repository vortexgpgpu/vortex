// C++ libraries
#include <utility> 
#include <iostream>
#include <map> 
#include <iterator>
#include <iomanip>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <math.h>
#include <algorithm>

#include "VX_define.h"
#include "ram.h"
#include "VVortex_SOC.h"
#include "verilated.h"

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

typedef struct
{
    int cycles_left;
    int data_length;
    unsigned base_addr;
    unsigned * data;
} dram_req_t;

class Vortex_SOC
{
    public:
        Vortex_SOC(RAM* ram);
        ~Vortex_SOC();
        bool simulate();
        void step();
        void reset();
        void flush_caches(uint32_t mem_addr, uint32_t size);
        bool is_busy();
    private:
        void print_stats(bool = true);
        bool ibus_driver();
        bool dbus_driver();
        void io_handler();        
        void send_snoops(uint32_t mem_addr, uint32_t size);
        void wait(uint32_t cycles);

        RAM* ram;

        VVortex_SOC * vortex;

        unsigned start_pc;
        bool     refill_d;
        unsigned refill_addr_d;
        bool     refill_i;
        unsigned refill_addr_i;
        long int curr_cycle;
        bool stop;
        bool unit_test;
        std::ofstream results;
        int stats_static_inst;
        int stats_dynamic_inst;
        int stats_total_cycles;
        int stats_fwd_stalls;
        int stats_branch_stalls;
        int debug_state;
        int ibus_state;
        int dbus_state;
        int debug_return;
        int debug_wait_num;
        int debug_inst_num;
        int debug_end_wait;
        int debug_debugAddr;
        double stats_sim_time;
        std::vector<dram_req_t> dram_req_vec;
        #ifdef VCD_OUTPUT
        VerilatedVcdC   *m_trace;
        #endif
};