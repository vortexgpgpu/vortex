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
#include "VVX_d_cache_encapsulate.h"
#include "verilated.h"

#include "d_cache_test_bench_debug.h"


#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

// void set_Index (auto & var, int index, int size, auto val)
// {
//     int real_shift
// }

class VX_d_cache
{
    public:
        VX_d_cache();
        ~VX_d_cache();
        bool simulate();
        bool operation(int, bool);

        VVX_d_cache_encapsulate * vx_d_cache_;
        long int curr_cycle;
        int stats_total_cycles = 0;
        int stats_dram_accesses = 0;
    #ifdef VCD_OUTPUT
        VerilatedVcdC   *m_trace;
    #endif
};



VX_d_cache::VX_d_cache() : curr_cycle(0), stats_total_cycles(0), stats_dram_accesses(0)
{

    this->vx_d_cache_ = new VVX_d_cache_encapsulate;
#ifdef VCD_OUTPUT
    this->m_trace = new VerilatedVcdC;
    this->vx_d_cache_->trace(m_trace, 99);
    this->m_trace->open("trace.vcd");
#endif
    //this->results.open("../results.txt");
}

VX_d_cache::~VX_d_cache()
{   
    delete this->vx_d_cache_;
#ifdef VCD_OUTPUT
    m_trace->close();
#endif
}

bool VX_d_cache::operation(int counter_value, bool do_op) {
    if (do_op) {
        vx_d_cache_->i_p_initial_request = 1;
    } else {
        vx_d_cache_->i_p_initial_request = 0;
    }
    
    if (counter_value == 0 && do_op) {                    // Write to bank 1-4  at index 64 
        vx_d_cache_->i_p_initial_request = 1;
        vx_d_cache_->i_p_read_or_write = 1;
        vx_d_cache_->i_m_ready = 0;
        for (int j = 0; j < NT; j++) {
            vx_d_cache_->i_p_valid[j] = 1;
            vx_d_cache_->i_p_writedata[j] = 0x7f6f8f6f;
            vx_d_cache_->i_m_readdata[j][0] = 1;
            if (j == 0) {
                vx_d_cache_->i_p_addr[0] = 0x30001004; // bank 1
            } else if (j == 1) {
                vx_d_cache_->i_p_addr[1] = 0x30001008; // bank 2
            } else if (j == 2) {
                vx_d_cache_->i_p_addr[2] = 0x3000100c; // bank 3
            } else {
                vx_d_cache_->i_p_addr[3] = 0x30010010; // bank 4  -- This is serviced 1st, then the other 3 banks are at once
            }
        }

    } else if (counter_value == 1 && do_op) {            // Write to bank 4-7  at index 108 
        vx_d_cache_->i_p_initial_request = 1;
        vx_d_cache_->i_p_read_or_write = 1;
        vx_d_cache_->i_m_ready = 0;
        for (int j = 0; j < NT; j++) {
            vx_d_cache_->i_p_valid[j] = 1;
            vx_d_cache_->i_p_writedata[j] = 0xd1d2d2d3;
            vx_d_cache_->i_m_readdata[j][0] = 1;
            if (j == 0) {
                vx_d_cache_->i_p_addr[0] = 0x30001c14; // bank 5
            } else if (j == 1) {
                vx_d_cache_->i_p_addr[1] = 0x30001c18; // bank 6
            } else if (j == 2) {
                vx_d_cache_->i_p_addr[2] = 0x30001c1c; // bank 7
            } else {
                vx_d_cache_->i_p_addr[3] = 0x30001c10; // bank 4
            }
        }

    } else if (counter_value == 2 && do_op) {        // Read from bank 1-4 at those indexes
        for (int j = 0; j < NT; j++) {
         vx_d_cache_->i_p_initial_request = 1;
        vx_d_cache_->i_p_read_or_write = 0;
        vx_d_cache_->i_m_ready = 0;
        for (int j = 0; j < NT; j++) {
            vx_d_cache_->i_p_valid[j] = 1;
            vx_d_cache_->i_p_writedata[j]   = 0x23232332;
            vx_d_cache_->i_m_readdata[j][0] = 1;
            if (j == 0) {
                vx_d_cache_->i_p_addr[0] = 0x30001004; // bank 1
            } else if (j == 1) {
                vx_d_cache_->i_p_addr[1] = 0x30001c18; // bank 5
            } else if (j == 2) {
                vx_d_cache_->i_p_addr[2] = 0x3000100c; // bank 3
            } else {
                vx_d_cache_->i_p_addr[3] = 0x30001c1c;; // bank 7
            }
        }           
        }
    } else if (counter_value == 3 && do_op) { // Write to Bank 1-5 (evictions will need to take place)
        vx_d_cache_->i_p_initial_request = 1;
        vx_d_cache_->i_p_read_or_write = 1;
        vx_d_cache_->i_m_ready = 0;
        for (int j = 0; j < NT; j++) {
            vx_d_cache_->i_p_valid[j] = 1;   
            vx_d_cache_->i_m_readdata[j][0] = 1;
            if (j == 0) {
                vx_d_cache_->i_p_addr[0] = 0x20001004; // bank 1
                vx_d_cache_->i_p_writedata[j] = 0xaaaabbb0;
            } else if (j == 1) {
                vx_d_cache_->i_p_addr[1] = 0x20001008; // bank 2
                vx_d_cache_->i_p_writedata[j] = 0xaaaabbb1;
            } else if (j == 2) {
                vx_d_cache_->i_p_addr[2] = 0x2000100c; // bank 3
                vx_d_cache_->i_p_writedata[j] = 0xaaaabbb2;
            } else {
                vx_d_cache_->i_p_addr[3] = 0x20001c14; // bank 5
                vx_d_cache_->i_p_writedata[j] = 0xaaaabbb3;
            }
        }
    } else if (counter_value == 4 && do_op) { // Read from addresses that were just overwritten above ^^^
        vx_d_cache_->i_p_initial_request = 1;
        vx_d_cache_->i_p_read_or_write = 0;
        vx_d_cache_->i_m_ready = 0;
        for (int j = 0; j < NT; j++) {
            vx_d_cache_->i_p_valid[j] = 1;
            vx_d_cache_->i_p_writedata[j] = 0x23232332;
            vx_d_cache_->i_m_readdata[j][0] = 1;
            if (j == 0) {
                vx_d_cache_->i_p_addr[0] = 0x20001004; // bank 1
            } else if (j == 1) {
                vx_d_cache_->i_p_addr[1] = 0x20001008; // bank 2
            } else if (j == 2) {
                vx_d_cache_->i_p_addr[2] = 0x2000100c; // bank 3
            } else {
                vx_d_cache_->i_p_addr[3] = 0x20001c14; // bank 5
            }
        }
    }
    /* These will check writing multiple threads writing to the same block
    } else if (counter_value == 3 && do_op) { // Write to Bank 0 
        vx_d_cache_->i_p_initial_request = 1;
        vx_d_cache_->i_p_read_or_write = 1;
        vx_d_cache_->i_m_ready = 0;
        for (int j = 0; j < NT; j++) {
            vx_d_cache_->i_p_valid[j] = 1;   
            vx_d_cache_->i_m_readdata[j][0] = 1;
            if (j == 0) {
                vx_d_cache_->i_p_addr[0] = 0x30001f00; // bank 0
                vx_d_cache_->i_p_writedata[j] = 0xaaaabbb0;
            } else if (j == 1) {
                vx_d_cache_->i_p_addr[1] = 0x30001c00; // bank 0
                vx_d_cache_->i_p_writedata[j] = 0xaaaabbb1;
            } else if (j == 2) {
                vx_d_cache_->i_p_addr[2] = 0x30001a00; // bank 0
                vx_d_cache_->i_p_writedata[j] = 0xaaaabbb2;
            } else {
                vx_d_cache_->i_p_addr[3] = 0x30001904; // bank 1
                vx_d_cache_->i_p_writedata[j] = 0xaaaabbb3;
            }
        }
    } else if (counter_value == 4 && do_op) { // Read from Bank 0 
        vx_d_cache_->i_p_initial_request = 1;
        vx_d_cache_->i_p_read_or_write = 0;
        vx_d_cache_->i_m_ready = 0;
        for (int j = 0; j < NT; j++) {
            vx_d_cache_->i_p_valid[j] = 1;
            vx_d_cache_->i_p_writedata[j] = 0x23232332;
            vx_d_cache_->i_m_readdata[j][0] = 1;
            if (j == 0) {
                vx_d_cache_->i_p_addr[0] = 0x30001f00; // bank 0
            } else if (j == 1) {
                vx_d_cache_->i_p_addr[1] = 0x30001c00; // bank 0
            } else if (j == 2) {
                vx_d_cache_->i_p_addr[2] = 0x30001a00; // bank 0
            } else {
                vx_d_cache_->i_p_addr[3] = 0x30001904; // bank 1
            }
        }
    }
    */
    // Handle Memory Accesses
    unsigned int read_data_from_mem = 0x1111 + counter_value + this->stats_total_cycles;
    
    if (vx_d_cache_->o_m_valid) {
        this->stats_dram_accesses = this->stats_dram_accesses + 1; // (assuming memory access takes 20 cycles)

        this->stats_total_cycles += 1;
        vx_d_cache_->clk = 0;
        vx_d_cache_->eval();
        #ifdef VCD_OUTPUT
        m_trace->dump(2*this->stats_total_cycles);
        #endif
        vx_d_cache_->clk = 1;
        vx_d_cache_->eval();
        #ifdef VCD_OUTPUT
        m_trace->dump((2*this->stats_total_cycles)+1);
        #endif  

        vx_d_cache_->i_m_ready = 1;
        for (int j1 = 0; j1 < 8; j1++) {
            for (int j2 = 0; j2 < 4; j2++) {
                vx_d_cache_->i_m_readdata[j1][j2] = read_data_from_mem;
            }
        }
    } else {
        vx_d_cache_->i_m_ready = 0;
    }


    if (vx_d_cache_->o_p_waitrequest == 0) {
        return true;
    } else {
        return false;
    }


}


bool VX_d_cache::simulate()
{

//    this->instruction_file_name = file_to_simulate;
    // this->results << "\n****************\t" << file_to_simulate << "\t****************\n";

//    this->ProcessFile();

    // auto start_time = std::chrono::high_resolution_clock::now();


    //static bool stop      = false;
    //static int counter    = 0;
    //counter = 0;
    //stop = false;

    // auto start_time = clock();


    vx_d_cache_->clk = 0;
    vx_d_cache_->rst = 1;
    //vortex->eval();
    //counter = 0;
    vx_d_cache_->rst = 0;

    bool cont = false;
    bool out_operation = false;
    bool do_operation = true;
    int other_counter = 0;
    //while (this->stop && ((other_counter < 5)))
    while (other_counter < 5)
    {

        // std::cout << "************* Cycle: " << (this->stats_total_cycles) << "\n";
       // istop =  ibus_driver();
        // dstop = !dbus_driver();

        vx_d_cache_->clk = 1;
        vx_d_cache_->eval();
        #ifdef VCD_OUTPUT
        m_trace->dump(2*this->stats_total_cycles);
        #endif

        //vortex->eval();
        //dstop = !dbus_driver();

        out_operation = operation(other_counter, do_operation);
        vx_d_cache_->clk = 0;
        vx_d_cache_->eval();
        #ifdef VCD_OUTPUT
        m_trace->dump((2*this->stats_total_cycles)+1);
        #endif        
        //vortex->eval();

        /*
        // stop = istop && dstop;
        stop = vortex->out_ebreak;
        if (stop || cont)
        {
            cont = true;
            counter++;
        } else
        {
            counter = 0;
        }
        */
        if (out_operation) {
            other_counter++;
            do_operation = true;
        } else {
            do_operation = false;
        }
        ++(this->stats_total_cycles);

        if (this->stats_total_cycles > 5000) {
            break;
        }

    }

    std::cerr << "New Total Cycles: " << (this->stats_total_cycles + (this->stats_dram_accesses * 20)) << "\n";

    //uint32_t status;
    //ram.getWord(0, &status);

    //this->print_stats();



    return (true);
}









