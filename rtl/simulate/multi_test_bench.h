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

#include "tb_debug.h"

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

unsigned long time_stamp = 0;

double sc_time_stamp() 
{ 
  return time_stamp / 1000.0; 
}

typedef struct
{
    int cycles_left;
    int data_length;
    unsigned base_addr;
    unsigned * data;
} dram_req_t;

class Vortex
{
    public:
        Vortex();
        ~Vortex();
        bool simulate(std::string);
    private:
    	void ProcessFile(void);
        void print_stats(bool = true);
        bool ibus_driver();
        bool dbus_driver();
        void io_handler();

        RAM ram;

        VVortex_SOC * vortex;

        unsigned start_pc;
        bool     refill_d;
        unsigned refill_addr_d;
        bool     refill_i;
        unsigned refill_addr_i;
        long int curr_cycle;
        bool stop;
        bool unit_test;
        std::string instruction_file_name;
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



Vortex::Vortex() : start_pc(0), curr_cycle(0), stop(true), unit_test(true), stats_static_inst(0), stats_dynamic_inst(-1),
                                                    stats_total_cycles(0), stats_fwd_stalls(0), stats_branch_stalls(0),
                                                    debug_state(0), ibus_state(0), dbus_state(0), debug_return(0),
                                                    debug_wait_num(0), debug_inst_num(0), debug_end_wait(0), debug_debugAddr(0)
{
	this->vortex  = new VVortex_SOC;
    #ifdef VCD_OUTPUT
    this->m_trace = new VerilatedVcdC;
    this->vortex->trace(m_trace, 99);
    this->m_trace->open("trace.vcd");
    #endif
    this->results.open("../results.txt");
}

Vortex::~Vortex()
{
    #ifdef VCD_OUTPUT
    m_trace->close();
    #endif
	this->results.close();
	delete this->vortex;
}


void Vortex::ProcessFile(void)
{
    loadHexImpl(this->instruction_file_name.c_str(), &this->ram);
}

void Vortex::print_stats(bool cycle_test)
{

    if (cycle_test)
    {
        this->results << std::left;
        // this->results << "# Static Instructions:\t" << std::dec << this->stats_static_inst << std::endl;
        this->results << std::setw(24) << "# Dynamic Instructions:" << std::dec << this->stats_dynamic_inst << std::endl;
        this->results << std::setw(24) << "# of total cycles:" << std::dec << this->stats_total_cycles << std::endl;
        this->results << std::setw(24) << "# of forwarding stalls:" << std::dec << this->stats_fwd_stalls << std::endl;
        this->results << std::setw(24) << "# of branch stalls:" << std::dec << this->stats_branch_stalls << std::endl;
        this->results << std::setw(24) << "# CPI:" << std::dec << (double) this->stats_total_cycles / (double) this->stats_dynamic_inst << std::endl;
        this->results << std::setw(24) << "# time to simulate: " << std::dec << this->stats_sim_time << " milliseconds" << std::endl;
    }
    else
    {
        this->results << std::left;
        this->results << std::setw(24) << "# of total cycles:" << std::dec << this->stats_total_cycles << std::endl;
        this->results << std::setw(24) << "# time to simulate: " << std::dec << this->stats_sim_time << " milliseconds" << std::endl;
    }


    uint32_t status;
    ram.getWord(0, &status);

    if (this->unit_test)
    {
        if (status == 1)
        {
            this->results << std::setw(24) << "# GRADE:" << "PASSING\n";
        } else
        {
            this->results << std::setw(24) << "# GRADE:" << "Failed on test: " << status << "\n";
        }
    }
    else
    {
        this->results << std::setw(24) << "# GRADE:" << "N/A [NOT A UNIT TEST]\n";
    }

    this->stats_static_inst   =  0;
    this->stats_dynamic_inst  = -1;
    this->stats_total_cycles  =  0;
    this->stats_fwd_stalls    =  0;
    this->stats_branch_stalls =  0;

}

bool Vortex::ibus_driver()
{

    return false;

}

void Vortex::io_handler()
{
    // std::cout << "Checking\n";
    for (int c = 0; c < vortex->number_cores; c++)
    {
        if (vortex->io_valid[c])
        {
            uint32_t data_write = (uint32_t) vortex->io_data[c];
            // std::cout << "IO VALID!\n";
            char c = (char) data_write;
            std::cerr << c;
            // std::cout << c;

            std::cout << std::flush;
        }
    }
}


bool Vortex::dbus_driver()
{

    // Iterate through each element, and get pop index
    int dequeue_index  = -1;
    bool dequeue_valid = false;
    for (int i = 0; i < this->dram_req_vec.size(); i++)
    {
        if (this->dram_req_vec[i].cycles_left > 0)
        {
            this->dram_req_vec[i].cycles_left -= 1;   
        }

        if ((this->dram_req_vec[i].cycles_left == 0) && (!dequeue_valid))
        {
            dequeue_index = i;
            dequeue_valid = true;
        }
    }


    if (vortex->out_dram_req)
    {
        if (vortex->out_dram_req_read)
        {
            // Need to add an element
            dram_req_t dram_req;
            dram_req.cycles_left = vortex->out_dram_expected_lat;
            dram_req.data_length = vortex->out_dram_req_size / 4;
            dram_req.base_addr   = vortex->out_dram_req_addr;
            dram_req.data        = (unsigned *) malloc(dram_req.data_length * sizeof(unsigned));

            for (int i = 0; i < dram_req.data_length; i++)
            {
                unsigned curr_addr = dram_req.base_addr + (i*4);
                unsigned data_rd;
                ram.getWord(curr_addr, &data_rd);
                dram_req.data[i] = data_rd;
            }
            // std::cout << "Fill Req -> Addr: " << std::hex << dram_req.base_addr << std::dec << "\n";
            this->dram_req_vec.push_back(dram_req);
        }

        if (vortex->out_dram_req_write)
        {
            unsigned base_addr   = vortex->out_dram_req_addr;
            unsigned data_length = vortex->out_dram_req_size / 4;

            for (int i = 0; i < data_length; i++)
            {
                unsigned curr_addr = base_addr + (i*4);
                unsigned data_wr   = vortex->out_dram_req_data[i];
                ram.writeWord(curr_addr, &data_wr);
            }
        }
    }

    if (vortex->out_dram_fill_accept && dequeue_valid)
    {
        vortex->out_dram_fill_rsp      = 1;
        vortex->out_dram_fill_rsp_addr = this->dram_req_vec[dequeue_index].base_addr;
        // std::cout << "Fill Rsp -> Addr: " << std::hex << (this->dram_req_vec[dequeue_index].base_addr) << std::dec << "\n";

        for (int i = 0; i < this->dram_req_vec[dequeue_index].data_length; i++)
        {
            vortex->out_dram_fill_rsp_data[i] = this->dram_req_vec[dequeue_index].data[i];
        }
        free(this->dram_req_vec[dequeue_index].data);

        this->dram_req_vec.erase(this->dram_req_vec.begin() + dequeue_index);
    }
    else
    {
        vortex->out_dram_fill_rsp      = 0;
        vortex->out_dram_fill_rsp_addr = 0;
    }

    return false;
}



bool Vortex::simulate(std::string file_to_simulate)
{

    this->instruction_file_name = file_to_simulate;
    // this->results << "\n****************\t" << file_to_simulate << "\t****************\n";

    this->ProcessFile();

    // auto start_time = std::chrono::high_resolution_clock::now();


    static bool stop      = false;
    static int counter    = 0;
    counter = 0;
    stop = false;

    // auto start_time = clock();


	// vortex->reset = 1;


	// vortex->reset = 0;

	unsigned curr_inst;
	unsigned new_PC;

	// while (this->stop && (!(stop && (counter > 5))))
	// {

	// 	// std::cout << "************* Cycle: " << cycle << "\n";
 //        bool istop =  ibus_driver();
 //        bool dstop = !dbus_driver();

	// 	vortex->clk = 1;
	// 	vortex->eval();



	// 	vortex->clk = 0;
	// 	vortex->eval();


 //        stop = istop && dstop;

 //        if (stop)
 //        {
 //            counter++;
 //        } else
 //        {
 //            counter = 0;
 //        }

 //        cycle++;
	// }

    bool istop;
    bool dstop;
    bool cont = false;
    // for (int i = 0; i < 500; i++)

    vortex->reset = 1;
    vortex->clk   = 0;
    vortex->eval();
    // m_trace->dump(10);
    vortex->reset = 1;
    vortex->clk   = 1;
    vortex->eval();
    // m_trace->dump(11);
    vortex->reset = 0;
    vortex->clk   = 0;

    // unsigned cycles;
    counter = 0;
    this->stats_total_cycles = 12;
    while (this->stop && ((counter < 5)))
    // while (this->stats_total_cycles < 10)
    {

        // printf("-------------------------\n");
        // std::cout << "Counter: " << counter << "\n";
        // if ((this->stats_total_cycles) % 5000 == 0) std::cout << "************* Cycle: " << (this->stats_total_cycles) << "\n";
        // dstop = !dbus_driver();
        #ifdef VCD_OUTPUT
        m_trace->dump(2*this->stats_total_cycles);
        #endif
        vortex->clk = 1;
        vortex->eval();
        istop =  ibus_driver();
        dstop = !dbus_driver();
                  io_handler();

        #ifdef VCD_OUTPUT
        m_trace->dump((2*this->stats_total_cycles)+1);
        #endif
        vortex->clk = 0;
        vortex->eval();
        // stop = istop && dstop;
        stop = vortex->out_ebreak;

        if (stop || cont)
        // if (istop)
        {
            cont = true;
            counter++;
        } else
        {
            counter = 0;
        }

        ++time_stamp;
        ++stats_total_cycles;
    }

    std::cerr << "New Total Cycles: " << (this->stats_total_cycles) << "\n";

    int status = 0;
    // int status = (unsigned int) vortex->Vortex_SOC__DOT__vx_back_end__DOT__VX_wb__DOT__last_data_wb & 0xf;

    // std::cout << "Last wb: " << std::hex << ((unsigned int) vortex->Vortex__DOT__vx_back_end__DOT__VX_wb__DOT__last_data_wb) << "\n";

    // std::cout << "Something: " <<  result << '\n';

    // uint32_t status;
    // ram.getWord(0, &status);

    this->print_stats();



    return (status == 1);
    // return (1 == 1);
}
