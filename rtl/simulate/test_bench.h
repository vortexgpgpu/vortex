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
#include "VVortex.h"
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

        VVortex * vortex;

        unsigned start_pc;
        bool     refill;
        unsigned refill_addr;
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
        #ifdef VCD_OUTPUT
        VerilatedVcdC   *m_trace;
        #endif
};



Vortex::Vortex() : start_pc(0), curr_cycle(0), stop(true), unit_test(true), stats_static_inst(0), stats_dynamic_inst(-1),
                                                    stats_total_cycles(0), stats_fwd_stalls(0), stats_branch_stalls(0),
                                                    debug_state(0), ibus_state(0), dbus_state(0), debug_return(0),
                                                    debug_wait_num(0), debug_inst_num(0), debug_end_wait(0), debug_debugAddr(0)
{
	this->vortex  = new VVortex;
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
    loadHexImpl("../../kernel/vortex_test.hex", &this->ram);
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
    
    ////////////////////// IBUS //////////////////////
    unsigned new_PC;
    bool stop = false;
    uint32_t curr_inst = 0;

    curr_inst = 0xdeadbeef;

    new_PC                             = vortex->icache_request_pc_address;
    ram.getWord(new_PC, &curr_inst);
    vortex->icache_response_instruction   = curr_inst;

    // std::cout << std::hex << "IReq: " << vortex->icache_request_pc_address << "\tResp: " << curr_inst << "\n";

    // printf("\n\n---------------------------------------------\n(%x) Inst: %x\n", new_PC, curr_inst);
    // printf("\n");
    ////////////////////// IBUS //////////////////////


    ////////////////////// STATS //////////////////////


    if (((((unsigned int)curr_inst) != 0) && (((unsigned int)curr_inst) != 0xffffffff)))
    {
        ++stats_dynamic_inst;
        stop = false;
    } else
    {
        // printf("Ibus requesting stop: %x\n", curr_inst);
        stop = true;
    }

    return stop;

}

void Vortex::io_handler()
{
    if (vortex->io_valid)
    {
        uint32_t data_write = (uint32_t) vortex->io_data;

        char c = (char) data_write;
        std::cerr << c;
    }
}


bool Vortex::dbus_driver()
{

    // printf("****************************\n");

    vortex->i_m_ready = 0;
    for (int i = 0; i < CACHE_NUM_BANKS; i++)
    {
        for (int j = 0; j < CACHE_WORDS_PER_BLOCK; j++)
        {
            vortex->i_m_readdata[i][j] = 0;
        }
    }


    if (this->refill)
    {
        this->refill = false;

        vortex->i_m_ready = 1;
        for (int curr_e = 0; curr_e < (CACHE_NUM_BANKS*CACHE_WORDS_PER_BLOCK); curr_e++)
        {
            unsigned new_addr = this->refill_addr + (4*curr_e);


            unsigned addr_without_byte = new_addr >> 2;
            unsigned bank_num          = addr_without_byte & 0x7;
            unsigned addr_wihtout_bank = addr_without_byte >> 3;
            unsigned offset_num        = addr_wihtout_bank & 0x3;

            unsigned value;
            ram.getWord(new_addr, &value);

            // printf("-------- (%x) i_m_readdata[%d][%d] (%d) = %d\n", new_addr, bank_num, offset_num, curr_e, value);
            vortex->i_m_readdata[bank_num][offset_num] = value;

        }
    }
    else
    {
        if (vortex->o_m_valid)
        {
            // printf("Valid o_m_valid\n");
            if (vortex->o_m_read_or_write)
            {
                // printf("Valid write\n");

                for (int curr_e = 0; curr_e < (CACHE_NUM_BANKS*CACHE_WORDS_PER_BLOCK); curr_e++)
                {
                    unsigned new_addr = vortex->o_m_evict_addr + (4*curr_e);


                    unsigned addr_without_byte = new_addr >> 2;
                    unsigned bank_num          = addr_without_byte & 0x7;
                    unsigned addr_wihtout_bank = addr_without_byte >> 3;
                    unsigned offset_num        = addr_wihtout_bank & 0x3;


                    unsigned new_value         = vortex->o_m_writedata[bank_num][offset_num];

                    ram.writeWord( new_addr, &new_value);

                    // printf("+++++++ (%x) writeback[%d][%d] (%d) = %d\n", new_addr, bank_num, offset_num, curr_e, new_value);
                    // printf("+++++++ (%x) i_m_readdata[%d][%d] (%d) = %d\n", new_addr, bank_num, offset_num, curr_e, value);
                }
                
            }

            // Respond next cycle
            this->refill = true;
            this->refill_addr = vortex->o_m_read_addr;
        }
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

    // int status = (unsigned int) vortex->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR[28][0] & 0xf;

    // std::cout << "Something: " <<  result << '\n';

    // uint32_t status;
    // ram.getWord(0, &status);

    this->print_stats();



    // return (status == 1);
    return (1 == 1);
}