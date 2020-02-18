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
    
    vortex->i_m_ready_i = false;

    {

        // int dcache_num_words_per_block

        if (refill_i)
        {
            refill_i            = false;
            vortex->i_m_ready_i = true;

            for (int curr_bank = 0; curr_bank < vortex->Vortex__DOT__icache_banks; curr_bank++)
            {
                for (int curr_word = 0; curr_word < vortex->Vortex__DOT__icache_num_words_per_block; curr_word++)
                {
                    unsigned curr_index = (curr_word * vortex->Vortex__DOT__icache_banks) + curr_bank;
                    unsigned curr_addr  = refill_addr_i + (4*curr_index);

                    unsigned curr_value;
                    ram.getWord(curr_addr, &curr_value);

                    vortex->i_m_readdata_i[curr_bank][curr_word] = curr_value;

                }
            }
        }
        else
        {
            if (vortex->o_m_valid_i)
            {

                if (vortex->o_m_read_or_write_i)
                {
                    // fprintf(stderr, "++++++++++++++++++++++++++++++++\n");
                    unsigned base_addr = vortex->o_m_evict_addr_i;

                    for (int curr_bank = 0; curr_bank < vortex->Vortex__DOT__icache_banks; curr_bank++)
                    {
                        for (int curr_word = 0; curr_word < vortex->Vortex__DOT__icache_num_words_per_block; curr_word++)
                        {
                            unsigned curr_index = (curr_word * vortex->Vortex__DOT__icache_banks) + curr_bank;
                            unsigned curr_addr  = base_addr + (4*curr_index);

                            unsigned curr_value = vortex->o_m_writedata_i[curr_bank][curr_word];

                            ram.writeWord( curr_addr, &curr_value);
                        }
                    }
                }

                // Respond next cycle
                refill_i      = true;
                refill_addr_i = vortex->o_m_read_addr_i;
            }
        }

    }


    return false;

}

void Vortex::io_handler()
{
    if (vortex->io_valid)
    {
        uint32_t data_write = (uint32_t) vortex->io_data;

        char c = (char) data_write;
        std::cerr << c;
        // std::cout << c;
    }
}


bool Vortex::dbus_driver()
{

    vortex->i_m_ready_d = false;

    {

        // int dcache_num_words_per_block

        if (refill_d)
        {
            refill_d            = false;
            vortex->i_m_ready_d = true;

            for (int curr_bank = 0; curr_bank < vortex->Vortex__DOT__dcache_banks; curr_bank++)
            {
                for (int curr_word = 0; curr_word < vortex->Vortex__DOT__dcache_num_words_per_block; curr_word++)
                {
                    unsigned curr_index = (curr_word * vortex->Vortex__DOT__dcache_banks) + curr_bank;
                    unsigned curr_addr  = refill_addr_d + (4*curr_index);

                    unsigned curr_value;
                    ram.getWord(curr_addr, &curr_value);

                    vortex->i_m_readdata_d[curr_bank][curr_word] = curr_value;

                }
            }
        }
        else
        {
            if (vortex->o_m_valid_d)
            {

                if (vortex->o_m_read_or_write_d)
                {
                    // fprintf(stderr, "++++++++++++++++++++++++++++++++\n");
                    unsigned base_addr = vortex->o_m_evict_addr_d;

                    for (int curr_bank = 0; curr_bank < vortex->Vortex__DOT__dcache_banks; curr_bank++)
                    {
                        for (int curr_word = 0; curr_word < vortex->Vortex__DOT__dcache_num_words_per_block; curr_word++)
                        {
                            unsigned curr_index = (curr_word * vortex->Vortex__DOT__dcache_banks) + curr_bank;
                            unsigned curr_addr  = base_addr + (4*curr_index);

                            unsigned curr_value = vortex->o_m_writedata_d[curr_bank][curr_word];

                            ram.writeWord( curr_addr, &curr_value);
                        }
                    }
                }

                // Respond next cycle
                refill_d      = true;
                refill_addr_d = vortex->o_m_read_addr_d;
            }
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

    uint32_t status;
    ram.getWord(0, &status);

    this->print_stats();



    return (status == 1);
    // return (1 == 1);
}