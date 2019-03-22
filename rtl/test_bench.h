

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

        RAM ram;

        VVortex * vortex;

        unsigned start_pc;
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
};



Vortex::Vortex() : start_pc(0), curr_cycle(0), stop(true), unit_test(true), stats_static_inst(0), stats_dynamic_inst(-1),
                                                    stats_total_cycles(0), stats_fwd_stalls(0), stats_branch_stalls(0),
                                                    debug_state(0), ibus_state(0), dbus_state(0), debug_return(0),
                                                    debug_wait_num(0), debug_inst_num(0), debug_end_wait(0), debug_debugAddr(0)
{
	this->vortex = new VVortex;
    this->results.open("../results.txt");
}

Vortex::~Vortex()
{
	this->results.close();
	delete this->vortex;
}


void Vortex::ProcessFile(void)
{
    loadHexImpl(this->instruction_file_name, &this->ram);
}


bool Vortex::ibus_driver()
{
    
    ////////////////////// IBUS //////////////////////
    unsigned new_PC;
    bool stop = false;
    uint32_t curr_inst = 0;

    curr_inst = 0xdeadbeef;

    new_PC                             = vortex->curr_PC;
    ram.getWord(new_PC, &curr_inst);
    vortex->fe_instruction      = curr_inst;

    ////////////////////// IBUS //////////////////////


    ////////////////////// STATS //////////////////////
    ++stats_total_cycles;


    if (((((unsigned int)curr_inst) != 0) && (((unsigned int)curr_inst) != 0xffffffff)) || (this->ibus_state == 1) || (this->dbus_state == 1))
    {
        ++stats_dynamic_inst;
        stop = false;
    } else
    {
        stop = true;
    }

    return stop;

}



bool Vortex::dbus_driver()
{
    uint32_t data_read;
    uint32_t data_write;
    uint32_t addr;
    // std::cout << "DBUS DRIVER\n" << std::endl;
    ////////////////////// DBUS //////////////////////


    if (vortex->out_cache_driver_in_mem_write != NO_MEM_WRITE)
    {
        data_write = (uint32_t) vortex->out_cache_driver_in_data;
        addr       = (uint32_t) vortex->out_cache_driver_in_address;

        if (vortex->out_cache_driver_in_mem_write == SB_MEM_WRITE)
        {
        	data_write = ( data_write) & 0xFF;
            ram.writeByte( addr, &data_write);

        } else if (vortex->out_cache_driver_in_mem_write == SH_MEM_WRITE)
        {
            data_write = ( data_write) & 0xFFFF;
            ram.writeHalf( addr, &data_write);
        } else if (vortex->out_cache_driver_in_mem_write == SW_MEM_WRITE)
        {
            data_write = data_write;
            ram.writeWord( addr, &data_write);
        }

    }

    if (vortex->out_cache_driver_in_mem_read != NO_MEM_READ)
    {

        addr = (uint32_t) vortex->out_cache_driver_in_address;
        ram.getWord(addr, &data_read);

    	if (vortex->out_cache_driver_in_mem_read == LB_MEM_READ)
    	{

            vortex->in_cache_driver_out_data = (data_read & 0x80) ? (data_read | 0xFFFFFF00) : (data_read & 0xFF);

    	} else if (vortex->out_cache_driver_in_mem_read == LH_MEM_READ)
    	{

            vortex->in_cache_driver_out_data = (data_read & 0x8000) ? (data_read | 0xFFFF0000) : (data_read & 0xFFFF);

    	} else if (vortex->out_cache_driver_in_mem_read == LW_MEM_READ)
    	{

            vortex->in_cache_driver_out_data = data_read;

    	} else if (vortex->out_cache_driver_in_mem_read == LBU_MEM_READ)
    	{

            vortex->in_cache_driver_out_data = (data_read & 0xFF);

    	} else if (vortex->out_cache_driver_in_mem_read == LHU_MEM_READ)
    	{

            vortex->in_cache_driver_out_data = (data_read & 0xFFFF);

    	}
    	else
    	{
            vortex->in_cache_driver_out_data = 0xbabebabe;
    	}
    }
    else
    {
        vortex->in_cache_driver_out_data = 0xbabebabe;
    }

    return false;
}



bool Vortex::simulate(std::string file_to_simulate)
{

    this->instruction_file_name = file_to_simulate;
    this->results << "\n****************\t" << file_to_simulate << "\t****************\n";

    this->ProcessFile();

    // auto start_time = std::chrono::high_resolution_clock::now();


    static bool stop      = false;
    static int counter    = 0;
    counter = 0;
    stop = false;

    // auto start_time = clock();


	vortex->clk = 0;
	vortex->reset = 1;
	vortex->eval();

	vortex->reset = 0;

	unsigned curr_inst;
	unsigned new_PC;
	while (this->stop && (!(stop && (counter > 5))))
	{

        bool istop =  ibus_driver();
        bool dstop = !dbus_driver();
        stop       =  istop && dstop;

		vortex->clk = 1;
		vortex->eval();

		vortex->clk = 0;
		vortex->eval();

        if (stop)
        {
            counter++;
        } else
        {
            counter = 0;
        }

	}

    uint32_t status;
    ram.getWord(0, &status);

    this->print_stats();



    return (status == 1);
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









