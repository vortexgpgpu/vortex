
// #include <iostream>

// #include "VX_define.h"


#include <../simulate/ram.h>
#include <stdio.h>
#include "svdpi.h"

#include "../simulate/VX_define.h"

// #include "vortex_dpi.h"

extern "C" {
    void load_file   (char * filename);
    void ibus_driver (bool clk, unsigned pc_addr, unsigned * instruction);
    void dbus_driver (bool clk, unsigned o_m_read_addr, unsigned o_m_evict_addr, bool o_m_valid, svLogicVecVal * o_m_writedata, bool o_m_read_or_write, svLogicVecVal * i_m_readdata, bool * i_m_ready);
    void io_handler  (bool clk, bool io_valid, unsigned io_data);
    void gracefulExit();
}

RAM ram;
bool     refill;
unsigned refill_addr;

unsigned num_cycles;

unsigned getIndex(int, int, int);
unsigned getIndex(int r, int c, int numCols)
{
	return (r * numCols) + c;
}


void load_file(char * filename)
{
    // printf("\n\n\n\n**********************\n");
	// printf("Inside load_file\n");

    fprintf(stderr, "\n\n\n\n**********************\n");
	loadHexImpl(filename, &ram);
	// printf("Filename: %s\n", filename);
    refill = false;
}

void ibus_driver(bool clk, unsigned pc_addr, unsigned * instruction)
{    
	// printf("Inside ibus_driver\n");
    if (clk)
    {
        num_cycles++;
        (*instruction) = 0;
    }
    else
    {
        uint32_t curr_inst = 0;
        curr_inst          = 0xdeadbeef;

        uint32_t u_pc_addr = (uint32_t) (pc_addr);

        ram.getWord(u_pc_addr, &curr_inst);

        // printf("PC_addr: %x, instruction: %x\n", pc_addr, instruction);

        (*instruction) = curr_inst;
    }

}

void dbus_driver(bool clk, unsigned o_m_read_addr, unsigned o_m_evict_addr, bool o_m_valid, svLogicVecVal * o_m_writedata, bool o_m_read_or_write, svLogicVecVal * i_m_readdata, bool * i_m_ready)
{


    // Default values
    { 
        s_vpi_vecval * real_i_m_readdata = (s_vpi_vecval *) i_m_readdata;
        (*i_m_ready) = false;
        for (int i = 0; i < CACHE_NUM_BANKS; i++)
        {
            for (int j = 0; j < CACHE_WORDS_PER_BLOCK; j++)
            {

                unsigned index = getIndex(i,j, CACHE_WORDS_PER_BLOCK);

                real_i_m_readdata[index].aval = 0x506070;

                // svGetArrElemPtr2(i_m_readdata, i, j);
                // svPutLogicArrElem2VecVal(i_m_readdata, i, j);
                // i_m_readdata[getIndex(i,j, CACHE_WORDS_PER_BLOCK)] = 0;
            }
        }
    }


    if (clk)
    {
        // Do nothing on positive edge
    }
    else
    {

        if (refill)
        {
            // svGetArrElemPtr2((*i_m_readdata), 0,0);
            // fprintf(stderr, "--------------------------------\n");
            refill = false;


            *i_m_ready                       = true;
            s_vpi_vecval * real_i_m_readdata = (s_vpi_vecval *) i_m_readdata;
            for (int curr_e = 0; curr_e < (CACHE_NUM_BANKS*CACHE_WORDS_PER_BLOCK); curr_e++)
            {
                unsigned new_addr = refill_addr + (4*curr_e);


                unsigned addr_without_byte = new_addr >> 2;
                unsigned bank_num          = addr_without_byte & 0x7;
                unsigned addr_wihtout_bank = addr_without_byte >> 3;
                unsigned offset_num        = addr_wihtout_bank & 0x3;

                unsigned value;
                ram.getWord(new_addr, &value);

                // fprintf(stderr, "-------- (%x) i_m_readdata[%d][%d] (%d) = %x\n", new_addr, bank_num, offset_num, curr_e, value);
                unsigned index = getIndex(bank_num,offset_num, CACHE_WORDS_PER_BLOCK);

                // fprintf(stderr, "Index: %d (%d, %d) = %x\n",  index, bank_num, offset_num, value);

                real_i_m_readdata[index].aval = value;

            }
        }
        else
        {
            if (o_m_valid)
            {

                s_vpi_vecval * real_o_m_writedata = (s_vpi_vecval *) o_m_writedata;

                if (o_m_read_or_write)
                {
                    // fprintf(stderr, "++++++++++++++++++++++++++++++++\n");

                    for (int curr_e = 0; curr_e < (CACHE_NUM_BANKS*CACHE_WORDS_PER_BLOCK); curr_e++)
                    {
                        unsigned new_addr = (o_m_evict_addr) + (4*curr_e);


                        unsigned addr_without_byte = new_addr >> 2;
                        unsigned bank_num          = addr_without_byte & 0x7;
                        unsigned addr_wihtout_bank = addr_without_byte >> 3;
                        unsigned offset_num        = addr_wihtout_bank & 0x3;
                        unsigned index             = getIndex(bank_num,offset_num, CACHE_WORDS_PER_BLOCK);



                        unsigned new_value = real_o_m_writedata[index].aval;

                        // new_value = (unsigned *) svGetArrElemPtr2(o_m_writedata, bank_num, offset_num);
                        // new_value          = getElem(o_m_writedata, index);
                        // unsigned new_value         = o_m_writedata[getIndex(bank_num,offset_num, CACHE_WORDS_PER_BLOCK)];


                        ram.writeWord( new_addr, &new_value);

                        // fprintf(stderr, "+++++++ (%x) writeback[%d][%d] (%d) = %x\n", new_addr, bank_num, offset_num, curr_e, new_value);
                    }
                    
                }

                // Respond next cycle
                refill      = true;
                refill_addr = o_m_read_addr;
            }
        }

    }
}


void io_handler(bool clk, bool io_valid, unsigned io_data)
{
	// printf("Inside io_handler\n");
    if (clk)
    {
        // Do nothing
    }
    else
    {
        if (io_valid)
        {
            uint32_t data_write = (uint32_t) (io_data);

            char c = (char) data_write;
            fprintf(stderr, "%c", c );
            fflush(stderr);
        }
    }
}

void gracefulExit()
{
    fprintf(stderr, "Num Cycles: %d\n", num_cycles);
    fprintf(stderr, "\n*********************\n\n");
}




