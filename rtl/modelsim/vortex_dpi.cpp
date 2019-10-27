
// #include <iostream>

// #include "VX_define.h"


#include <../simulate/ram.h>
#include <stdio.h>
#include "svdpi.h"

#include "../simulate/VX_define.h"

extern "C" {
	void load_file(char * filename);
	void ibus_driver(int pc_addr, int * instruction);
	void dbus_driver(int o_m_read_addr, int o_m_evict_addr, bool o_m_valid, int * o_m_writedata, bool o_m_read_or_write, int * i_m_readdata, bool * i_m_ready);
	void io_handler(bool io_valid, int io_data);
}

RAM ram;

unsigned getIndex(int r, int c, int numCols)
{
	return (r * numCols) + c;
}


void load_file(char * filename)
{
    printf("\n\n\n\n**********************\n");
	// printf("Inside load_file\n");
	loadHexImpl(filename, &ram);
	// printf("Filename: %s\n", filename);
}

void ibus_driver(int pc_addr, int * instruction)
{    
	// printf("Inside ibus_driver\n");
	uint32_t curr_inst = 0;
	curr_inst          = 0xdeadbeef;


	uint32_t u_pc_addr = (uint32_t) (pc_addr);

	ram.getWord(u_pc_addr, &curr_inst);

    // printf("PC_addr: %x, instruction: %x\n", pc_addr, instruction);

	(*instruction) = curr_inst;

}

bool     refill;
unsigned refill_addr;

void dbus_driver(int o_m_read_addr, int o_m_evict_addr, bool o_m_valid, int * o_m_writedata, bool o_m_read_or_write, int * i_m_readdata, bool * i_m_ready)
{
	// printf("Inside dbus_driver\n");

    (*i_m_ready )= 0;
    for (int i = 0; i < CACHE_NUM_BANKS; i++)
    {
        for (int j = 0; j < CACHE_WORDS_PER_BLOCK; j++)
        {
            i_m_readdata[getIndex(i,j, CACHE_WORDS_PER_BLOCK)] = 0;
        }
    }


    if (refill)
    {
        refill = false;

        *i_m_ready = 1;
        for (int curr_e = 0; curr_e < (CACHE_NUM_BANKS*CACHE_WORDS_PER_BLOCK); curr_e++)
        {
            unsigned new_addr = refill_addr + (4*curr_e);


            unsigned addr_without_byte = new_addr >> 2;
            unsigned bank_num          = addr_without_byte & 0x7;
            unsigned addr_wihtout_bank = addr_without_byte >> 3;
            unsigned offset_num        = addr_wihtout_bank & 0x3;

            unsigned value;
            ram.getWord(new_addr, &value);

            // printf("-------- (%x) i_m_readdata[%d][%d] (%d) = %d\n", new_addr, bank_num, offset_num, curr_e, value);
            i_m_readdata[getIndex(bank_num,offset_num, CACHE_NUM_BANKS)] = value;

        }
    }
    else
    {
        if (o_m_valid)
        {
            // printf("Valid o_m_valid\n");
            if (o_m_read_or_write)
            {
                // printf("Valid write\n");

                for (int curr_e = 0; curr_e < (CACHE_NUM_BANKS*CACHE_WORDS_PER_BLOCK); curr_e++)
                {
                    unsigned new_addr = (o_m_evict_addr) + (4*curr_e);


                    unsigned addr_without_byte = new_addr >> 2;
                    unsigned bank_num          = addr_without_byte & 0x7;
                    unsigned addr_wihtout_bank = addr_without_byte >> 3;
                    unsigned offset_num        = addr_wihtout_bank & 0x3;


                    unsigned new_value         = o_m_writedata[getIndex(bank_num,offset_num, CACHE_NUM_BANKS)];

                    ram.writeWord( new_addr, &new_value);

                    // printf("+++++++ (%x) writeback[%d][%d] (%d) = %d\n", new_addr, bank_num, offset_num, curr_e, new_value);
                    // printf("+++++++ (%x) i_m_readdata[%d][%d] (%d) = %d\n", new_addr, bank_num, offset_num, curr_e, value);
                }
                
            }

            // Respond next cycle
            refill = true;
            refill_addr = o_m_read_addr;
        }
    }
}


void io_handler(bool io_valid, int io_data)
{
	// printf("Inside io_handler\n");
    if (io_valid)
    {
        uint32_t data_write = (uint32_t) (io_data);

        char c = (char) data_write;
        printf("%c", c);
        printf("YOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYO\n");
        fflush(stdout);
    }
}


