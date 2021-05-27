
// #include <iostream>

// #include "VX_define.h"


#include <../simulate/ram.h>
#include <stdio.h>
#include <math.h>
#include "svdpi.h"

#include "../simulate/VX_define.h"

// #include "vortex_dpi.h"

extern "C" {
    void load_file   (char * filename);
    void ibus_driver (bool clk, unsigned o_m_read_addr, unsigned o_m_evict_addr, bool o_m_valid, svLogicVecVal * o_m_writedata, bool o_m_read_or_write, unsigned cache_banks, unsigned num_words_per_block, svLogicVecVal * i_m_readdata, bool * i_m_ready);
    void dbus_driver (bool clk, unsigned o_m_read_addr, unsigned o_m_evict_addr, bool o_m_valid, svLogicVecVal * o_m_writedata, bool o_m_read_or_write, unsigned cache_banks, unsigned num_words_per_block, svLogicVecVal * i_m_readdata, bool * i_m_ready);
    void io_handler  (bool clk, bool io_valid, unsigned io_data);
    void gracefulExit(int);
}

RAM ram;
bool     refill;
unsigned refill_addr;
bool     i_refill;
unsigned i_refill_addr;

unsigned num_cycles;

unsigned getIndex(int, int, int);
unsigned calculate_bits_per_bank_num(int);

unsigned getIndex(int r, int c, int numCols)
{
	return (r * numCols) + c;
}

unsigned calculate_bits_per_bank_num(int num)
{
	int shifted_num = 0;
	for(int i = 0; i < num; i++){
		shifted_num = (shifted_num << 1)| 1 ;
	}
	return shifted_num;
}


void load_file(char * filename)
{
    num_cycles = 0;
    // printf("\n\n\n\n**********************\n");
	// printf("Inside load_file\n");

    fprintf(stderr, "\n\n\n\n**********************\n");
	loadHexImage(filename, &ram);
	// printf("Filename: %s\n", filename);
    refill = false;
    i_refill = false;
}

void ibus_driver(bool clk, unsigned o_m_read_addr, unsigned o_m_evict_addr, bool o_m_valid, svLogicVecVal * o_m_writedata, bool o_m_read_or_write, unsigned cache_banks, unsigned num_words_per_block, svLogicVecVal * i_m_readdata, bool * i_m_ready)
{


    // Default values
    { 
        s_vpi_vecval * real_i_m_readdata = (s_vpi_vecval *) i_m_readdata;
        (*i_m_ready) = false;
        for (int i = 0; i < cache_banks; i++)
        {
            for (int j = 0; j < num_words_per_block; j++)
            {

                unsigned index = getIndex(i,j, num_words_per_block);

                real_i_m_readdata[index].aval = 0x506070;

                // svGetArrElemPtr2(i_m_readdata, i, j);
                // svPutLogicArrElem2VecVal(i_m_readdata, i, j);
                // i_m_readdata[getIndex(i,j, num_words_per_block)] = 0;
            }
        }
    }


    if (clk)
    {
        // Do nothing on positive edge
    }
    else
    {

        if (i_refill)
        {
            // svGetArrElemPtr2((*i_m_readdata), 0,0);
            // fprintf(stderr, "--------------------------------\n");
            i_refill = false;


            *i_m_ready                       = true;
            s_vpi_vecval * real_i_m_readdata = (s_vpi_vecval *) i_m_readdata;
            for (int curr_e = 0; curr_e < (cache_banks*num_words_per_block); curr_e++)
            {
                unsigned new_addr = i_refill_addr + (4*curr_e);


                unsigned addr_without_byte = new_addr >> 2;
                
                unsigned bits_per_bank     = (int)log2(cache_banks); 
                // unsigned maskbits_per_bank = calculate_bits_per_bank_num(bits_per_bank); 
                unsigned maskbits_per_bank = cache_banks - 1;
                unsigned bank_num          = addr_without_byte & maskbits_per_bank;
                unsigned addr_wihtout_bank = addr_without_byte >> bits_per_bank;
                unsigned offset_num        = addr_wihtout_bank & (num_words_per_block-1);

                unsigned value;
                ram.getWord(new_addr, &value);

                fprintf(stdout, "-------- (%x) i_m_readdata[%d][%d] (%d) = %x\n", new_addr, bank_num, offset_num, curr_e, value);
                unsigned index = getIndex(bank_num,offset_num, num_words_per_block);

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

                    for (int curr_e = 0; curr_e < (cache_banks*num_words_per_block); curr_e++)
                    {
                        unsigned new_addr = (o_m_evict_addr) + (4*curr_e);


                        unsigned addr_without_byte = new_addr >> 2;
                        unsigned bits_per_bank = (int)log2(cache_banks);
                        // unsigned maskbits_per_bank = calculate_bits_per_bank_num(bits_per_bank); 
                        unsigned maskbits_per_bank = cache_banks - 1;
                        unsigned bank_num          = addr_without_byte & maskbits_per_bank;
                        unsigned addr_wihtout_bank = addr_without_byte >> bits_per_bank;
                        unsigned offset_num        = addr_wihtout_bank & (num_words_per_block-1);
                        // unsigned offset_num        = addr_wihtout_bank & 0x3;
                        unsigned index             = getIndex(bank_num,offset_num, num_words_per_block);



                        unsigned new_value = real_o_m_writedata[index].aval;

                        // new_value = (unsigned *) svGetArrElemPtr2(o_m_writedata, bank_num, offset_num);
                        // new_value          = getElem(o_m_writedata, index);
                        // unsigned new_value         = o_m_writedata[getIndex(bank_num,offset_num, num_words_per_block)];


                        ram.writeWord( new_addr, &new_value);

                        fprintf(stdout, "+++++++ (%x) writeback[%d][%d] (%d) = %x\n", new_addr, bank_num, offset_num, curr_e, new_value);
                    }
                    
                }

                // Respond next cycle
                i_refill      = true;
                i_refill_addr = o_m_read_addr;
            }
        }

    }
}


void dbus_driver(bool clk, unsigned o_m_read_addr, unsigned o_m_evict_addr, bool o_m_valid, svLogicVecVal * o_m_writedata, bool o_m_read_or_write, unsigned cache_banks, unsigned num_words_per_block, svLogicVecVal * i_m_readdata, bool * i_m_ready)
{


    // Default values
    { 
        s_vpi_vecval * real_i_m_readdata = (s_vpi_vecval *) i_m_readdata;
        (*i_m_ready) = false;
        for (int i = 0; i < cache_banks; i++)
        {
            for (int j = 0; j < num_words_per_block; j++)
            {

                unsigned index = getIndex(i,j, num_words_per_block);

                real_i_m_readdata[index].aval = 0x506070;

                // svGetArrElemPtr2(i_m_readdata, i, j);
                // svPutLogicArrElem2VecVal(i_m_readdata, i, j);
                // i_m_readdata[getIndex(i,j, num_words_per_block)] = 0;
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
            for (int curr_e = 0; curr_e < (cache_banks*num_words_per_block); curr_e++)
            {
                unsigned new_addr = refill_addr + (4*curr_e);


                unsigned addr_without_byte = new_addr >> 2;
                
        		unsigned bits_per_bank     = (int)log2(cache_banks); 
        		// unsigned maskbits_per_bank = calculate_bits_per_bank_num(bits_per_bank); 
                unsigned maskbits_per_bank = cache_banks - 1;
                unsigned bank_num          = addr_without_byte & maskbits_per_bank;
                unsigned addr_wihtout_bank = addr_without_byte >> bits_per_bank;
                unsigned offset_num        = addr_wihtout_bank & (num_words_per_block-1);

                unsigned value;
                ram.getWord(new_addr, &value);

                fprintf(stdout, "-------- (%x) i_m_readdata[%d][%d] (%d) = %x\n", new_addr, bank_num, offset_num, curr_e, value);
                unsigned index = getIndex(bank_num,offset_num, num_words_per_block);

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

                    for (int curr_e = 0; curr_e < (cache_banks*num_words_per_block); curr_e++)
                    {
                        unsigned new_addr = (o_m_evict_addr) + (4*curr_e);


                        unsigned addr_without_byte = new_addr >> 2;
            			unsigned bits_per_bank = (int)log2(cache_banks);
            			// unsigned maskbits_per_bank = calculate_bits_per_bank_num(bits_per_bank); 
                        unsigned maskbits_per_bank = cache_banks - 1;
                        unsigned bank_num          = addr_without_byte & maskbits_per_bank;
                        unsigned addr_wihtout_bank = addr_without_byte >> bits_per_bank;
                        unsigned offset_num        = addr_wihtout_bank & (num_words_per_block-1);
                        // unsigned offset_num        = addr_wihtout_bank & 0x3;
                        unsigned index             = getIndex(bank_num,offset_num, num_words_per_block);



                        unsigned new_value = real_o_m_writedata[index].aval;

                        // new_value = (unsigned *) svGetArrElemPtr2(o_m_writedata, bank_num, offset_num);
                        // new_value          = getElem(o_m_writedata, index);
                        // unsigned new_value         = o_m_writedata[getIndex(bank_num,offset_num, num_words_per_block)];


                        ram.writeWord( new_addr, &new_value);

                        fprintf(stdout, "+++++++ (%x) writeback[%d][%d] (%d) = %x\n", new_addr, bank_num, offset_num, curr_e, new_value);
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

            fprintf(stderr, "%c", (char) data_write);
            fflush(stderr);
        }
    }
}

void gracefulExit(int cycles)
{
    fprintf(stderr, "*********************\n\n");
    fprintf(stderr, "DPI Cycle Num: %d\tVerilog Cycle Num: %d\n", num_cycles, cycles);
}




