#include <stdlib.h>
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "VVX_raster_te_arbiter.h"
#include "VX_config.h"

#define MAX_SIM_TIME 200
vluint64_t sim_time = 0;

static bool trace_enabled = false;
static uint64_t trace_start_time = 0;
static uint64_t trace_stop_time = -1ull;
static uint64_t timestamp = 0;

double sc_time_stamp() { 
  	return timestamp;
}

bool sim_trace_enabled() {
	if (timestamp >= trace_start_time 
	&& timestamp < trace_stop_time)
			return true;
	return trace_enabled;
}

void sim_trace_enable (bool enable) {
  	trace_enabled = enable;
}

void eval(VVX_raster_te_arbiter* dut, VerilatedVcdC* m_trace)
{    printf("Running: Clk=%d, Push=%d data=%d, Pop=%d data=%d\n", dut->clk, dut->fifo_push, dut->data_push, dut->fifo_pop, dut->data_pop);
    

    dut->clk = not dut->clk;
    dut->eval();
    timestamp += 5;
    m_trace->dump(timestamp);
    dut->clk = not dut->clk;
    dut->eval();
    timestamp += 5;
    m_trace->dump(timestamp);
    printf("Clk=%d, Push=%d data=%d, Pop=%d data=%d\n", dut->clk, dut->fifo_push, dut->data_push, dut->fifo_pop, dut->data_pop);
    
}

int main(int argc, char** argv, char** env) {
    VVX_raster_te_arbiter *dut = new VVX_raster_te_arbiter();

    Verilated::traceEverOn(true);
    VerilatedVcdC *m_trace = new VerilatedVcdC;
    dut->trace(m_trace, 20);
    m_trace->open("waveform.vcd");

    // while (sim_time < MAX_SIM_TIME) {
    //     // dut->clk ^= 1;
    //     // dut->x_loc = 0; dut->y_loc = 256;
    //     dut->edge_func_val[0] = 56210;
    //     dut->eval();
    //     m_trace->dump(sim_time);
    //     sim_time++;
    // }

    // Config #1: Tile = 16, Block = 8
    dut->clk = 1;
    dut->reset = 1;
    eval(dut, m_trace);
    dut->reset = 0;
    // dut->fifo_push = 1; dut->fifo_pop = 0; dut->data_push = 12;
    // eval(dut, m_trace);
    // dut->fifo_push = 0; dut->fifo_pop = 1;
    // eval(dut, m_trace);
    // dut->fifo_push = 0; dut->fifo_pop = 0;
    // eval(dut, m_trace);

    // Write 4 items
    /*
    // Single FIFO test Test #1
    for (int i = 0; i < 4; ++i)
    {
        dut->fifo_push = 1; dut->fifo_pop = 0; dut->data_push = 12 + i*i;
        eval(dut, m_trace);
    }

    // Read 4 items
    for (int i = 0; i < 4; ++i)
    {
        dut->fifo_push = 0; dut->fifo_pop = 1;
        eval(dut, m_trace);
    }


    for (int i = 0; i < 2; ++i)
    {
        eval(dut, m_trace);
    }
    */
    dut->fifo_push = 1;
    for (int j = 0; j < 4; ++j)
    {
        for (int i = 0; i < 4; ++i)
        {
            dut->fifo_pop = 0; dut->data_push[i] = j*4 + i;
        }
        eval(dut, m_trace);
    }
    dut->fifo_push = 0; 
    for (int i = 10; i >= 0; --i)
    {
        dut->fifo_pop = dut->fifo_index_onehot;
        eval(dut, m_trace); 
    }



    m_trace->close();
    delete dut;
    exit(EXIT_SUCCESS);
}