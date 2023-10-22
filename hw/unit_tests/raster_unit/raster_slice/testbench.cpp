#include <stdlib.h>
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "VVX_raster_slice.h"
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

void eval(VVX_raster_slice* dut, VerilatedVcdC* m_trace)
{
    dut->clk = not dut->clk;
    dut->eval();
    timestamp += 5;
    m_trace->dump(timestamp);
    // if (dut->out_valid_block && dut->ready == 0)
    //printf("%u %u %u\n", dut->clk, dut->out_quad_x_loc, dut->out_quad_y_loc, dut->out_quad_masks);
    dut->clk = not dut->clk;
    dut->eval();
    timestamp += 5;
    m_trace->dump(timestamp);
}

int main(int argc, char** argv, char** env) {
    VVX_raster_slice *dut = new VVX_raster_slice();

    Verilated::traceEverOn(true);
    VerilatedVcdC *m_trace = new VerilatedVcdC;
    dut->trace(m_trace, 20);
    m_trace->open("waveform.vcd");

    // Config #1: Tile = 16, Block = 8
    dut->clk = 1;
    dut->reset = 1;
    eval(dut, m_trace);
    dut->reset = 0;
    dut->input_valid = 1;
    dut->x_loc = 0; dut->y_loc = 256;
    dut->edge_func_val[0] = 518; dut->edge_func_val[1] = 42976; dut->edge_func_val[2] = 0;
    //dut->edge_func_val[0] = 500; dut->edge_func_val[1] = 200; dut->edge_func_val[2] = 500;
    dut->edges[0][0] = -73; dut->edges[0][1] = -36; dut->edges[0][2] = 65456;
    dut->edges[1][0] = 5; dut->edges[1][1] = -89; dut->edges[1][2] = 65440;
    dut->edges[2][0] = 0; dut->edges[2][1] = 255; dut->edges[2][2] = -65280;
    dut->extents[0] = 0; dut->extents[1] = 320; dut->extents[2] = 16320;
    eval(dut, m_trace);
    dut->input_valid = 0;
    for (int i = 0; i < 100; ++i)
        eval(dut, m_trace);

    assert(dut->quad_queue_empty == 0);

    dut->pop_quad = 1;
    for (int i = 0; i < 100; ++i)
    {
        if (dut->quad_queue_empty == 1)
        {
            //std::cout << "Breaking at time " << timestamp << " as queue empty\n";
            break;
        }
        eval(dut, m_trace);
        for (int j = 0; j < 4; ++j)
            // Print the set pixel
            for (int quad_i = 0; quad_i < 2; ++quad_i)
            {
                for (int quad_j = 0; quad_j < 2; ++quad_j)
                {
                    int index = quad_i*2 + quad_j;
                    if (dut->out_quad_masks[j] & 1 << (index))
            // printf("Testing %d x_loc=%d, y_loc=%d, mask=%d, valid=%d\n", 
            //     j, dut->out_quad_x_loc[j], dut->out_quad_y_loc[j],
            //     dut->out_quad_masks[j], dut->valid[j]);
                    printf("%d %d\n",  dut->out_quad_x_loc[j]+quad_i, dut->out_quad_y_loc[j]+quad_j);
                }
            }

    }
    dut->pop_quad = 0;
    eval(dut, m_trace);
    eval(dut, m_trace);


    m_trace->close();
    delete dut;
    exit(EXIT_SUCCESS);
}