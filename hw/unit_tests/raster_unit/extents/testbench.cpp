#include <stdlib.h>
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "VVX_raster_extents.h"
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

void eval(VVX_raster_extents* dut, VerilatedVcdC* m_trace)
{
    // dut->clk = not dut->clk;
    dut->eval();
    timestamp += 10;
    m_trace->dump(timestamp);
    // dut->clk = not dut->clk;
    dut->eval();
    timestamp += 10;
    m_trace->dump(timestamp);
    // if (dut->clk == 1)
    // printf("time=%d clk=%d ready=%d x=%d y=%d masks=%d in_x=%d in_y=%d id=%d\n",
    //    timestamp, dut->clk, dut->ready, dut->out_quad_x_loc, dut->out_quad_y_loc,
    //    dut->out_quad_masks,
    //    dut->x_loc, dut->y_loc);
}

int main(int argc, char** argv, char** env) {
    VVX_raster_extents *dut = new VVX_raster_extents();

    Verilated::traceEverOn(true);
    VerilatedVcdC *m_trace = new VerilatedVcdC;
    dut->trace(m_trace, 10);
    m_trace->open("waveform.vcd");

    // dut->x_loc = 0; dut->y_loc = 256;
    // dut->edge_func_val[0] = 518; dut->edge_func_val[1] = 42976; dut->edge_func_val[2] = 0;
    // //dut->edge_func_val[0] = 500; dut->edge_func_val[1] = 200; dut->edge_func_val[2] = 500;
    // dut->edges[0] = -73; dut->edges[1] = -36; dut->edges[2] = 65456;
    //dut->edges[0] = 5; dut->edges[1] = -89; dut->edges[2] = 65440;
    dut->edges[0] = 0; dut->edges[1] = 255; dut->edges[2] = -65280;
    // dut->extents[0] = 0; dut->extents[1] = 320; dut->extents[2] = 16320;
    for (int i = 0; i< 10; ++i)
        dut->eval();
    printf("%u\n", dut->extents);
    m_trace->close();
    delete dut;
    exit(EXIT_SUCCESS);
}