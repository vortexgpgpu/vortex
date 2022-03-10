#include <stdlib.h>
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "VVX_raster_te.h"
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

void eval(VVX_raster_te* dut, VerilatedVcdC* m_trace)
{
    dut->eval();
    timestamp += 20;
    m_trace->dump(timestamp);
    // printf("time=%d valid_tile=%d valid_block=%d\n", timestamp,
    //     dut->valid_tile, dut->valid_block);
    // for (int k = 0; k < 4; ++k)
    // {
    //     printf("\tx=%d, y=%d\n\t", dut->tile_x_loc[k], dut->tile_y_loc[k]);
    //     for(int i = 0; i < 3; ++i)
    //     {
    //         printf("%d val=%d ", i, dut->tile_edge_func_val[k][i]);
    //     }
    //     printf("\n");
    // }
        
}
int main(int argc, char** argv, char** env) {
VVX_raster_te *dut = new VVX_raster_te();

    Verilated::traceEverOn(true);
    VerilatedVcdC *m_trace = new VerilatedVcdC;
    dut->trace(m_trace, 10);
    m_trace->open("waveform.vcd");

    // Config #1: Tile = 16, Block = 8
    dut->x_loc = 0; dut->y_loc = 256;
    dut->edge_func_val[0] = 56210; dut->edge_func_val[1] = 26112; dut->edge_func_val[2] = 0;
    //dut->edge_func_val[0] = 0; dut->edge_func_val[1] = -7000; dut->edge_func_val[2] = 0;
    dut->edges[0][0] = -73; dut->edges[0][1] = -36; dut->edges[0][2] = 65456;
    dut->edges[1][0] = 102; dut->edges[1][1] = -153; dut->edges[1][2] = 65280;
    dut->edges[2][0] = 0; dut->edges[2][1] = 255; dut->edges[2][2] = -65280;
    dut->extents[0] = 0; dut->extents[1] = 1632; dut->extents[2] = 4080;
    dut->level = 0;
    for (int i = 0; i < 10; ++i)
    {
        eval(dut, m_trace);
    }

    assert(dut->tile_edge_func_val[0][0] == 56210);
    assert(dut->tile_edge_func_val[0][1] == 26112);
    assert(dut->tile_edge_func_val[0][2] == 0);
    assert(dut->tile_edge_func_val[1][0] == 55922);
    assert(dut->tile_edge_func_val[1][1] == 24888);
    assert(dut->tile_edge_func_val[1][2] == 2040);
    assert(dut->tile_edge_func_val[2][0] == 55626);
    assert(dut->tile_edge_func_val[2][1] == 26928);
    assert(dut->tile_edge_func_val[2][2] == 0);
    assert(dut->tile_edge_func_val[3][0] == 55338);
    assert(dut->tile_edge_func_val[3][1] == 25704);
    assert(dut->tile_edge_func_val[3][2] == 2040);

    assert(dut->tile_x_loc[0] == 0);
    assert(dut->tile_y_loc[0] == 256);
    assert(dut->tile_x_loc[1] == 0);
    assert(dut->tile_y_loc[1] == 264);
    assert(dut->tile_x_loc[2] == 8);
    assert(dut->tile_y_loc[2] == 256);
    assert(dut->tile_x_loc[3] == 8);
    assert(dut->tile_y_loc[3] == 264);
    assert(dut->valid_block == 0);
    assert(dut->valid_tile == 1);

   // Config #3; Tile = 16, Block = 8
    dut->x_loc = 8; dut->y_loc = 264;
    dut->edge_func_val[0] = 55338; dut->edge_func_val[1] = 25704; dut->edge_func_val[2] = 2040;
    //dut->edge_func_val[0] = 0; dut->edge_func_val[1] = -7000; dut->edge_func_val[2] = 0;
    dut->edges[0][0] = -73; dut->edges[0][1] = -36; dut->edges[0][2] = 65456;
    dut->edges[1][0] = 102; dut->edges[1][1] = -153; dut->edges[1][2] = 65280;
    dut->edges[2][0] = 0; dut->edges[2][1] = 255; dut->edges[2][2] = -65280;
    dut->extents[0] = 0; dut->extents[1] = 1632; dut->extents[2] = 4080;
    dut->level = 1;
    eval(dut, m_trace);
    eval(dut, m_trace);
    eval(dut, m_trace);
    for (int i = 0; i < 2; ++i)
    {
        eval(dut, m_trace);
        //if (dut->ready == 1) break;
    }
    assert(dut->valid_tile == 0);
    assert(dut->valid_block == 1);

   /*
    // Config #2, Tile = 16, Block = 4 
    dut->x_loc = 8; dut->y_loc = 264;
    dut->edge_func_val[0] = 55338; dut->edge_func_val[1] = 25704; dut->edge_func_val[2] = 2040;
    //dut->edge_func_val[0] = 0; dut->edge_func_val[1] = -7000; dut->edge_func_val[2] = 0;
    dut->edges[0][0] = -73; dut->edges[0][1] = -36; dut->edges[0][2] = 65456;
    dut->edges[1][0] = 102; dut->edges[1][1] = -153; dut->edges[1][2] = 65280;
    dut->edges[2][0] = 0; dut->edges[2][1] = 255; dut->edges[2][2] = -65280;
    dut->extents[0] = 0; dut->extents[1] = 1632; dut->extents[2] = 4080;
    dut->level = 1;
    eval(dut, m_trace);
    eval(dut, m_trace);
    eval(dut, m_trace);
    for (int i = 0; i < 2; ++i)
    {
        eval(dut, m_trace);
        //if (dut->ready == 1) break;
    }

    assert(dut->tile_edge_func_val[0][0] == 55338);
    assert(dut->tile_edge_func_val[0][1] == 25704);
    assert(dut->tile_edge_func_val[0][2] == 2040);
    assert(dut->tile_edge_func_val[1][0] == 55194);
    assert(dut->tile_edge_func_val[1][1] == 25092);
    assert(dut->tile_edge_func_val[1][2] == 3060);
    assert(dut->tile_edge_func_val[2][0] == 55046);
    assert(dut->tile_edge_func_val[2][1] == 26112);
    assert(dut->tile_edge_func_val[2][2] == 2040);
    assert(dut->tile_edge_func_val[3][0] == 54902);
    assert(dut->tile_edge_func_val[3][1] == 25500);
    assert(dut->tile_edge_func_val[3][2] == 3060);

    assert(dut->tile_x_loc[0] == 8);
    assert(dut->tile_y_loc[0] == 264);
    assert(dut->tile_x_loc[1] == 8);
    assert(dut->tile_y_loc[1] == 268);
    assert(dut->tile_x_loc[2] == 12);
    assert(dut->tile_y_loc[2] == 264);
    assert(dut->tile_x_loc[3] == 12);
    assert(dut->tile_y_loc[3] == 268);

    
    assert(dut->valid_block == 0);
    assert(dut->valid_tile == 1);
    */

    std::cout << "NOTE: On assertion failure double check the tile and block sizes\n";

    m_trace->close();
    delete dut;
    exit(EXIT_SUCCESS);
}