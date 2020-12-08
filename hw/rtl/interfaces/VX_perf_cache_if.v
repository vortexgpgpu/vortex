`ifndef VX_PERF_CACHE_IF
`define VX_PERF_CACHE_IF

`include "VX_define.vh"

interface VX_perf_cache_if ();

wire [63:0]     read_miss;
wire [63:0]     write_miss;
wire [63:0]     dram_stall;
wire [63:0]     dram_rsp;
wire [63:0]     core_rsp_stall;
wire [63:0]     msrq_stall;
wire [63:0]     total_stall;
wire [63:0]     total_read;
wire [63:0]     total_write;
wire [63:0]     total_eviction;
wire [63:0]     dram_latency;

endinterface

`endif