`include "VX_define.vh"

module VX_lsu_unit (
    input wire              clk,
    input wire              reset,
    input wire              no_slot_mem,
    VX_lsu_req_if           lsu_req_if,

    // Write back to GPR
    VX_inst_mem_wb_if       mem_wb_if,

    VX_gpu_dcache_rsp_if    dcache_rsp_if,
    VX_gpu_dcache_req_if    dcache_req_if,
    output wire             delay
);
    // Generate Addresses
    wire[`NUM_THREADS-1:0][31:0] address;
    VX_lsu_addr_gen VX_lsu_addr_gen    (
        .base_address (lsu_req_if.base_address),
        .offset       (lsu_req_if.offset),
        .address      (address)
    );

    wire[`NUM_THREADS-1:0][31:0]     use_address;
    wire[`NUM_THREADS-1:0][31:0]     use_store_data;
    wire[`NUM_THREADS-1:0]           use_valid;
    wire[2:0]            use_mem_read; 
    wire[2:0]            use_mem_write;
    wire[4:0]            use_rd;
    wire[`NW_BITS-1:0]   use_warp_num;
    wire[1:0]            use_wb;
    wire[31:0]           use_pc;    

    wire zero = 0;

    VX_generic_register #(
        .N(45 + `NW_BITS-1 + 1 + `NUM_THREADS*65)
    ) lsu_buffer(
        .clk  (clk),
        .reset(reset),
        .stall(delay),
        .flush(zero),
        .in   ({address    , lsu_req_if.store_data, lsu_req_if.valid, lsu_req_if.mem_read, lsu_req_if.mem_write, lsu_req_if.rd, lsu_req_if.warp_num, lsu_req_if.wb, lsu_req_if.lsu_pc}),
        .out  ({use_address, use_store_data       , use_valid       , use_mem_read       , use_mem_write       , use_rd       , use_warp_num       , use_wb       , use_pc           })
    );

    // Core Request
    assign dcache_req_if.core_req_valid      = use_valid;
    assign dcache_req_if.core_req_addr       = use_address;
    assign dcache_req_if.core_req_data       = use_store_data;
    assign dcache_req_if.core_req_read       = {`NUM_THREADS{use_mem_read}};
    assign dcache_req_if.core_req_write      = {`NUM_THREADS{use_mem_write}};
    assign dcache_req_if.core_req_rd         = use_rd;
    assign dcache_req_if.core_req_wb         = {`NUM_THREADS{use_wb}};
    assign dcache_req_if.core_req_warp_num   = use_warp_num;
    assign dcache_req_if.core_req_pc         = use_pc;

    // Core can't accept response
    assign dcache_rsp_if.core_rsp_ready      = ~no_slot_mem;    

    // Cache can't accept request
    assign delay = ~dcache_req_if.core_req_ready;

    // Core Response
    assign mem_wb_if.rd          = dcache_rsp_if.core_rsp_read;
    assign mem_wb_if.wb          = dcache_rsp_if.core_rsp_write;
    assign mem_wb_if.wb_valid    = dcache_rsp_if.core_rsp_valid;
    assign mem_wb_if.wb_warp_num = dcache_rsp_if.core_rsp_warp_num;
    assign mem_wb_if.loaded_data = dcache_rsp_if.core_rsp_data;
    
    wire[(`LOG2UP(`NUM_THREADS))-1:0] use_pc_index;

`DEBUG_BEGIN
    wire found;
`DEBUG_END

    VX_generic_priority_encoder #(
        .N(`NUM_THREADS)
    ) pick_first_pc (
        .valids(dcache_rsp_if.core_rsp_valid),
        .index (use_pc_index),
        .found (found)
    );

    assign mem_wb_if.mem_wb_pc   = dcache_rsp_if.core_rsp_pc[use_pc_index];
    
endmodule // Memory



