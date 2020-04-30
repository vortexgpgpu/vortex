`include "VX_define.vh"

module VX_lsu_unit (
    input wire              clk,
    input wire              reset,

    input wire              no_slot_mem,
    VX_lsu_req_if           lsu_req_if,

    // Write back to GPR
    VX_wb_if                mem_wb_if,

   // Dcache interface
    VX_cache_core_rsp_if    dcache_rsp_if,
    VX_cache_core_req_if    dcache_req_if,

    output wire             delay
);
    // Generate Addresses
    wire[`NUM_THREADS-1:0][31:0] address;
    VX_lsu_addr_gen VX_lsu_addr_gen (
        .base_address (lsu_req_if.base_address),
        .offset       (lsu_req_if.offset),
        .address      (address)
    );

    wire[`NUM_THREADS-1:0][31:0]     use_address;
    wire[`NUM_THREADS-1:0][31:0]     use_store_data;
    wire[`NUM_THREADS-1:0]           use_valid;
    wire[`WORD_SEL_BITS-1:0]         use_mem_read; 
    wire[`WORD_SEL_BITS-1:0]         use_mem_write;
    wire[4:0]            use_rd;
    wire[`NW_BITS-1:0]   use_warp_num;
    wire[1:0]            use_wb;
    wire[31:0]           use_pc;    
    wire[(`LOG2UP(`NUM_THREADS))-1:0] tag_index;

    wire zero = 0;

    VX_generic_register #(
        .N(45 + `NW_BITS-1 + 1 + `NUM_THREADS*65)
    ) lsu_buffer (
        .clk  (clk),
        .reset(reset),
        .stall(delay),
        .flush(zero),
        .in   ({address    , lsu_req_if.store_data, lsu_req_if.valid, lsu_req_if.mem_read, lsu_req_if.mem_write, lsu_req_if.rd, lsu_req_if.warp_num, lsu_req_if.wb, lsu_req_if.lsu_pc}),
        .out  ({use_address, use_store_data       , use_valid       , use_mem_read       , use_mem_write       , use_rd       , use_warp_num       , use_wb       , use_pc           })
    );

    // Core Request
    assign dcache_req_if.core_req_valid = use_valid;
    assign dcache_req_if.core_req_read  = {`NUM_THREADS{use_mem_read}};
    assign dcache_req_if.core_req_write = {`NUM_THREADS{use_mem_write}};
    assign dcache_req_if.core_req_addr  = use_address;
    assign dcache_req_if.core_req_data  = use_store_data;    
    assign dcache_req_if.core_req_tag   = {`NUM_THREADS{use_pc, use_wb, use_rd, use_warp_num}};
    assign delay = ~dcache_req_if.core_req_ready;

    // Core Response
    assign mem_wb_if.valid = dcache_rsp_if.core_rsp_valid;
    assign mem_wb_if.data  = dcache_rsp_if.core_rsp_data;
    assign dcache_rsp_if.core_rsp_ready = ~no_slot_mem;    
    assign {mem_wb_if.pc, mem_wb_if.wb, mem_wb_if.rd, mem_wb_if.warp_num} = dcache_rsp_if.core_rsp_tag[tag_index];

    // select first valid entry in tag array
    VX_generic_priority_encoder #(
        .N(`NUM_THREADS)
    ) tag_select (
        .valids(dcache_rsp_if.core_rsp_valid),
        .index (tag_index),
    `IGNORE_WARNINGS_BEGIN
        .found ()
    `IGNORE_WARNINGS_END
    );

    /*always_comb begin
        if (1'($time & 1) && dcache_req_if.core_req_ready && |dcache_req_if.core_req_valid) begin
            $display("*** %t: D$ req: valid=%b, addr=%0h, r=%d, w=%d, rd=%d, warp=%d, data=%0h", $time, use_valid, use_address, use_mem_read, use_mem_write, use_rd, use_warp_num, use_store_data);
        end
        if (1'($time & 1) && dcache_rsp_if.core_rsp_ready && |dcache_rsp_if.core_rsp_valid) begin
            $display("*** %t: D$ rsp: valid=%b, rd=%d, warp=%d, data=%0h", $time, dcache_rsp_if.core_rsp_valid, mem_wb_if.rd, mem_wb_if.warp_num, mem_wb_if.data);
        end
    end*/
    
endmodule // Memory



