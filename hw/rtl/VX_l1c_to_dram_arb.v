`include "VX_define.vh"

module VX_l1c_to_dram_arb #(
    parameter REQQ_SIZE = 8
) (
    input wire              clk,
    input wire              reset,

    VX_cache_dram_req_if    dcache_dram_req_if,
    VX_cache_dram_rsp_if    dcache_dram_rsp_if,

    VX_cache_dram_req_if    icache_dram_req_if,
    VX_cache_dram_rsp_if    icache_dram_rsp_if,

    VX_cache_dram_req_if    dram_req_if,
    VX_cache_dram_rsp_if    dram_rsp_if
);
    reg cache_sel;
    wire icache_req_valid, icache_sel_out, icache_sel_in;

    assign icache_req_valid = icache_dram_req_if.dram_req_read || icache_dram_req_if.dram_req_write;

    assign icache_sel_out = icache_req_valid && (cache_sel == 0);

    assign dram_req_if.dram_req_read  = icache_sel_out ? icache_dram_req_if.dram_req_read  : dcache_dram_req_if.dram_req_read;
    assign dram_req_if.dram_req_write = icache_sel_out ? icache_dram_req_if.dram_req_write : dcache_dram_req_if.dram_req_write;
    assign dram_req_if.dram_req_addr  = icache_sel_out ? icache_dram_req_if.dram_req_addr  : dcache_dram_req_if.dram_req_addr;
    assign dram_req_if.dram_req_data  = icache_sel_out ? icache_dram_req_if.dram_req_data  : dcache_dram_req_if.dram_req_data;
    assign dram_req_if.dram_req_tag   = {icache_sel_out ? icache_dram_req_if.dram_req_tag  : dcache_dram_req_if.dram_req_tag, icache_sel_out};

    assign icache_dram_req_if.dram_req_ready = dram_req_if.dram_req_ready && (cache_sel == 0); 
    assign dcache_dram_req_if.dram_req_ready = dram_req_if.dram_req_ready && (cache_sel == 1);

    assign icache_sel_in = dram_rsp_if.dram_rsp_tag[0];

    assign icache_dram_rsp_if.dram_rsp_valid = dram_rsp_if.dram_rsp_valid && icache_sel_in;
    assign icache_dram_rsp_if.dram_rsp_data  = dram_rsp_if.dram_rsp_data;
    assign icache_dram_rsp_if.dram_rsp_tag   = dram_rsp_if.dram_rsp_tag[1 +: $bits(icache_dram_rsp_if.dram_rsp_tag)];

    assign dcache_dram_rsp_if.dram_rsp_valid = dram_rsp_if.dram_rsp_valid && ~icache_sel_in;
    assign dcache_dram_rsp_if.dram_rsp_data  = dram_rsp_if.dram_rsp_data;
    assign dcache_dram_rsp_if.dram_rsp_tag   = dram_rsp_if.dram_rsp_tag[1 +: $bits(dcache_dram_rsp_if.dram_rsp_tag)];

    assign dram_rsp_if.dram_rsp_ready = icache_dram_rsp_if.dram_rsp_ready && dcache_dram_rsp_if.dram_rsp_ready;

    always @(posedge clk) begin
        if (reset) begin      
            cache_sel <= 0;
        end else begin
            cache_sel <= ~cache_sel;
        end
    end

endmodule