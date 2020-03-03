`include "VX_cache_config.v"


module VX_cache (
	input wire clk,
	input wire reset,

    // Req Info
    input wire [`NUMBER_REQUESTS-1:0]        core_req_valid,
    input wire [`NUMBER_REQUESTS-1:0][31:0]  core_req_addr,
    input wire [`NUMBER_REQUESTS-1:0][31:0]  core_req_writedata,
    input wire[2:0]                          core_req_mem_read,
    input wire[2:0]                          core_req_mem_write,

    // Req meta
    input wire [4:0]                         core_req_rd,
    input wire [1:0]                         core_req_wb,
    input wire [`NW_M1:0]                    core_req_warp_num,

    // Core Writeback
    input  wire                              core_no_wb_slot,
    output wire [`NUMBER_REQUESTS-1:0]       core_wb_valid,
    output wire [4:0]                        core_wb_req_rd,
    output wire [1:0]                        core_wb_req_wb,
    output wire [`NW_M1:0]                   core_wb_warp_num,
    output wire [`NUMBER_REQUESTS-1:0][31:0] core_wb_readdata,


    // Dram Fill Response
    input  wire                              dram_fill_rsp,
    input  wire [31:0]                       dram_fill_rsp_addr,
    input  wire [`BANK_LINE_SIZE_RNG][31:0]  dram_fill_rsp_data,

    // Dram request
    output wire                              dram_req,
    output wire                              dram_req_write,
    output wire                              dram_req_read,
    output wire [31:0]                       dram_req_addr,
    output wire [31:0]                       dram_req_size,
    output wire [`BANK_LINE_SIZE_RNG][31:0]  dram_req_data
);


    wire [`NUMBER_BANKS-1:0][`NUMBER_REQUESTS-1:0]     per_bank_valids;
    wire [`NUMBER_BANKS-1:0]                           per_bank_wb_pop;
    wire [`NUMBER_BANKS-1:0][`NUMBER_REQUESTS-1:0]     per_bank_wb_tid;
    wire [`NUMBER_BANKS-1:0][4:0]                      per_bank_wb_rd;
    wire [`NUMBER_BANKS-1:0][1:0]                      per_bank_wb_wb;
    wire [`NUMBER_BANKS-1:0][`NW_M1:0]                 per_bank_wb_warp_num;
    wire [`NUMBER_BANKS-1:0][31:0]                     per_bank_wb_data;


    wire                                               dfqq_full;
    wire[`NUMBER_BANKS-1:0]                            per_bank_dram_fill_req;
    wire[`NUMBER_BANKS-1:0][31:0]                      per_bank_dram_fill_req_addr;

    wire[`NUMBER_BANKS-1:0]                            per_bank_dram_wb_queue_pop;
    wire[`NUMBER_BANKS-1:0]                            per_bank_dram_wb_req;
    wire[`NUMBER_BANKS-1:0][31:0]                      per_bank_dram_wb_req_addr;
    wire[`NUMBER_BANKS-1]:0[`BANK_LINE_SIZE_RNG][31:0] per_bank_dram_wb_req_data;

    VX_cache_dram_req_arb VX_cache_dram_req_arb(
        .clk                        (clk),
        .reset                      (reset),
        .dfqq_full                  (dfqq_full),
        .per_bank_dram_fill_req     (per_bank_dram_fill_req),
        .per_bank_dram_fill_req_addr(per_bank_dram_fill_req_addr),
        .per_bank_dram_wb_queue_pop (per_bank_dram_wb_queue_pop),
        .per_bank_dram_wb_req       (per_bank_dram_wb_req),
        .per_bank_dram_wb_req_addr  (per_bank_dram_wb_req_addr),
        .per_bank_dram_wb_req_data  (per_bank_dram_wb_req_data),
        .dram_req                   (dram_req),
        .dram_req_write             (dram_req_write),
        .dram_req_read              (dram_req_read),
        .dram_req_addr              (dram_req_addr),
        .dram_req_size              (dram_req_size),
        .dram_req_data              (dram_req_data)
        );


    VX_cache_core_req_bank_sel VX_cache_core_req_bank_sel(
        .core_req_valid (core_req_valid),
        .core_req_addr  (core_req_addr),
        .per_bank_valids(per_bank_valids)
        );


    VX_cache_wb_sel_merge VX_cache_core_req_bank_sel(
        .per_bank_wb_tid     (per_bank_wb_tid),
        .per_bank_wb_rd      (per_bank_wb_rd),
        .per_bank_wb_wb      (per_bank_wb_wb),
        .per_bank_wb_warp_num(per_bank_wb_warp_num),
        .per_bank_wb_data    (per_bank_wb_data),
        .per_bank_wb_pop     (per_bank_wb_pop),

        .core_wb_valid       (core_wb_valid),
        .core_wb_req_rd      (core_wb_req_rd),
        .core_wb_req_wb      (core_wb_req_wb),
        .core_wb_warp_num    (core_wb_warp_num),
        .core_wb_readdata    (core_wb_readdata)
        );

    generate
        integer curr_bank;
        for (curr_bank = 0; curr_bank < `NUMBER_BANKS; curr_bank=curr_bank+1) begin
            wire [`NUMBER_REQUESTS-1:0]        curr_bank_valids;
            wire [`NUMBER_REQUESTS-1:0][31:0]  curr_bank_addr;
            wire [`NUMBER_REQUESTS-1:0][31:0]  curr_bank_writedata;
            wire [4:0]                         curr_bank_rd;
            wire [1:0]                         curr_bank_wb;
            wire [`NW_M1:0]                    curr_bank_warp_num;
            wire [2:0]                         curr_bank_mem_read;  
            wire [2:0]                         curr_bank_mem_write;

            wire                               curr_bank_wb_pop;
            wire [`NUMBER_REQUESTS-1:0]        curr_bank_wb_tid;
            wire [4:0]                         curr_bank_wb_rd;
            wire [1:0]                         curr_bank_wb_wb;
            wire [`NW_M1:0]                    curr_bank_wb_warp_num;
            wire [31:0]                        curr_bank_wb_data;

            wire                               curr_bank_dram_fill_rsp;
            wire [31:0]                        curr_bank_dram_fill_rsp_addr;
            wire [`BANK_LINE_SIZE_RNG][31:0]   curr_bank_dram_fill_rsp_data;

            wire                               curr_bank_dfqq_full;
            wire                               curr_bank_dram_fill_req;
            wire[31:0]                         curr_bank_dram_fill_req_addr;

            wire                               curr_bank_dram_wb_queue_pop;
            wire                               curr_bank_dram_wb_req;
            wire[31:0]                         curr_bank_dram_wb_req_addr;
            wire[`BANK_LINE_SIZE_RNG][31:0]    curr_bank_dram_wb_req_data;

            // Core Req
            assign curr_bank_valids     = per_bank_valids[curr_bank];
            assign curr_bank_addr       = core_req_addr;
            assign curr_bank_writedata  = core_req_writedata;
            assign curr_bank_rd         = core_req_rd;
            assign curr_bank_wb         = core_req_wb;
            assign curr_bank_warp_num   = core_req_warp_num;
            assign curr_bank_mem_read   = core_req_mem_read;
            assign curr_bank_mem_write  = core_req_mem_write;

            // Core WB
            assign curr_bank_wb_pop                = per_bank_wb_pop[curr_bank];
            assign per_bank_wb_tid     [curr_bank] = curr_bank_wb_tid;
            assign per_bank_wb_rd      [curr_bank] = curr_bank_wb_rd;
            assign per_bank_wb_wb      [curr_bank] = curr_bank_wb_wb;
            assign per_bank_wb_warp_num[curr_bank] = curr_bank_wb_warp_num;
            assign per_bank_wb_data    [curr_bank] = curr_bank_wb_data;

            // Dram fill response
            assign curr_bank_dram_fill_rsp      = (`NUMBER_BANKS == 1) || (dram_fill_addr[`BANK_SELECT_ADDR_RNG] == curr_bank);
            assign curr_bank_dram_fill_rsp_addr = dram_fill_rsp_addr;
            assign curr_bank_dram_fill_rsp_data = dram_fill_rsp_data;

            // Dram fill request
            assign curr_bank_dfqq_full                    = dfqq_full;
            assign per_bank_dram_fill_req[curr_bank]      = curr_bank_dram_fill_req;
            assign per_bank_dram_fill_req_addr[curr_bank] = curr_bank_dram_fill_req_addr;

            // Dram writeback request
            assign curr_bank_dram_wb_queue_pop          = per_bank_dram_wb_queue_pop[curr_bank];
            assign per_bank_dram_wb_req[curr_bank]      = curr_bank_dram_wb_req;
            assign per_bank_dram_wb_req_addr[curr_bank] = curr_bank_dram_wb_req_addr;
            assign per_bank_dram_wb_req_data[curr_bank] = curr_bank_dram_wb_req_data;


            VX_cache_bank bank (

                );

        end
    endgenerate



endmodule