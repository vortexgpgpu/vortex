`include "VX_cache_config.v"

module VX_cache
    #(
    // Size of cache in bytes
    parameter CACHE_SIZE_BYTES              = 1024, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE_BYTES          = 16, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUMBER_BANKS                  = 8, 
    // Size of a word in bytes
    parameter WORD_SIZE_BYTES               = 16, 
    // Number of Word requests per cycle {1, 2, 4, 8, ...}
    parameter NUMBER_REQUESTS               = 2, 
    // Number of cycles to complete stage 1 (read from memory)
    parameter STAGE_1_CYCLES                = 2, 
    // Function ID, {Dcache=0, Icache=1, Sharedmemory=2}
    parameter FUNC_ID                       = 3,

// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

    // Core Request Queue Size
    parameter REQQ_SIZE                     = 8, 
    // Miss Reserv Queue Knob
    parameter MRVQ_SIZE                     = 8, 
    // Dram Fill Rsp Queue Size
    parameter DFPQ_SIZE                     = 2, 
    // Snoop Req Queue
    parameter SNRQ_SIZE                     = 8, 

// Queues for writebacks Knobs {1, 2, 4, 8, ...}
    // Core Writeback Queue Size
    parameter CWBQ_SIZE                     = 8, 
    // Dram Writeback Queue Size
    parameter DWBQ_SIZE                     = 4, 
    // Dram Fill Req Queue Size
    parameter DFQQ_SIZE                     = 8, 
    // Lower Level Cache Hit Queue Size
    parameter LLVQ_SIZE                     = 16, 
    // Fill Forward SNP Queue
    parameter FFSQ_SIZE                     = 8,

    // Fill Invalidator Size {Fill invalidator must be active}
    parameter FILL_INVALIDAOR_SIZE          = 16, 

// Dram knobs
    parameter SIMULATED_DRAM_LATENCY_CYCLES = 10


    )
    (
	input wire clk,
	input wire reset,

    // Req Info
    input wire [NUMBER_REQUESTS-1:0]                 core_req_valid,
    input wire [NUMBER_REQUESTS-1:0][31:0]           core_req_addr,
    input wire [NUMBER_REQUESTS-1:0][`WORD_SIZE_RNG] core_req_writedata,
    input wire[NUMBER_REQUESTS-1:0][2:0]             core_req_mem_read,
    input wire[NUMBER_REQUESTS-1:0][2:0]             core_req_mem_write,

    // Req meta
    input wire [4:0]                         core_req_rd,
    input wire [NUMBER_REQUESTS-1:0][1:0]    core_req_wb,
    input wire [`NW_M1:0]                    core_req_warp_num,
    input wire [31:0]                        core_req_pc,
    output wire                              delay_req,

    // Core Writeback
    input  wire                              core_no_wb_slot,
    output wire [NUMBER_REQUESTS-1:0]        core_wb_valid,
    output wire [4:0]                        core_wb_req_rd,
    output wire [1:0]                        core_wb_req_wb,
    output wire [`NW_M1:0]                   core_wb_warp_num,
    output wire [NUMBER_REQUESTS-1:0][`WORD_SIZE_RNG] core_wb_readdata,
    output wire [NUMBER_REQUESTS-1:0][31:0]  core_wb_pc,
    output wire [NUMBER_REQUESTS-1:0][31:0]  core_wb_address,


    // Dram Fill Response
    input  wire                              dram_fill_rsp,
    input  wire [31:0]                       dram_fill_rsp_addr,
    input  wire [`IBANK_LINE_SIZE_RNG][31:0] dram_fill_rsp_data,
    output wire                              dram_fill_accept,

    // Dram request
    output wire                              dram_req,
    output wire                              dram_req_write,
    output wire                              dram_req_read,
    output wire [31:0]                       dram_req_addr,
    output wire [31:0]                       dram_req_size,
    output wire [`IBANK_LINE_SIZE_RNG][31:0] dram_req_data,
    output wire                              dram_req_because_of_wb,
    output wire                              dram_snp_full,


    // Snoop Req
    input wire                               snp_req,
    input wire[31:0]                         snp_req_addr,
    output wire                              snp_req_delay,

    // Snoop Forward
    output wire                              snp_fwd,
    output wire[31:0]                        snp_fwd_addr,
    input  wire                              snp_fwd_delay

);


    wire [NUMBER_BANKS-1:0][NUMBER_REQUESTS-1:0]             per_bank_valids;
    wire [NUMBER_BANKS-1:0]                                  per_bank_wb_pop;
    wire [NUMBER_BANKS-1:0]                                  per_bank_wb_valid;
    wire [NUMBER_BANKS-1:0][`vx_clog2(NUMBER_REQUESTS)-1:0]  per_bank_wb_tid;
    wire [NUMBER_BANKS-1:0][4:0]                             per_bank_wb_rd;
    wire [NUMBER_BANKS-1:0][1:0]                             per_bank_wb_wb;
    wire [NUMBER_BANKS-1:0][`NW_M1:0]                        per_bank_wb_warp_num;
    wire [NUMBER_BANKS-1:0][`WORD_SIZE_RNG]                  per_bank_wb_data;
    wire [NUMBER_BANKS-1:0][31:0]                            per_bank_wb_pc;
    wire [NUMBER_BANKS-1:0][31:0]                            per_bank_wb_address;


    wire                                                     dfqq_full;
    wire[NUMBER_BANKS-1:0]                                   per_bank_dram_fill_req;
    wire[NUMBER_BANKS-1:0][31:0]                             per_bank_dram_fill_req_addr;
    wire[NUMBER_BANKS-1:0]                                   per_bank_dram_fill_accept;

    wire[NUMBER_BANKS-1:0]                                   per_bank_dram_wb_queue_pop;
    wire[NUMBER_BANKS-1:0]                                   per_bank_dram_wb_req;
    wire[NUMBER_BANKS-1:0]                                   per_bank_dram_because_of_snp;
    wire[NUMBER_BANKS-1:0][31:0]                             per_bank_dram_wb_req_addr;
    wire[NUMBER_BANKS-1:0][`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0] per_bank_dram_wb_req_data;

    wire[NUMBER_BANKS-1:0]                                   per_bank_reqq_full;

    wire[NUMBER_BANKS-1:0]                                   per_bank_snrq_full;

    wire[NUMBER_BANKS-1:0]                                   per_bank_snp_fwd;
    wire[NUMBER_BANKS-1:0][31:0]                             per_bank_snp_fwd_addr;
    wire[NUMBER_BANKS-1:0]                                   per_bank_snp_fwd_pop;


    assign delay_req = (|per_bank_reqq_full);


    assign snp_req_delay = (|per_bank_snrq_full);


    assign dram_fill_accept = (NUMBER_BANKS == 1) ? per_bank_dram_fill_accept[0] : per_bank_dram_fill_accept[dram_fill_rsp_addr[`BANK_SELECT_ADDR_RNG]];

    VX_cache_dram_req_arb  #(
        .CACHE_SIZE_BYTES             (CACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (BANK_LINE_SIZE_BYTES),
        .NUMBER_BANKS                 (NUMBER_BANKS),
        .WORD_SIZE_BYTES              (WORD_SIZE_BYTES),
        .NUMBER_REQUESTS              (NUMBER_REQUESTS),
        .STAGE_1_CYCLES               (STAGE_1_CYCLES),
        .REQQ_SIZE                    (REQQ_SIZE),
        .MRVQ_SIZE                    (MRVQ_SIZE),
        .DFPQ_SIZE                    (DFPQ_SIZE),
        .SNRQ_SIZE                    (SNRQ_SIZE),
        .CWBQ_SIZE                    (CWBQ_SIZE),
        .DWBQ_SIZE                    (DWBQ_SIZE),
        .DFQQ_SIZE                    (DFQQ_SIZE),
        .LLVQ_SIZE                    (LLVQ_SIZE),
        .FILL_INVALIDAOR_SIZE         (FILL_INVALIDAOR_SIZE),
        .SIMULATED_DRAM_LATENCY_CYCLES(SIMULATED_DRAM_LATENCY_CYCLES)
        )
        VX_cache_dram_req_arb
        (
        .clk                        (clk),
        .reset                      (reset),
        .dfqq_full                  (dfqq_full),
        .per_bank_dram_fill_req     (per_bank_dram_fill_req),
        .per_bank_dram_fill_req_addr(per_bank_dram_fill_req_addr),
        .per_bank_dram_wb_queue_pop (per_bank_dram_wb_queue_pop),
        .per_bank_dram_wb_req       (per_bank_dram_wb_req),
        .per_bank_dram_because_of_snp(per_bank_dram_because_of_snp),
        .per_bank_dram_wb_req_addr  (per_bank_dram_wb_req_addr),
        .per_bank_dram_wb_req_data  (per_bank_dram_wb_req_data),
        .dram_req                   (dram_req),
        .dram_req_write             (dram_req_write),
        .dram_req_read              (dram_req_read),
        .dram_req_addr              (dram_req_addr),
        .dram_req_size              (dram_req_size),
        .dram_req_data              (dram_req_data),
        .dram_req_because_of_wb     (dram_req_because_of_wb)
        );


    VX_cache_core_req_bank_sel  #(
        .CACHE_SIZE_BYTES             (CACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (BANK_LINE_SIZE_BYTES),
        .NUMBER_BANKS                 (NUMBER_BANKS),
        .WORD_SIZE_BYTES              (WORD_SIZE_BYTES),
        .NUMBER_REQUESTS              (NUMBER_REQUESTS),
        .STAGE_1_CYCLES               (STAGE_1_CYCLES),
        .REQQ_SIZE                    (REQQ_SIZE),
        .MRVQ_SIZE                    (MRVQ_SIZE),
        .DFPQ_SIZE                    (DFPQ_SIZE),
        .SNRQ_SIZE                    (SNRQ_SIZE),
        .CWBQ_SIZE                    (CWBQ_SIZE),
        .DWBQ_SIZE                    (DWBQ_SIZE),
        .DFQQ_SIZE                    (DFQQ_SIZE),
        .LLVQ_SIZE                    (LLVQ_SIZE),
        .FILL_INVALIDAOR_SIZE         (FILL_INVALIDAOR_SIZE),
        .SIMULATED_DRAM_LATENCY_CYCLES(SIMULATED_DRAM_LATENCY_CYCLES)
        )
        VX_cache_core_req_bank_sell
        (
        .core_req_valid (core_req_valid),
        .core_req_addr  (core_req_addr),
        .per_bank_valids(per_bank_valids)
        );


    VX_cache_wb_sel_merge  #(
        .CACHE_SIZE_BYTES             (CACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (BANK_LINE_SIZE_BYTES),
        .NUMBER_BANKS                 (NUMBER_BANKS),
        .WORD_SIZE_BYTES              (WORD_SIZE_BYTES),
        .NUMBER_REQUESTS              (NUMBER_REQUESTS),
        .STAGE_1_CYCLES               (STAGE_1_CYCLES),
        .FUNC_ID                      (FUNC_ID),
        .REQQ_SIZE                    (REQQ_SIZE),
        .MRVQ_SIZE                    (MRVQ_SIZE),
        .DFPQ_SIZE                    (DFPQ_SIZE),
        .SNRQ_SIZE                    (SNRQ_SIZE),
        .CWBQ_SIZE                    (CWBQ_SIZE),
        .DWBQ_SIZE                    (DWBQ_SIZE),
        .DFQQ_SIZE                    (DFQQ_SIZE),
        .LLVQ_SIZE                    (LLVQ_SIZE),
        .FILL_INVALIDAOR_SIZE         (FILL_INVALIDAOR_SIZE),
        .SIMULATED_DRAM_LATENCY_CYCLES(SIMULATED_DRAM_LATENCY_CYCLES)
        ) 
        VX_cache_core_wb_sel_merge
        (
        .per_bank_wb_valid   (per_bank_wb_valid),
        .per_bank_wb_tid     (per_bank_wb_tid),
        .per_bank_wb_rd      (per_bank_wb_rd),
        .per_bank_wb_pc      (per_bank_wb_pc),
        .per_bank_wb_wb      (per_bank_wb_wb),
        .per_bank_wb_warp_num(per_bank_wb_warp_num),
        .per_bank_wb_data    (per_bank_wb_data),
        .per_bank_wb_pop     (per_bank_wb_pop),
        .per_bank_wb_address (per_bank_wb_address),

        .core_no_wb_slot     (core_no_wb_slot),
        .core_wb_valid       (core_wb_valid),
        .core_wb_req_rd      (core_wb_req_rd),
        .core_wb_req_wb      (core_wb_req_wb),
        .core_wb_warp_num    (core_wb_warp_num),
        .core_wb_readdata    (core_wb_readdata),
        .core_wb_address     (core_wb_address),
        .core_wb_pc          (core_wb_pc)
        );




    // Snoop Forward Logic
    VX_snp_fwd_arb #(.NUMBER_BANKS(NUMBER_BANKS)) VX_snp_fwd_arb(
        .per_bank_snp_fwd     (per_bank_snp_fwd),
        .per_bank_snp_fwd_addr(per_bank_snp_fwd_addr),
        .per_bank_snp_fwd_pop (per_bank_snp_fwd_pop),
        .snp_fwd              (snp_fwd),
        .snp_fwd_addr         (snp_fwd_addr),
        .snp_fwd_delay        (snp_fwd_delay)
        );

    // Snoop Forward Logic

    genvar curr_bank;
    generate
        for (curr_bank = 0; curr_bank < NUMBER_BANKS; curr_bank=curr_bank+1) begin
            wire [NUMBER_REQUESTS-1:0]            curr_bank_valids;
            wire [NUMBER_REQUESTS-1:0][31:0]      curr_bank_addr;
            wire [NUMBER_REQUESTS-1:0][`WORD_SIZE_RNG] curr_bank_writedata;
            wire [4:0]                             curr_bank_rd;
            wire [NUMBER_REQUESTS-1:0][1:0]        curr_bank_wb;
            wire [`NW_M1:0]                        curr_bank_warp_num;
            wire [NUMBER_REQUESTS-1:0][2:0]        curr_bank_mem_read;  
            wire [NUMBER_REQUESTS-1:0][2:0]        curr_bank_mem_write;
            wire [31:0]                            curr_bank_pc;

            wire                                   curr_bank_wb_pop;
            wire                                   curr_bank_wb_valid;
            wire [`vx_clog2(NUMBER_REQUESTS)-1:0]  curr_bank_wb_tid;
            wire [31:0]                            curr_bank_wb_pc;
            wire [4:0]                             curr_bank_wb_rd;
            wire [1:0]                             curr_bank_wb_wb;
            wire [`NW_M1:0]                        curr_bank_wb_warp_num;
            wire [`WORD_SIZE_RNG]                  curr_bank_wb_data;
            wire [31:0]                            curr_bank_wb_address;

            wire                                   curr_bank_dram_fill_rsp;
            wire [31:0]                            curr_bank_dram_fill_rsp_addr;
            wire [`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0]       curr_bank_dram_fill_rsp_data;
            wire                                   curr_bank_dram_fill_accept;

            wire                                   curr_bank_dfqq_full;
            wire                                   curr_bank_dram_fill_req;
            wire                                   curr_bank_dram_because_of_snp;
            wire                                   curr_bank_dram_snp_full;
            wire[31:0]                             curr_bank_dram_fill_req_addr;

            wire                                   curr_bank_dram_wb_queue_pop;
            wire                                   curr_bank_dram_wb_req;
            wire[31:0]                             curr_bank_dram_wb_req_addr;
            wire[`BANK_LINE_SIZE_RNG][`WORD_SIZE-1:0] curr_bank_dram_wb_req_data;

            wire                                   curr_bank_snp_req;
            wire[31:0]                             curr_bank_snp_req_addr;

            wire                                   curr_bank_reqq_full;

            wire                                   curr_bank_snp_fwd;
            wire[31:0]                             curr_bank_snp_fwd_addr;
            wire                                   curr_bank_snp_fwd_pop;
            wire                                    curr_bank_snrq_full;

            

            // Core Req
            assign curr_bank_valids              = per_bank_valids[curr_bank];
            assign curr_bank_addr                = core_req_addr;
            assign curr_bank_writedata           = core_req_writedata;
            assign curr_bank_rd                  = core_req_rd;
            assign curr_bank_wb                  = core_req_wb;
            assign curr_bank_pc                  = core_req_pc;
            assign curr_bank_warp_num            = core_req_warp_num;
            assign curr_bank_mem_read            = core_req_mem_read;
            assign curr_bank_mem_write           = core_req_mem_write;
            assign per_bank_reqq_full[curr_bank] = curr_bank_reqq_full;

            // Core WB
            assign curr_bank_wb_pop                = per_bank_wb_pop[curr_bank];
            assign per_bank_wb_valid   [curr_bank] = curr_bank_wb_valid;
            assign per_bank_wb_tid     [curr_bank] = curr_bank_wb_tid;
            assign per_bank_wb_rd      [curr_bank] = curr_bank_wb_rd;
            assign per_bank_wb_wb      [curr_bank] = curr_bank_wb_wb;
            assign per_bank_wb_warp_num[curr_bank] = curr_bank_wb_warp_num;
            assign per_bank_wb_data    [curr_bank] = curr_bank_wb_data;
            assign per_bank_wb_pc      [curr_bank] = curr_bank_wb_pc;
            assign per_bank_wb_address [curr_bank] = curr_bank_wb_address;

            // Dram fill request
            assign curr_bank_dfqq_full                    = dfqq_full;
            assign per_bank_dram_fill_req[curr_bank]      = curr_bank_dram_fill_req;
            assign per_bank_dram_fill_req_addr[curr_bank] = curr_bank_dram_fill_req_addr;

            // Dram fill response
            assign curr_bank_dram_fill_rsp              = (NUMBER_BANKS == 1) || (dram_fill_rsp && (curr_bank_dram_fill_rsp_addr[`BANK_SELECT_ADDR_RNG] == curr_bank));
            assign curr_bank_dram_fill_rsp_addr         = dram_fill_rsp_addr;
            assign curr_bank_dram_fill_rsp_data         = dram_fill_rsp_data;
            assign per_bank_dram_fill_accept[curr_bank] = curr_bank_dram_fill_accept;

            // Dram writeback request
            assign curr_bank_dram_wb_queue_pop             = per_bank_dram_wb_queue_pop[curr_bank];
            assign per_bank_dram_wb_req[curr_bank]         = curr_bank_dram_wb_req;
            assign per_bank_dram_because_of_snp[curr_bank] = curr_bank_dram_because_of_snp;
            assign per_bank_dram_wb_req_addr[curr_bank]    = curr_bank_dram_wb_req_addr;
            assign per_bank_dram_wb_req_data[curr_bank]    = curr_bank_dram_wb_req_data;

            // Snoop Request
            assign curr_bank_snp_req        = snp_req && (snp_req_addr[`BANK_SELECT_ADDR_RNG] == curr_bank);
            assign curr_bank_snp_req_addr   = snp_req_addr;
            assign per_bank_snrq_full[curr_bank] = curr_bank_snrq_full;

            // Snoop Fwd
            assign curr_bank_snp_fwd_pop            = per_bank_snp_fwd_pop[curr_bank];
            assign per_bank_snp_fwd[curr_bank]      = curr_bank_snp_fwd;
            assign per_bank_snp_fwd_addr[curr_bank] = curr_bank_snp_fwd_addr;

            
            VX_bank #(
            .CACHE_SIZE_BYTES             (CACHE_SIZE_BYTES),
            .BANK_LINE_SIZE_BYTES         (BANK_LINE_SIZE_BYTES),
            .NUMBER_BANKS                 (NUMBER_BANKS),
            .WORD_SIZE_BYTES              (WORD_SIZE_BYTES),
            .NUMBER_REQUESTS              (NUMBER_REQUESTS),
            .STAGE_1_CYCLES               (STAGE_1_CYCLES),
            .FUNC_ID                      (FUNC_ID),
            .REQQ_SIZE                    (REQQ_SIZE),
            .MRVQ_SIZE                    (MRVQ_SIZE),
            .DFPQ_SIZE                    (DFPQ_SIZE),
            .SNRQ_SIZE                    (SNRQ_SIZE),
            .CWBQ_SIZE                    (CWBQ_SIZE),
            .DWBQ_SIZE                    (DWBQ_SIZE),
            .DFQQ_SIZE                    (DFQQ_SIZE),
            .LLVQ_SIZE                    (LLVQ_SIZE),
            .FFSQ_SIZE                    (FFSQ_SIZE),
            .FILL_INVALIDAOR_SIZE         (FILL_INVALIDAOR_SIZE),
            .SIMULATED_DRAM_LATENCY_CYCLES(SIMULATED_DRAM_LATENCY_CYCLES)
            )
            bank
            (
                .clk                     (clk),
                .reset                   (reset),
                // Core req
                .delay_req               (delay_req),
                .bank_valids             (curr_bank_valids),
                .bank_addr               (curr_bank_addr),
                .bank_writedata          (curr_bank_writedata),
                .bank_rd                 (curr_bank_rd),
                .bank_wb                 (curr_bank_wb),
                .bank_pc                 (curr_bank_pc),
                .bank_warp_num           (curr_bank_warp_num),
                .bank_mem_read           (curr_bank_mem_read),
                .bank_mem_write          (curr_bank_mem_write),
                .reqq_full               (curr_bank_reqq_full),

                // Output core wb
                .bank_wb_pop             (curr_bank_wb_pop),
                .bank_wb_valid           (curr_bank_wb_valid),
                .bank_wb_tid             (curr_bank_wb_tid),
                .bank_wb_rd              (curr_bank_wb_rd),
                .bank_wb_wb              (curr_bank_wb_wb),
                .bank_wb_warp_num        (curr_bank_wb_warp_num),
                .bank_wb_data            (curr_bank_wb_data),
                .bank_wb_pc              (curr_bank_wb_pc),
                .bank_wb_address         (curr_bank_wb_address),

                // Dram fill req
                .dram_fill_req           (curr_bank_dram_fill_req),
                .dram_fill_req_addr      (curr_bank_dram_fill_req_addr),
                .dram_fill_req_queue_full(curr_bank_dfqq_full),

                // Dram fill rsp
                .dram_fill_rsp           (curr_bank_dram_fill_rsp),
                .dram_fill_addr          (curr_bank_dram_fill_rsp_addr),
                .dram_fill_rsp_data      (curr_bank_dram_fill_rsp_data),
                .dram_fill_accept        (curr_bank_dram_fill_accept),

                // Dram writeback
                .dram_wb_queue_pop       (curr_bank_dram_wb_queue_pop),
                .dram_wb_req             (curr_bank_dram_wb_req),
                .dram_wb_req_addr        (curr_bank_dram_wb_req_addr),
                .dram_wb_req_data        (curr_bank_dram_wb_req_data),
                .dram_because_of_snp     (curr_bank_dram_because_of_snp),
                .dram_snp_full           (curr_bank_dram_snp_full),

                // Snoop Request
                .snp_req                 (curr_bank_snp_req),
                .snp_req_addr            (curr_bank_snp_req_addr),
                .snrq_full               (curr_bank_snrq_full),

                // Snoop Fwd
                .snp_fwd                 (curr_bank_snp_fwd),
                .snp_fwd_addr            (curr_bank_snp_fwd_addr),
                .snp_fwd_pop             (curr_bank_snp_fwd_pop)

                );

        end
    endgenerate



endmodule