`include "VX_define.vh"
`include "VX_cache_config.vh"

module Vortex_Cluster 
    #(
        parameter CLUSTER_ID = 0
    ) ( 

    // Clock
    input  wire             clk,
    input  wire             reset,

    // IO
    output wire[`NUM_CORES_PER_CLUSTER-1:0]       io_valid,
    output wire[`NUM_CORES_PER_CLUSTER-1:0][31:0] io_data,

    // DRAM Req
    output wire             out_dram_req,
    output wire             out_dram_req_write,
    output wire             out_dram_req_read,
    output wire [31:0]      out_dram_req_addr,
    output wire [31:0]      out_dram_req_size,
    output wire [31:0]      out_dram_req_data[`DBANK_LINE_WORDS-1:0],
    output wire [31:0]      out_dram_expected_lat,
    input  wire             out_dram_req_delay,

    // DRAM Res
    output wire             out_dram_fill_accept,
    input  wire             out_dram_fill_rsp,
    input  wire [31:0]      out_dram_fill_rsp_addr,
    input  wire [31:0]      out_dram_fill_rsp_data[`DBANK_LINE_WORDS-1:0],

    // LLC Snooping
    input  wire             llc_snp_req,
    input  wire[31:0]       llc_snp_req_addr,
    output wire             llc_snp_req_delay,

    output wire             out_ebreak
);
    // DRAM Dcache Req
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_dram_req;
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_dram_req_write;
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_dram_req_read;
    wire[`NUM_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_dram_req_addr;
    wire[`NUM_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_dram_req_size;
    wire[`NUM_CORES_PER_CLUSTER-1:0][`DBANK_LINE_WORDS-1:0][31:0]  per_core_dram_req_data;
    wire[`NUM_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_dram_expected_lat;

    // DRAM Dcache Res
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_dram_fill_accept;
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_dram_fill_rsp;
    wire[`NUM_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_dram_fill_rsp_addr;
    wire[`NUM_CORES_PER_CLUSTER-1:0][`DBANK_LINE_WORDS-1:0][31:0]  per_core_dram_fill_rsp_data;

    // DRAM Icache Req
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_I_dram_req;
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_I_dram_req_write;
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_I_dram_req_read;
    wire[`NUM_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_I_dram_req_addr;
    wire[`NUM_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_I_dram_req_size;
    wire[`NUM_CORES_PER_CLUSTER-1:0][`IBANK_LINE_WORDS-1:0][31:0]  per_core_I_dram_req_data;
    wire[`NUM_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_I_dram_expected_lat;

    // DRAM Icache Res
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_I_dram_fill_accept;
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_I_dram_fill_rsp;
    wire[`NUM_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_I_dram_fill_rsp_addr;
    wire[`NUM_CORES_PER_CLUSTER-1:0][`IBANK_LINE_WORDS-1:0][31:0]  per_core_I_dram_fill_rsp_data;

    // Out ebreak
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_out_ebreak;

    wire[`NUM_CORES_PER_CLUSTER-1:0]                              per_core_io_valid;
    wire[`NUM_CORES_PER_CLUSTER-1:0][31:0]                        per_core_io_data;

    wire                                                             l2c_core_accept;

    wire                                                             snp_fwd;
    wire[31:0]                                                       snp_fwd_addr;
    wire[`NUM_CORES_PER_CLUSTER-1:0]                              snp_fwd_delay;

    assign out_ebreak = (&per_core_out_ebreak);

    genvar curr_core;
    generate

        for (curr_core = 0; curr_core < `NUM_CORES_PER_CLUSTER; curr_core=curr_core+1) begin

            wire [`IBANK_LINE_WORDS-1:0][31:0] curr_core_I_dram_req_data;
            wire [`DBANK_LINE_WORDS-1:0][31:0] curr_core_dram_req_data ;

            assign io_valid[curr_core]  = per_core_io_valid[curr_core];
            assign io_data [curr_core]  = per_core_io_data [curr_core];

            Vortex #(
                .CORE_ID(curr_core + (CLUSTER_ID * `NUM_CORES_PER_CLUSTER))
            ) vortex_core(
                .clk                        (clk),
                .reset                      (reset),
                .io_valid                   (per_core_io_valid            [curr_core]),
                .io_data                    (per_core_io_data             [curr_core]),
                .dram_req                   (per_core_dram_req            [curr_core]),
                .dram_req_write             (per_core_dram_req_write      [curr_core]),
                .dram_req_read              (per_core_dram_req_read       [curr_core]),
                .dram_req_addr              (per_core_dram_req_addr       [curr_core]),
                .dram_req_size              (per_core_dram_req_size       [curr_core]),
                .dram_req_data              (curr_core_dram_req_data                 ),
                .dram_expected_lat          (per_core_dram_expected_lat   [curr_core]),
                .dram_fill_accept           (per_core_dram_fill_accept    [curr_core]),
                .dram_fill_rsp              (per_core_dram_fill_rsp       [curr_core]),
                .dram_fill_rsp_addr         (per_core_dram_fill_rsp_addr  [curr_core]),
                .dram_fill_rsp_data         (per_core_dram_fill_rsp_data  [curr_core]),
                .I_dram_req                 (per_core_I_dram_req          [curr_core]),
                .I_dram_req_write           (per_core_I_dram_req_write    [curr_core]),
                .I_dram_req_read            (per_core_I_dram_req_read     [curr_core]),
                .I_dram_req_addr            (per_core_I_dram_req_addr     [curr_core]),
                .I_dram_req_size            (per_core_I_dram_req_size     [curr_core]),
                .I_dram_req_data            (curr_core_I_dram_req_data               ),
                .I_dram_expected_lat        (per_core_I_dram_expected_lat [curr_core]),
                .I_dram_fill_accept         (per_core_I_dram_fill_accept  [curr_core]),
                .I_dram_fill_rsp            (per_core_I_dram_fill_rsp     [curr_core]),
                .I_dram_fill_rsp_addr       (per_core_I_dram_fill_rsp_addr[curr_core]),
                .I_dram_fill_rsp_data       (per_core_I_dram_fill_rsp_data[curr_core]),
                .dram_req_delay             (l2c_core_accept                         ),
                .out_ebreak                 (per_core_out_ebreak          [curr_core]),
                .snp_req                    (snp_fwd),
                .snp_req_addr               (snp_fwd_addr),
                .snp_req_delay              (snp_fwd_delay[curr_core]),
                .I_snp_req                  (0),
                .I_snp_req_addr             (),
                .I_snp_req_delay            ()
            );

            assign per_core_dram_req_data  [curr_core] = curr_core_dram_req_data;
            assign per_core_I_dram_req_data[curr_core] = curr_core_I_dram_req_data;
        end
    endgenerate

    //////////////////// L2 Cache ////////////////////
    wire[`L2NUMBER_REQUESTS-1:0]                             l2c_core_req;
    wire[`L2NUMBER_REQUESTS-1:0][2:0]                        l2c_core_req_mem_write;
    wire[`L2NUMBER_REQUESTS-1:0][2:0]                        l2c_core_req_mem_read;
    wire[`L2NUMBER_REQUESTS-1:0][31:0]                       l2c_core_req_addr;
    wire[`L2NUMBER_REQUESTS-1:0][`IBANK_LINE_WORDS-1:0][31:0] l2c_core_req_data;
    wire[`L2NUMBER_REQUESTS-1:0][1:0]                        l2c_core_req_wb;

    wire[`L2NUMBER_REQUESTS-1:0]                             l2c_core_no_wb_slot;

    wire[`L2NUMBER_REQUESTS-1:0]                                  l2c_wb;
    wire[`L2NUMBER_REQUESTS-1:0] [31:0]                           l2c_wb_addr;
    wire[`L2NUMBER_REQUESTS-1:0][`IBANK_LINE_WORDS-1:0][31:0]      l2c_wb_data;

    wire[`DBANK_LINE_WORDS-1:0][31:0]                             dram_req_data_port;
    wire[`DBANK_LINE_WORDS-1:0][31:0]                             dram_fill_rsp_data_port;

    genvar llb_index;
    generate
        for (llb_index = 0; llb_index < `DBANK_LINE_WORDS; llb_index=llb_index+1) begin
            assign out_dram_req_data          [llb_index] = dram_req_data_port[llb_index];
            assign dram_fill_rsp_data_port[llb_index] = out_dram_fill_rsp_data[llb_index];
        end
    endgenerate

    genvar l2c_curr_core;
    generate
        for (l2c_curr_core = 0; l2c_curr_core < `L2NUMBER_REQUESTS; l2c_curr_core=l2c_curr_core+2) begin
            // Core Request
            assign l2c_core_req           [l2c_curr_core]   = per_core_dram_req  [(l2c_curr_core/2)];
            assign l2c_core_req           [l2c_curr_core+1] = per_core_I_dram_req[(l2c_curr_core/2)];

            assign l2c_core_req_mem_write [l2c_curr_core]   = per_core_dram_req_write[(l2c_curr_core/2)] ? `SW_MEM_WRITE : `NO_MEM_WRITE;
            assign l2c_core_req_mem_write [l2c_curr_core+1] = `NO_MEM_WRITE; // I caches don't write

            assign l2c_core_req_mem_read  [l2c_curr_core]   = per_core_dram_req_read[(l2c_curr_core/2)] ? `LW_MEM_READ : `NO_MEM_READ;
            assign l2c_core_req_mem_read  [l2c_curr_core+1] = `LW_MEM_READ; // I caches don't write

            assign l2c_core_req_wb        [l2c_curr_core]   = per_core_dram_req_read[(l2c_curr_core/2)] ? 1 : 0;
            assign l2c_core_req_wb        [l2c_curr_core+1] = 1; // I caches don't write

            assign l2c_core_req_addr      [l2c_curr_core]   = per_core_dram_req_addr  [(l2c_curr_core/2)];
            assign l2c_core_req_addr      [l2c_curr_core+1] = per_core_I_dram_req_addr[(l2c_curr_core/2)];

            assign l2c_core_req_data      [l2c_curr_core]   = per_core_dram_req_data  [(l2c_curr_core/2)];
            assign l2c_core_req_data      [l2c_curr_core+1] = per_core_I_dram_req_data[(l2c_curr_core/2)];

            // Core can't accept Response
            assign l2c_core_no_wb_slot    [l2c_curr_core]   = ~per_core_dram_fill_accept  [(l2c_curr_core/2)];
            assign l2c_core_no_wb_slot    [l2c_curr_core+1] = ~per_core_I_dram_fill_accept[(l2c_curr_core/2)];       

            // Cache Fill Response
            assign per_core_dram_fill_rsp     [(l2c_curr_core/2)] = l2c_wb[l2c_curr_core];
            assign per_core_I_dram_fill_rsp   [(l2c_curr_core/2)] = l2c_wb[l2c_curr_core+1];

            assign per_core_dram_fill_rsp_data[(l2c_curr_core/2)]   = l2c_wb_data[l2c_curr_core];
            assign per_core_I_dram_fill_rsp_data[(l2c_curr_core/2)] = l2c_wb_data[l2c_curr_core+1];

            assign per_core_dram_fill_rsp_addr[(l2c_curr_core/2)]   = l2c_wb_addr[l2c_curr_core];
            assign per_core_I_dram_fill_rsp_addr[(l2c_curr_core/2)] = l2c_wb_addr[l2c_curr_core+1];
        end
    endgenerate

    wire dram_snp_full;
    wire dram_req_because_of_wb;
    VX_cache #(
        .CACHE_SIZE_BYTES             (`L2CACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (`L2BANK_LINE_SIZE_BYTES),
        .NUMBER_BANKS                 (`L2NUMBER_BANKS),
        .WORD_SIZE_BYTES              (`L2WORD_SIZE_BYTES),
        .NUMBER_REQUESTS              (`L2NUMBER_REQUESTS),
        .STAGE_1_CYCLES               (`L2STAGE_1_CYCLES),
        .FUNC_ID                      (`L2FUNC_ID),
        .REQQ_SIZE                    (`L2REQQ_SIZE),
        .MRVQ_SIZE                    (`L2MRVQ_SIZE),
        .DFPQ_SIZE                    (`L2DFPQ_SIZE),
        .SNRQ_SIZE                    (`L2SNRQ_SIZE),
        .CWBQ_SIZE                    (`L2CWBQ_SIZE),
        .DWBQ_SIZE                    (`L2DWBQ_SIZE),
        .DFQQ_SIZE                    (`L2DFQQ_SIZE),
        .LLVQ_SIZE                    (`L2LLVQ_SIZE),
        .FFSQ_SIZE                    (`L2FFSQ_SIZE),
        .PRFQ_SIZE                    (`L2PRFQ_SIZE),
        .PRFQ_STRIDE                  (`L2PRFQ_STRIDE),
        .FILL_INVALIDAOR_SIZE         (`L2FILL_INVALIDAOR_SIZE),
        .SIMULATED_DRAM_LATENCY_CYCLES(`L2SIMULATED_DRAM_LATENCY_CYCLES)
    ) gpu_l2cache (
        .clk               (clk),
        .reset             (reset),

        // Core Req (DRAM Fills/WB) To L2 Request
        .core_req_valid    (l2c_core_req),
        .core_req_addr     (l2c_core_req_addr),
        .core_req_writedata({l2c_core_req_data}),
        .core_req_mem_read (l2c_core_req_mem_read),
        .core_req_mem_write(l2c_core_req_mem_write),
        .core_req_rd       (0),
        .core_req_wb       (l2c_core_req_wb),
        .core_req_warp_num (0),
        .core_req_pc       (0),

        // L2 can't accept Core Request
        .delay_req         (l2c_core_accept),

        // Core can't accept L2 Request
        .core_no_wb_slot   (|l2c_core_no_wb_slot),

        // Core Writeback
        .core_wb_valid     (l2c_wb),
        .core_wb_req_rd    (),
        .core_wb_req_wb    (),
        .core_wb_warp_num  (),
        .core_wb_readdata  ({l2c_wb_data}),
        .core_wb_address   (l2c_wb_addr),
        .core_wb_pc        (),

        // L2 Cache DRAM Fill response
        .dram_fill_rsp     (out_dram_fill_rsp),
        .dram_fill_rsp_addr(out_dram_fill_rsp_addr),
        .dram_fill_rsp_data({dram_fill_rsp_data_port}),

        // L2 Cache can't accept Fill Response
        .dram_fill_accept  (out_dram_fill_accept),

        // L2 Cache DRAM Fill Request
        .dram_req          (out_dram_req),
        .dram_req_write    (out_dram_req_write),
        .dram_req_read     (out_dram_req_read),
        .dram_req_addr     (out_dram_req_addr),
        .dram_req_size     (out_dram_req_size),
        .dram_req_data     ({dram_req_data_port}),
        .dram_req_delay    (out_dram_req_delay),

        // Snoop Response
        .dram_req_because_of_wb(dram_req_because_of_wb),
        .dram_snp_full         (dram_snp_full),

        // Snoop Request
        .snp_req               (llc_snp_req),
        .snp_req_addr          (llc_snp_req_addr),
        .snp_req_delay         (llc_snp_req_delay),

        .snp_fwd               (snp_fwd),
        .snp_fwd_addr          (snp_fwd_addr),
        .snp_fwd_delay         (|snp_fwd_delay)
    );

endmodule