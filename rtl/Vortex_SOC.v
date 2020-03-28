`include "VX_define.v"
`include "VX_cache_config.v"

module Vortex_SOC (

    // Clock
    input  wire             clk,
    input  wire             reset,

    // IO
    output wire             io_valid[`NUMBER_CORES-1:0],
    output wire[31:0]       io_data [`NUMBER_CORES-1:0],

    output wire[31:0]       number_cores,

    // DRAM Req
    output wire             out_dram_req,
    output wire             out_dram_req_write,
    output wire             out_dram_req_read,
    output wire [31:0]      out_dram_req_addr,
    output wire [31:0]      out_dram_req_size,
    output wire [31:0]      out_dram_req_data[`DBANK_LINE_SIZE_RNG],
    output wire [31:0]      out_dram_expected_lat,

    // DRAM Res
    output wire             out_dram_fill_accept,
    input  wire             out_dram_fill_rsp,
    input  wire [31:0]      out_dram_fill_rsp_addr,
    input  wire [31:0]      out_dram_fill_rsp_data[`DBANK_LINE_SIZE_RNG],

    // LLC Snooping
    input  wire             llc_snp_req,
    input  wire             llc_snp_req_addr,
    output wire             llc_snp_req_delay,

    output wire             out_ebreak
);

`ifdef L3C

    // DRAM Dcache Req
    wire [`NUMBER_CLUSTERS-1:0]                             dram_req;
    wire [`NUMBER_CLUSTERS-1:0]                             dram_req_write;
    wire [`NUMBER_CLUSTERS-1:0]                             dram_req_read;
    wire [`NUMBER_CLUSTERS-1:0][31:0]                       dram_req_addr;
    wire [`NUMBER_CLUSTERS-1:0][31:0]                       dram_req_size;
    wire [`NUMBER_CLUSTERS-1:0][`DBANK_LINE_SIZE_RNG][31:0] dram_req_data;
    wire [`NUMBER_CLUSTERS-1:0][31:0]                       dram_expected_lat;

    // DRAM Dcache Res
    wire [`NUMBER_CLUSTERS-1:0]                             dram_fill_accept;
    wire [`NUMBER_CLUSTERS-1:0]                             dram_fill_rsp;
    wire [`NUMBER_CLUSTERS-1:0][31:0]                       dram_fill_rsp_addr;
    wire [`NUMBER_CLUSTERS-1:0][`DBANK_LINE_SIZE_RNG][31:0] dram_fill_rsp_data;

    assign number_cores = `NUMBER_CORES;

     // Out ebreak
    wire[`NUMBER_CORES-1:0]                              per_core_out_ebreak;
    assign out_ebreak = (&per_core_out_ebreak);

    wire[`L3NUMBER_REQUESTS-1:0]                             l3c_core_req;
    wire[`L3NUMBER_REQUESTS-1:0][2:0]                        l3c_core_req_mem_write;
    wire[`L3NUMBER_REQUESTS-1:0][2:0]                        l3c_core_req_mem_read;
    wire[`L3NUMBER_REQUESTS-1:0][31:0]                       l3c_core_req_addr;
    wire[`L3NUMBER_REQUESTS-1:0][`IBANK_LINE_SIZE_RNG][31:0] l3c_core_req_data;
    wire[`L3NUMBER_REQUESTS-1:0][1:0]                        l3c_core_req_wb;

    wire                                                     l3c_core_accept;

    wire                                                     l3c_snp_fwd;
    wire[31:0]                                               l3c_snp_fwd_addr;
    wire[`L3NUMBER_REQUESTS-1:0]                             l3c_snp_fwd_delay_temp;
    wire                                                     l3c_snp_fwd_delay;

    assign l3c_snp_fwd_delay = (|l3c_snp_fwd_delay_temp);


    wire[`L3NUMBER_REQUESTS-1:0]                             l3c_wb;
    wire[`L3NUMBER_REQUESTS-1:0] [31:0]                      l3c_wb_addr;
    wire[`L3NUMBER_REQUESTS-1:0][`IBANK_LINE_SIZE_RNG][31:0] l3c_wb_data;

    wire[`IBANK_LINE_SIZE_RNG][31:0] l3c_dram_req_data;
    wire[`IBANK_LINE_SIZE_RNG][31:0] l3c_dram_fill_rsp_data;

    genvar curr_l;
    generate
        for (curr_l = 0; curr_l < `IBANK_LINE_SIZE_WORDS; curr_l=curr_l+1) begin
            assign out_dram_req_data[curr_l][31:0]       = l3c_dram_req_data[curr_l][31:0];
            assign l3c_dram_fill_rsp_data[curr_l][31:0]  = out_dram_fill_rsp_data[curr_l][31:0];
        end
    endgenerate

    // 
    genvar l3c_curr_core;
    generate
        for (l3c_curr_core = 0; l3c_curr_core < `L3NUMBER_REQUESTS; l3c_curr_core=l3c_curr_core+1) begin
            // Core Request
            assign l3c_core_req           [l3c_curr_core]   = dram_req  [(l3c_curr_core)];

            assign l3c_core_req_mem_write [l3c_curr_core]   = dram_req_write ? `SW_MEM_WRITE : `NO_MEM_WRITE;

            assign l3c_core_req_mem_read  [l3c_curr_core]   = dram_req_read ? `LW_MEM_READ : `NO_MEM_READ;

            assign l3c_core_req_wb        [l3c_curr_core]   = dram_req_read ? 1 : 0;

            assign l3c_core_req_addr      [l3c_curr_core]   = dram_req_addr  [(l3c_curr_core)];

            assign l3c_core_req_data      [l3c_curr_core]   = dram_req_data  [(l3c_curr_core)];

            // L2 can't accept requests
            assign dram_fill_accept  [(l3c_curr_core)]   = l3c_core_accept;

            // Cache Fill Response
            assign dram_fill_rsp     [(l3c_curr_core)]   = l3c_wb[l3c_curr_core];

            assign dram_fill_rsp_data[(l3c_curr_core)]   = l3c_wb_data[l3c_curr_core];

            assign dram_fill_rsp_addr[(l3c_curr_core)]   = l3c_wb_addr[l3c_curr_core];
        end
    endgenerate

    wire dram_snp_full;
    wire dram_req_because_of_wb;


    VX_cache #(
        .CACHE_SIZE_BYTES             (`L3CACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (`L3BANK_LINE_SIZE_BYTES),
        .NUMBER_BANKS                 (`L3NUMBER_BANKS),
        .WORD_SIZE_BYTES              (`L3WORD_SIZE_BYTES),
        .NUMBER_REQUESTS              (`L3NUMBER_REQUESTS),
        .STAGE_1_CYCLES               (`L3STAGE_1_CYCLES),
        .FUNC_ID                      (`LLFUNC_ID),
        .REQQ_SIZE                    (`L3REQQ_SIZE),
        .MRVQ_SIZE                    (`L3MRVQ_SIZE),
        .DFPQ_SIZE                    (`L3DFPQ_SIZE),
        .SNRQ_SIZE                    (`L3SNRQ_SIZE),
        .CWBQ_SIZE                    (`L3CWBQ_SIZE),
        .DWBQ_SIZE                    (`L3DWBQ_SIZE),
        .DFQQ_SIZE                    (`L3DFQQ_SIZE),
        .LLVQ_SIZE                    (`L3LLVQ_SIZE),
        .FFSQ_SIZE                    (`L3FFSQ_SIZE),
        .FILL_INVALIDAOR_SIZE         (`L3FILL_INVALIDAOR_SIZE),
        .SIMULATED_DRAM_LATENCY_CYCLES(`L3SIMULATED_DRAM_LATENCY_CYCLES)
        )
        gpu_l3cache
        (
        .clk               (clk),
        .reset             (reset),

        // Core Req (DRAM Fills/WB) To L2 Request
        .core_req_valid    (l3c_core_req),
        .core_req_addr     (l3c_core_req_addr),
        .core_req_writedata({l3c_core_req_data}),
        .core_req_mem_read (l3c_core_req_mem_read),
        .core_req_mem_write(l3c_core_req_mem_write),
        .core_req_rd       (0),
        .core_req_wb       (l3c_core_req_wb),
        .core_req_warp_num (0),
        .core_req_pc       (0),

        // L2 can't accept Core Request
        .delay_req         (l3c_core_accept),

        // Core can't accept L2 Request
        .core_no_wb_slot   (0),

        // Core Writeback
        .core_wb_valid     (l3c_wb),
        .core_wb_req_rd    (),
        .core_wb_req_wb    (),
        .core_wb_warp_num  (),
        .core_wb_readdata  ({l3c_wb_data}),
        .core_wb_address   (l3c_wb_addr),
        .core_wb_pc        (),

        // L2 Cache DRAM Fill response
        .dram_fill_rsp     (out_dram_fill_rsp),
        .dram_fill_rsp_addr(out_dram_fill_rsp_addr),
        .dram_fill_rsp_data({l3c_dram_fill_rsp_data}),

        // L2 Cache can't accept Fill Response
        .dram_fill_accept  (out_dram_fill_accept),

        // L2 Cache DRAM Fill Request
        .dram_req          (out_dram_req),
        .dram_req_write    (out_dram_req_write),
        .dram_req_read     (out_dram_req_read),
        .dram_req_addr     (out_dram_req_addr),
        .dram_req_size     (out_dram_req_size),
        .dram_req_data     ({l3c_dram_req_data}),

        // Snoop Response
        .dram_req_because_of_wb(dram_req_because_of_wb),
        .dram_snp_full         (dram_snp_full),

        // Snoop Request
        .snp_req               (llc_snp_req),
        .snp_req_addr          (llc_snp_req_addr),
        .snp_req_delay         (llc_snp_req_delay),

        .snp_fwd               (l3c_snp_fwd),
        .snp_fwd_addr          (l3c_snp_fwd_addr),
        .snp_fwd_delay         (l3c_snp_fwd_delay)
        );



    //////////////////// L3 Cache ////////////////////



    genvar curr_cluster;
    genvar curr_core;
    genvar llb_index;
    genvar l2c_curr_core;

    generate
        for (curr_cluster = 0; curr_cluster < `NUMBER_CLUSTERS; curr_cluster=curr_cluster+1) begin
        ////////////////////// BEGIN CLUSTER /////////////////

        // DRAM Dcache Req
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_dram_req;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_dram_req_write;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_dram_req_read;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_dram_req_addr;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_dram_req_size;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0][`DBANK_LINE_SIZE_RNG][31:0]  per_core_dram_req_data;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_dram_expected_lat;

        // DRAM Dcache Res
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_dram_fill_accept;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_dram_fill_rsp;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_dram_fill_rsp_addr;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0][`DBANK_LINE_SIZE_RNG][31:0]  per_core_dram_fill_rsp_data;


        // DRAM Icache Req
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_I_dram_req;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_I_dram_req_write;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_I_dram_req_read;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_I_dram_req_addr;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_I_dram_req_size;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0][`IBANK_LINE_SIZE_RNG][31:0]  per_core_I_dram_req_data;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_I_dram_expected_lat;

        // DRAM Icache Res
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_I_dram_fill_accept;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_I_dram_fill_rsp;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0] [31:0]                       per_core_I_dram_fill_rsp_addr;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0][`IBANK_LINE_SIZE_RNG][31:0]  per_core_I_dram_fill_rsp_data;

        // Snoop Requests
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_dcache_snp_req;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0][31:0]                        per_core_dcache_snp_req_addr;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_dcache_snp_req_delay;

        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_icache_snp_req;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0][31:0]                        per_core_icache_snp_req_addr;
        wire[`NUMBER_CORES_PER_CLUSTER-1:0]                              per_core_icache_snp_req_delay;

        // generate
            for (curr_core = 0; curr_core < `NUMBER_CORES_PER_CLUSTER; curr_core=curr_core+1) begin

                wire [`IBANK_LINE_SIZE_RNG][31:0] curr_core_I_dram_req_data;
                wire [`DBANK_LINE_SIZE_RNG][31:0] curr_core_dram_req_data ;

                // assign io_valid[curr_core*curr_cluster]  = per_core_io_valid[curr_core];
                // assign io_data [curr_core*curr_cluster]  = per_core_io_data [curr_core];
                Vortex #(.CORE_ID(curr_core+(curr_cluster*`NUMBER_CORES_PER_CLUSTER))) vortex_core(
                        .clk                        (clk),
                        .reset                      (reset),
                        .io_valid                   (io_valid                     [curr_core+(curr_cluster*`NUMBER_CORES_PER_CLUSTER)]),
                        .io_data                    (io_data                      [curr_core+(curr_cluster*`NUMBER_CORES_PER_CLUSTER)]),
                        .out_ebreak                 (per_core_out_ebreak          [curr_core+(curr_cluster*`NUMBER_CORES_PER_CLUSTER)]),
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
                        .snp_req                    (per_core_dcache_snp_req      [curr_core]),
                        .snp_req_addr               (per_core_dcache_snp_req_addr [curr_core]),
                        .snp_req_delay              (per_core_dcache_snp_req_delay[curr_core]),
                        .I_snp_req                  (per_core_icache_snp_req      [curr_core]),
                        .I_snp_req_addr             (per_core_icache_snp_req_addr [curr_core]),
                        .I_snp_req_delay            (per_core_icache_snp_req_delay[curr_core])
                        );

                assign per_core_dram_req_data  [curr_core] = curr_core_dram_req_data;
                assign per_core_I_dram_req_data[curr_core] = curr_core_I_dram_req_data;
            end
        // endgenerate


        //////////////////// L2 Cache ////////////////////
        wire[`LLNUMBER_REQUESTS-1:0]                             l2c_core_req;
        wire[`LLNUMBER_REQUESTS-1:0][2:0]                        l2c_core_req_mem_write;
        wire[`LLNUMBER_REQUESTS-1:0][2:0]                        l2c_core_req_mem_read;
        wire[`LLNUMBER_REQUESTS-1:0][31:0]                       l2c_core_req_addr;
        wire[`LLNUMBER_REQUESTS-1:0][`IBANK_LINE_SIZE_RNG][31:0] l2c_core_req_data;
        wire[`LLNUMBER_REQUESTS-1:0][1:0]                        l2c_core_req_wb;

        wire                                                     l2c_core_accept;

        wire                                                      l2c_snp_fwd;
        wire[31:0]                                                l2c_snp_fwd_addr;
        wire                                                       l2c_snp_fwd_delay;

        assign l2c_snp_fwd_delay = (|per_core_dcache_snp_req_delay) || (|per_core_icache_snp_req_delay);


        wire[`LLNUMBER_REQUESTS-1:0]                                  l2c_wb;
        wire[`LLNUMBER_REQUESTS-1:0] [31:0]                           l2c_wb_addr;
        wire[`LLNUMBER_REQUESTS-1:0][`IBANK_LINE_SIZE_RNG][31:0]      l2c_wb_data;

        // endgenerate


        // generate
            for (l2c_curr_core = 0; l2c_curr_core < `LLNUMBER_REQUESTS; l2c_curr_core=l2c_curr_core+2) begin
                // Core Request
                assign l2c_core_req           [l2c_curr_core]   = per_core_dram_req  [(l2c_curr_core/2)];
                assign l2c_core_req           [l2c_curr_core+1] = per_core_I_dram_req[(l2c_curr_core/2)];

                assign l2c_core_req_mem_write [l2c_curr_core]   = per_core_dram_req_write ? `SW_MEM_WRITE : `NO_MEM_WRITE;
                assign l2c_core_req_mem_write [l2c_curr_core+1] = `NO_MEM_WRITE; // I caches don't write

                assign l2c_core_req_mem_read  [l2c_curr_core]   = per_core_dram_req_read ? `LW_MEM_READ : `NO_MEM_READ;
                assign l2c_core_req_mem_read  [l2c_curr_core+1] = `LW_MEM_READ; // I caches don't write

                assign l2c_core_req_wb        [l2c_curr_core]   = per_core_dram_req_read ? 1 : 0;
                assign l2c_core_req_wb        [l2c_curr_core+1] = 1; // I caches don't write

                assign l2c_core_req_addr      [l2c_curr_core]   = per_core_dram_req_addr  [(l2c_curr_core/2)];
                assign l2c_core_req_addr      [l2c_curr_core+1] = per_core_I_dram_req_addr[(l2c_curr_core/2)];

                assign l2c_core_req_data      [l2c_curr_core]   = per_core_dram_req_data  [(l2c_curr_core/2)];
                assign l2c_core_req_data      [l2c_curr_core+1] = per_core_I_dram_req_data[(l2c_curr_core/2)];

                // L2 can't accept requests
                assign per_core_dram_fill_accept  [(l2c_curr_core/2)] = l2c_core_accept;
                assign per_core_I_dram_fill_accept[(l2c_curr_core/2)] = l2c_core_accept;

                // Cache Fill Response
                assign per_core_dram_fill_rsp     [(l2c_curr_core/2)] = l2c_wb[l2c_curr_core];
                assign per_core_I_dram_fill_rsp   [(l2c_curr_core/2)] = l2c_wb[l2c_curr_core+1];

                assign per_core_dram_fill_rsp_data[(l2c_curr_core/2)]   = l2c_wb_data[l2c_curr_core];
                assign per_core_I_dram_fill_rsp_data[(l2c_curr_core/2)] = l2c_wb_data[l2c_curr_core+1];

                assign per_core_dram_fill_rsp_addr[(l2c_curr_core/2)]   = l2c_wb_addr[l2c_curr_core];
                assign per_core_I_dram_fill_rsp_addr[(l2c_curr_core/2)] = l2c_wb_addr[l2c_curr_core+1];

                assign per_core_dcache_snp_req     [(l2c_curr_core/2)]  = l2c_snp_fwd;
                assign per_core_dcache_snp_req_addr[(l2c_curr_core/2)]  = l2c_snp_fwd_addr;

                assign per_core_icache_snp_req     [(l2c_curr_core/2)]  = l2c_snp_fwd;
                assign per_core_icache_snp_req_addr[(l2c_curr_core/2)]  = l2c_snp_fwd_addr;
            end
        // endgenerate

        wire dram_snp_full;
        wire dram_req_because_of_wb;


        VX_cache #(
            .CACHE_SIZE_BYTES             (`LLCACHE_SIZE_BYTES),
            .BANK_LINE_SIZE_BYTES         (`LLBANK_LINE_SIZE_BYTES),
            .NUMBER_BANKS                 (`LLNUMBER_BANKS),
            .WORD_SIZE_BYTES              (`LLWORD_SIZE_BYTES),
            .NUMBER_REQUESTS              (`LLNUMBER_REQUESTS),
            .STAGE_1_CYCLES               (`LLSTAGE_1_CYCLES),
            .FUNC_ID                      (`LLFUNC_ID),
            .REQQ_SIZE                    (`LLREQQ_SIZE),
            .MRVQ_SIZE                    (`LLMRVQ_SIZE),
            .DFPQ_SIZE                    (`LLDFPQ_SIZE),
            .SNRQ_SIZE                    (`LLSNRQ_SIZE),
            .CWBQ_SIZE                    (`LLCWBQ_SIZE),
            .DWBQ_SIZE                    (`LLDWBQ_SIZE),
            .DFQQ_SIZE                    (`LLDFQQ_SIZE),
            .LLVQ_SIZE                    (`LLLLVQ_SIZE),
            .FFSQ_SIZE                    (`LLFFSQ_SIZE),
            .FILL_INVALIDAOR_SIZE         (`LLFILL_INVALIDAOR_SIZE),
            .SIMULATED_DRAM_LATENCY_CYCLES(`LLSIMULATED_DRAM_LATENCY_CYCLES)
            )
            gpu_l2cache
            (
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
            .core_no_wb_slot   (0),

            // Core Writeback
            .core_wb_valid     (l2c_wb),
            .core_wb_req_rd    (),
            .core_wb_req_wb    (),
            .core_wb_warp_num  (),
            .core_wb_readdata  ({l2c_wb_data}),
            .core_wb_address   (l2c_wb_addr),
            .core_wb_pc        (),

            // L2 Cache DRAM Fill response
            .dram_fill_rsp     (dram_fill_rsp[curr_cluster]),
            .dram_fill_rsp_addr(dram_fill_rsp_addr[curr_cluster]),
            .dram_fill_rsp_data({dram_fill_rsp_data[curr_cluster]}),

            // L2 Cache can't accept Fill Response
            .dram_fill_accept  (dram_fill_accept),

            // L2 Cache DRAM Fill Request
            .dram_req          (dram_req[curr_cluster]),
            .dram_req_write    (dram_req_write[curr_cluster]),
            .dram_req_read     (dram_req_read[curr_cluster]),
            .dram_req_addr     (dram_req_addr[curr_cluster]),
            .dram_req_size     (dram_req_size[curr_cluster]),
            .dram_req_data     ({dram_req_data[curr_cluster]}),

            // Snoop Response
            .dram_req_because_of_wb(dram_req_because_of_wb),
            .dram_snp_full         (dram_snp_full),

            // Snoop Request
            .snp_req               (l3c_snp_fwd),
            .snp_req_addr          (l3c_snp_fwd_addr),
            .snp_req_delay         (l3c_snp_fwd_delay_temp[curr_cluster]),

            .snp_fwd               (l2c_snp_fwd),
            .snp_fwd_addr          (l2c_snp_fwd_addr),
            .snp_fwd_delay         (l2c_snp_fwd_delay)
            );

            // // Snoop Request
            // .snp_req               (VX_gpu_icache_snp_req.snp_req),
            // .snp_req_addr          (VX_gpu_icache_snp_req.snp_req_addr),
            // .snp_req_delay         (VX_gpu_icache_snp_req.snp_delay),



        //////////////////// L2 Cache ////////////////////


        //////////////////// END CLUSTER ///////////////////
        end
    endgenerate

`else 

    assign number_cores = `NUMBER_CORES;

    // IO
    wire                                                 per_core_io_valid[`NUMBER_CORES-1:0];
    wire[31:0]                                           per_core_io_data[`NUMBER_CORES-1:0];

    // DRAM Dcache Req
    wire[`NUMBER_CORES-1:0]                              per_core_dram_req;
    wire[`NUMBER_CORES-1:0]                              per_core_dram_req_write;
    wire[`NUMBER_CORES-1:0]                              per_core_dram_req_read;
    wire[`NUMBER_CORES-1:0] [31:0]                       per_core_dram_req_addr;
    wire[`NUMBER_CORES-1:0] [31:0]                       per_core_dram_req_size;
    wire[`NUMBER_CORES-1:0][`DBANK_LINE_SIZE_RNG][31:0]  per_core_dram_req_data;
    wire[`NUMBER_CORES-1:0] [31:0]                       per_core_dram_expected_lat;

    // DRAM Dcache Res
    wire[`NUMBER_CORES-1:0]                              per_core_dram_fill_accept;
    wire[`NUMBER_CORES-1:0]                              per_core_dram_fill_rsp;
    wire[`NUMBER_CORES-1:0] [31:0]                       per_core_dram_fill_rsp_addr;
    wire[`NUMBER_CORES-1:0][`DBANK_LINE_SIZE_RNG][31:0]  per_core_dram_fill_rsp_data;


    // DRAM Icache Req
    wire[`NUMBER_CORES-1:0]                              per_core_I_dram_req;
    wire[`NUMBER_CORES-1:0]                              per_core_I_dram_req_write;
    wire[`NUMBER_CORES-1:0]                              per_core_I_dram_req_read;
    wire[`NUMBER_CORES-1:0] [31:0]                       per_core_I_dram_req_addr;
    wire[`NUMBER_CORES-1:0] [31:0]                       per_core_I_dram_req_size;
    wire[`NUMBER_CORES-1:0][`IBANK_LINE_SIZE_RNG][31:0]  per_core_I_dram_req_data;
    wire[`NUMBER_CORES-1:0] [31:0]                       per_core_I_dram_expected_lat;

    // DRAM Icache Res
    wire[`NUMBER_CORES-1:0]                              per_core_I_dram_fill_accept;
    wire[`NUMBER_CORES-1:0]                              per_core_I_dram_fill_rsp;
    wire[`NUMBER_CORES-1:0] [31:0]                       per_core_I_dram_fill_rsp_addr;
    wire[`NUMBER_CORES-1:0][`IBANK_LINE_SIZE_RNG][31:0]  per_core_I_dram_fill_rsp_data;

    // Out ebreak
    wire[`NUMBER_CORES-1:0]                              per_core_out_ebreak;

    assign out_ebreak = (&per_core_out_ebreak);

    genvar curr_core;
    generate

        for (curr_core = 0; curr_core < `NUMBER_CORES; curr_core=curr_core+1) begin

            wire [`IBANK_LINE_SIZE_RNG][31:0] curr_core_I_dram_req_data;
            wire [`DBANK_LINE_SIZE_RNG][31:0] curr_core_dram_req_data ;

            assign io_valid[curr_core]  = per_core_io_valid[curr_core];
            assign io_data [curr_core]  = per_core_io_data [curr_core];

            Vortex #(.CORE_ID(curr_core)) vortex_core(
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
                    .out_ebreak                 (per_core_out_ebreak          [curr_core])
                    );

            assign per_core_dram_req_data  [curr_core] = curr_core_dram_req_data;
            assign per_core_I_dram_req_data[curr_core] = curr_core_I_dram_req_data;
        end
    endgenerate


    //////////////////// L2 Cache ////////////////////
    wire[`LLNUMBER_REQUESTS-1:0]                             l2c_core_req;
    wire[`LLNUMBER_REQUESTS-1:0][2:0]                        l2c_core_req_mem_write;
    wire[`LLNUMBER_REQUESTS-1:0][2:0]                        l2c_core_req_mem_read;
    wire[`LLNUMBER_REQUESTS-1:0][31:0]                       l2c_core_req_addr;
    wire[`LLNUMBER_REQUESTS-1:0][`IBANK_LINE_SIZE_RNG][31:0] l2c_core_req_data;
    wire[`LLNUMBER_REQUESTS-1:0][1:0]                        l2c_core_req_wb;

    wire                                                     l2c_core_accept;


    wire[`LLNUMBER_REQUESTS-1:0]                                  l2c_wb;
    wire[`LLNUMBER_REQUESTS-1:0] [31:0]                           l2c_wb_addr;
    wire[`LLNUMBER_REQUESTS-1:0][`IBANK_LINE_SIZE_RNG][31:0]      l2c_wb_data;


    wire[`DBANK_LINE_SIZE_RNG][31:0]                             dram_req_data_port;
    wire[`DBANK_LINE_SIZE_RNG][31:0]                             dram_fill_rsp_data_port;

    genvar llb_index;
    generate
        for (llb_index = 0; llb_index < `DBANK_LINE_SIZE_WORDS; llb_index=llb_index+1) begin
            assign out_dram_req_data          [llb_index] = dram_req_data_port[llb_index];
            assign dram_fill_rsp_data_port[llb_index] = out_dram_fill_rsp_data[llb_index];
        end
    endgenerate

    // genvar l2c_index;
    // genvar l2c_bank_index;
    // generate
    //     for (l2c_index = 0; l2c_index < `LLNUMBER_REQUESTS; l2c_index=l2c_index+1) begin
    //         assign l2c_wb     [l2c_index]                 = l2c_wb_port     [l2c_index];
    //         assign l2c_wb_addr[l2c_index]                 = l2c_wb_addr_port[l2c_index];
    //         for (l2c_bank_index = 0; l2c_bank_index < `LLNUMBER_REQUESTS; l2c_bank_index=l2c_bank_index+1) begin
    //             assign l2c_wb_data[l2c_index][l2c_bank_index] = l2c_wb_data_port[l2c_index][l2c_bank_index];
    //         end
    //     end
    // endgenerate

    // 
    genvar l2c_curr_core;
    generate
        for (l2c_curr_core = 0; l2c_curr_core < `LLNUMBER_REQUESTS; l2c_curr_core=l2c_curr_core+2) begin
            // Core Request
            assign l2c_core_req           [l2c_curr_core]   = per_core_dram_req  [(l2c_curr_core/2)];
            assign l2c_core_req           [l2c_curr_core+1] = per_core_I_dram_req[(l2c_curr_core/2)];

            assign l2c_core_req_mem_write [l2c_curr_core]   = per_core_dram_req_write ? `SW_MEM_WRITE : `NO_MEM_WRITE;
            assign l2c_core_req_mem_write [l2c_curr_core+1] = `NO_MEM_WRITE; // I caches don't write

            assign l2c_core_req_mem_read  [l2c_curr_core]   = per_core_dram_req_read ? `LW_MEM_READ : `NO_MEM_READ;
            assign l2c_core_req_mem_read  [l2c_curr_core+1] = `LW_MEM_READ; // I caches don't write

            assign l2c_core_req_wb        [l2c_curr_core]   = per_core_dram_req_read ? 1 : 0;
            assign l2c_core_req_wb        [l2c_curr_core+1] = 1; // I caches don't write

            assign l2c_core_req_addr      [l2c_curr_core]   = per_core_dram_req_addr  [(l2c_curr_core/2)];
            assign l2c_core_req_addr      [l2c_curr_core+1] = per_core_I_dram_req_addr[(l2c_curr_core/2)];

            assign l2c_core_req_data      [l2c_curr_core]   = per_core_dram_req_data  [(l2c_curr_core/2)];
            assign l2c_core_req_data      [l2c_curr_core+1] = per_core_I_dram_req_data[(l2c_curr_core/2)];

            // L2 can't accept requests
            assign per_core_dram_fill_accept  [(l2c_curr_core/2)] = l2c_core_accept;
            assign per_core_I_dram_fill_accept[(l2c_curr_core/2)] = l2c_core_accept;

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
        .CACHE_SIZE_BYTES             (`LLCACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (`LLBANK_LINE_SIZE_BYTES),
        .NUMBER_BANKS                 (`LLNUMBER_BANKS),
        .WORD_SIZE_BYTES              (`LLWORD_SIZE_BYTES),
        .NUMBER_REQUESTS              (`LLNUMBER_REQUESTS),
        .STAGE_1_CYCLES               (`LLSTAGE_1_CYCLES),
        .FUNC_ID                      (`LLFUNC_ID),
        .REQQ_SIZE                    (`LLREQQ_SIZE),
        .MRVQ_SIZE                    (`LLMRVQ_SIZE),
        .DFPQ_SIZE                    (`LLDFPQ_SIZE),
        .SNRQ_SIZE                    (`LLSNRQ_SIZE),
        .CWBQ_SIZE                    (`LLCWBQ_SIZE),
        .DWBQ_SIZE                    (`LLDWBQ_SIZE),
        .DFQQ_SIZE                    (`LLDFQQ_SIZE),
        .LLVQ_SIZE                    (`LLLLVQ_SIZE),
        .FFSQ_SIZE                    (`LLFFSQ_SIZE),
        .FILL_INVALIDAOR_SIZE         (`LLFILL_INVALIDAOR_SIZE),
        .SIMULATED_DRAM_LATENCY_CYCLES(`LLSIMULATED_DRAM_LATENCY_CYCLES)
        )
        gpu_l2cache
        (
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
        .core_no_wb_slot   (0),

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

        // Snoop Response
        .dram_req_because_of_wb(dram_req_because_of_wb),
        .dram_snp_full         (dram_snp_full),

        // Snoop Request
        .snp_req               (llc_snp_req),
        .snp_req_addr          (llc_snp_req_addr)
        );

`endif
	

endmodule