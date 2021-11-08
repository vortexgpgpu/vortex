`include "VX_cache_define.vh"

module VX_shared_mem #(
    parameter CACHE_ID                      = 0,

    // Size of cache in bytes
    parameter CACHE_SIZE                    = (1024*16), 
    // Number of banks
    parameter NUM_BANKS                     = 2, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 
    // Number of Word requests per cycle
    parameter NUM_REQS                      = 4, 

    // Core Request Queue Size
    parameter CREQ_SIZE                     = 2,
    // Core Response Queue Size
    parameter CRSQ_SIZE                     = 2,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 8,

    // core request tag size
    parameter CORE_TAG_WIDTH                = (2 + CORE_TAG_ID_BITS),

    // bank offset from beginning of index range
    parameter BANK_ADDR_OFFSET              = `CLOG2(256)
 ) (    
    input wire clk,
    input wire reset,   

    // PERF
`ifdef PERF_ENABLE
    VX_perf_cache_if.master perf_cache_if,
`endif

    // Core request    
    input wire [NUM_REQS-1:0]                   core_req_valid,
    input wire [NUM_REQS-1:0]                   core_req_rw,
    input wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0] core_req_addr,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]    core_req_byteen,
    input wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]  core_req_data,
    input wire [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0] core_req_tag,
    output wire [NUM_REQS-1:0]                  core_req_ready,

    // Core response
    output wire                                 core_rsp_valid,
    output wire [NUM_REQS-1:0]                  core_rsp_tmask,
    output wire [NUM_REQS-1:0][`WORD_WIDTH-1:0] core_rsp_data,
    output wire [CORE_TAG_WIDTH-1:0]            core_rsp_tag,
    input  wire                                 core_rsp_ready
);

    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid value"))
    `UNUSED_PARAM (CACHE_ID)
    `UNUSED_PARAM (CORE_TAG_ID_BITS)

    localparam CACHE_LINE_SIZE = WORD_SIZE;

    wire [NUM_BANKS-1:0]                    per_bank_core_req_valid_unqual; 
    wire [NUM_BANKS-1:0]                    per_bank_core_req_rw_unqual;  
    wire [NUM_BANKS-1:0][`LINE_ADDR_WIDTH-1:0] per_bank_core_req_addr_unqual;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_core_req_byteen_unqual;
    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]   per_bank_core_req_data_unqual;
    wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0] per_bank_core_req_tag_unqual;
    wire [NUM_BANKS-1:0][`REQS_BITS-1:0]    per_bank_core_req_tid_unqual;
    wire [NUM_BANKS-1:0]                    per_bank_core_req_ready_unqual;
    
    VX_core_req_bank_sel #(
        .CACHE_ID        (CACHE_ID),
        .CACHE_LINE_SIZE (WORD_SIZE),
        .NUM_BANKS       (NUM_BANKS),
        .NUM_PORTS       (1),
        .WORD_SIZE       (WORD_SIZE),
        .NUM_REQS        (NUM_REQS),
        .CORE_TAG_WIDTH  (CORE_TAG_WIDTH),
        .BANK_ADDR_OFFSET(BANK_ADDR_OFFSET)
    ) core_req_bank_sel (        
        .clk        (clk),
        .reset      (reset),
    `ifdef PERF_ENABLE        
        .bank_stalls(perf_cache_if.bank_stalls),
    `endif     
        .core_req_valid          (core_req_valid),
        .core_req_rw             (core_req_rw),
        .core_req_addr           (core_req_addr),
        .core_req_byteen         (core_req_byteen),
        .core_req_data           (core_req_data),
        .core_req_tag            (core_req_tag),
        .core_req_ready          (core_req_ready),
        .per_bank_core_req_valid (per_bank_core_req_valid_unqual),
        .per_bank_core_req_tid   (per_bank_core_req_tid_unqual),
        .per_bank_core_req_rw    (per_bank_core_req_rw_unqual),
        .per_bank_core_req_addr  (per_bank_core_req_addr_unqual),
        .per_bank_core_req_byteen(per_bank_core_req_byteen_unqual),
        .per_bank_core_req_tag   (per_bank_core_req_tag_unqual),
        .per_bank_core_req_data  (per_bank_core_req_data_unqual),
        .per_bank_core_req_ready (per_bank_core_req_ready_unqual),
        `UNUSED_PIN (per_bank_core_req_pmask),
        `UNUSED_PIN (per_bank_core_req_wsel)
    );

    wire [NUM_BANKS-1:0]                    per_bank_core_req_valid; 
    wire [NUM_BANKS-1:0]                    per_bank_core_req_rw;      
    wire [NUM_BANKS-1:0][`LINE_ADDR_WIDTH-1:0] per_bank_core_req_addr;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_core_req_byteen;
    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]   per_bank_core_req_data;
    wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0] per_bank_core_req_tag;
    wire [NUM_BANKS-1:0][`REQS_BITS-1:0]    per_bank_core_req_tid;

    wire creq_out_valid, creq_out_ready;
    wire creq_in_valid, creq_in_ready;

    wire creq_in_fire = creq_in_valid && creq_in_ready;
    `UNUSED_VAR (creq_in_fire)

    wire creq_out_fire = creq_out_valid && creq_out_ready;
    `UNUSED_VAR (creq_out_fire)

    assign creq_in_valid = (| core_req_valid);
    assign per_bank_core_req_ready_unqual = {NUM_BANKS{creq_in_ready}};

    wire [NUM_BANKS-1:0] core_req_read_mask, core_req_read_mask_unqual;
    wire core_req_writeonly, core_req_writeonly_unqual;

    assign core_req_read_mask_unqual = per_bank_core_req_valid_unqual & ~per_bank_core_req_rw_unqual;
    assign core_req_writeonly_unqual = ~(| core_req_read_mask_unqual);

    VX_elastic_buffer #(
        .DATAW   (NUM_BANKS * (1 + 1 + `LINE_ADDR_WIDTH + WORD_SIZE + `WORD_WIDTH + CORE_TAG_WIDTH + `REQS_BITS) + NUM_BANKS + 1), 
        .SIZE    (CREQ_SIZE),
        .OUT_REG (1)   // output should be registered for the data_store addr port
    ) core_req_queue (
        .clk        (clk),
        .reset      (reset),
        .ready_in   (creq_in_ready),
        .valid_in   (creq_in_valid),
        .data_in    ({per_bank_core_req_valid_unqual,
                      per_bank_core_req_rw_unqual, 
                      per_bank_core_req_addr_unqual, 
                      per_bank_core_req_byteen_unqual, 
                      per_bank_core_req_data_unqual, 
                      per_bank_core_req_tag_unqual,
                      per_bank_core_req_tid_unqual,
                      core_req_read_mask_unqual,
                      core_req_writeonly_unqual}),
        .data_out   ({per_bank_core_req_valid,
                      per_bank_core_req_rw, 
                      per_bank_core_req_addr, 
                      per_bank_core_req_byteen, 
                      per_bank_core_req_data, 
                      per_bank_core_req_tag,
                      per_bank_core_req_tid,
                      core_req_read_mask,
                      core_req_writeonly}),
        .ready_out  (creq_out_ready),
        .valid_out  (creq_out_valid)
    );

    wire crsq_in_valid, crsq_in_ready;
    wire crsq_last_read;
    
    assign creq_out_ready = core_req_writeonly 
                         || (crsq_in_ready && crsq_last_read);

    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0] per_bank_core_rsp_data; 

    for (genvar i = 0; i < NUM_BANKS; i++) begin

        wire [WORD_SIZE-1:0] wren = per_bank_core_req_byteen[i]
                                  & {WORD_SIZE{per_bank_core_req_valid[i] 
                                            && per_bank_core_req_rw[i]}};

        wire [`LINE_SELECT_BITS-1:0] addr = per_bank_core_req_addr[i][`LINE_SELECT_BITS-1:0];

        VX_sp_ram #(
            .DATAW      (`WORD_WIDTH),
            .SIZE       (`LINES_PER_BANK),
            .BYTEENW    (WORD_SIZE),
            .NO_RWCHECK (1)
        ) data_store (
            .clk   (clk),
            .addr  (addr),
            .wren  (wren),
            .wdata (per_bank_core_req_data[i]),
            .rdata (per_bank_core_rsp_data[i])
        );
    end  

    // The core response bus handles a single tag at the time
    // We first need to select the current tag to process,
    // then send all bank responses for that tag as a batch

    reg [NUM_REQS-1:0] core_rsp_valids_in;
    reg [NUM_REQS-1:0][`WORD_WIDTH-1:0] core_rsp_data_in;
    wire [CORE_TAG_WIDTH-1:0] core_rsp_tag_in;
    reg [NUM_BANKS-1:0] bank_rsp_sel_r, bank_rsp_sel_n;

    wire crsq_in_fire = crsq_in_valid && crsq_in_ready;

    assign crsq_last_read = (bank_rsp_sel_n == core_req_read_mask);

    always @(posedge clk) begin
        if (reset) begin
            bank_rsp_sel_r <= 0;
        end else begin
            if (crsq_in_fire) begin
                if (crsq_last_read) begin
                    bank_rsp_sel_r <= 0;
                end else begin
                    bank_rsp_sel_r <= bank_rsp_sel_n;
                end
            end
        end
    end   

    VX_find_first #(
        .N     (NUM_BANKS),
        .DATAW (CORE_TAG_WIDTH)
    ) find_first (
        .valid_i (core_req_read_mask & ~bank_rsp_sel_r),
        .data_i  (per_bank_core_req_tag),
        .data_o  (core_rsp_tag_in),
        `UNUSED_PIN (valid_o)
    );

    always @(*) begin
        core_rsp_valids_in = 0;
        core_rsp_data_in   = 'x;
        bank_rsp_sel_n     = bank_rsp_sel_r;
        for (integer i = 0; i < NUM_BANKS; i++) begin 
            if (core_req_read_mask[i] 
             && (core_rsp_tag_in[CORE_TAG_ID_BITS-1:0] == per_bank_core_req_tag[i][CORE_TAG_ID_BITS-1:0])) begin
                core_rsp_valids_in[per_bank_core_req_tid[i]] = 1;
                core_rsp_data_in[per_bank_core_req_tid[i]] = per_bank_core_rsp_data[i];
                bank_rsp_sel_n[i] = 1;
            end
        end
    end

    assign crsq_in_valid = creq_out_valid && ~core_req_writeonly;

    VX_elastic_buffer #(
        .DATAW (NUM_BANKS * (1 + `WORD_WIDTH) + CORE_TAG_WIDTH),
        .SIZE  (CRSQ_SIZE)
    ) core_rsp_req (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (crsq_in_valid),        
        .data_in   ({core_rsp_valids_in, core_rsp_data_in, core_rsp_tag_in}),
        .ready_in  (crsq_in_ready),      
        .valid_out (core_rsp_valid),
        .data_out  ({core_rsp_tmask, core_rsp_data, core_rsp_tag}),
        .ready_out (core_rsp_ready)
    );

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_UNUSED_BEGIN
    wire [NUM_BANKS-1:0][31:0]         debug_pc_st0, debug_pc_st1;
    wire [NUM_BANKS-1:0][`NW_BITS-1:0] debug_wid_st0, debug_wid_st1;
`IGNORE_UNUSED_END

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin        
            assign {debug_wid_st0[i], debug_pc_st0[i]} = per_bank_core_req_tag_unqual[i][`CACHE_REQ_INFO_RNG];
            assign {debug_wid_st1[i], debug_pc_st1[i]} = per_bank_core_req_tag[i][`CACHE_REQ_INFO_RNG];        
        end else begin
            assign {debug_wid_st0[i], debug_pc_st0[i]} = 0;
            assign {debug_wid_st1[i], debug_pc_st1[i]} = 0;
        end
    end
`endif

`ifdef DBG_TRACE_CACHE_BANK
    
    reg is_multi_tag_req;
`IGNORE_UNUSED_BEGIN
    reg [CORE_TAG_WIDTH-1:0] core_req_tag_sel;
`IGNORE_UNUSED_END

    VX_find_first #(
        .N     (NUM_BANKS),
        .DATAW (CORE_TAG_WIDTH)
    ) find_first_d (
        .valid_i (per_bank_core_req_valid),
        .data_i  (per_bank_core_req_tag),
        .data_o  (core_req_tag_sel),
        `UNUSED_PIN (valid_o)
    );

    always @(*) begin
        is_multi_tag_req = 0;
        for (integer i = 0; i < NUM_BANKS; ++i) begin
            if (per_bank_core_req_valid[i] 
             && (core_req_tag_sel[CORE_TAG_ID_BITS-1:0] != per_bank_core_req_tag[i][CORE_TAG_ID_BITS-1:0])) begin
                is_multi_tag_req = creq_out_valid;
            end
        end
    end

    always @(posedge clk) begin        
        if (!crsq_in_ready) begin
            dpi_trace("%d: *** cache%0d pipeline-stall\n", $time, CACHE_ID);        
        end
        if (is_multi_tag_req) begin
            dpi_trace("%d: *** cache%0d multi-tag request!\n", $time, CACHE_ID);
        end
        if (creq_in_fire) begin
            for (integer i = 0; i < NUM_BANKS; ++i) begin
                if (per_bank_core_req_valid_unqual[i]) begin
                    if (per_bank_core_req_rw_unqual[i]) begin
                        dpi_trace("%d: cache%0d:%0d core-wr-req: addr=%0h, tag=%0h, byteen=%b, data=%0h, wid=%0d, PC=%0h\n", 
                            $time, CACHE_ID, i, `LINE_TO_BYTE_ADDR(per_bank_core_req_addr_unqual[i], i), per_bank_core_req_tag_unqual[i], per_bank_core_req_byteen_unqual[i], per_bank_core_req_data_unqual[i], debug_wid_st0[i], debug_pc_st0[i]);
                    end else begin
                        dpi_trace("%d: cache%0d:%0d core-rd-req: addr=%0h, tag=%0h, byteen=%b, wid=%0d, PC=%0h\n", 
                            $time, CACHE_ID, i, `LINE_TO_BYTE_ADDR(per_bank_core_req_addr_unqual[i], i), per_bank_core_req_tag_unqual[i], per_bank_core_req_byteen_unqual[i], debug_wid_st0[i], debug_pc_st0[i]);
                    end
                end
            end
        end
        if (creq_out_fire) begin
            for (integer i = 0; i < NUM_BANKS; ++i) begin
                if (per_bank_core_req_valid[i]) begin
                    if (per_bank_core_req_rw[i]) begin
                        dpi_trace("%d: cache%0d:%0d core-wr-rsp: addr=%0h, tag=%0h, byteen=%b, data=%0h, wid=%0d, PC=%0h\n", 
                            $time, CACHE_ID, i, `LINE_TO_BYTE_ADDR(per_bank_core_req_addr[i], i), per_bank_core_req_tag[i], per_bank_core_req_byteen[i], per_bank_core_req_data[i], debug_wid_st1[i], debug_pc_st1[i]);
                    end else begin
                        dpi_trace("%d: cache%0d:%0d core-rd-rsp: addr=%0h, tag=%0h, byteen=%b, data=%0h, wid=%0d, PC=%0h\n", 
                            $time, CACHE_ID, i, `LINE_TO_BYTE_ADDR(per_bank_core_req_addr[i], i), per_bank_core_req_tag[i], per_bank_core_req_byteen[i], per_bank_core_rsp_data[i], debug_wid_st1[i], debug_pc_st1[i]);
                    end
                end
            end
        end
    end    
`endif

`ifdef PERF_ENABLE
    // per cycle: core_reads, core_writes
    wire [$clog2(NUM_REQS+1)-1:0] perf_core_reads_per_cycle;
    wire [$clog2(NUM_REQS+1)-1:0] perf_core_writes_per_cycle;
    wire [$clog2(NUM_REQS+1)-1:0] perf_crsp_stall_per_cycle;

    wire [NUM_REQS-1:0] perf_core_reads_per_mask = core_req_valid & core_req_ready & ~core_req_rw;
    wire [NUM_REQS-1:0] perf_core_writes_per_mask = core_req_valid & core_req_ready & core_req_rw;

    `POP_COUNT(perf_core_reads_per_cycle, perf_core_reads_per_mask);
    `POP_COUNT(perf_core_writes_per_cycle, perf_core_writes_per_mask);
    
    if (CORE_TAG_ID_BITS != 0) begin
        wire [NUM_REQS-1:0] perf_crsp_stall_per_mask = core_rsp_tmask & {NUM_REQS{core_rsp_valid && ~core_rsp_ready}};
        `POP_COUNT(perf_crsp_stall_per_cycle, perf_crsp_stall_per_mask);
    end else begin
        wire [NUM_REQS-1:0] perf_crsp_stall_per_mask = core_rsp_valid & ~core_rsp_ready;
        `POP_COUNT(perf_crsp_stall_per_cycle, perf_crsp_stall_per_mask);
    end

    reg [`PERF_CTR_BITS-1:0] perf_core_reads;
    reg [`PERF_CTR_BITS-1:0] perf_core_writes;
    reg [`PERF_CTR_BITS-1:0] perf_crsp_stalls;

    always @(posedge clk) begin
        if (reset) begin
            perf_core_reads   <= 0;
            perf_core_writes  <= 0;
            perf_crsp_stalls  <= 0;
        end else begin
            perf_core_reads   <= perf_core_reads  + `PERF_CTR_BITS'(perf_core_reads_per_cycle);
            perf_core_writes  <= perf_core_writes + `PERF_CTR_BITS'(perf_core_writes_per_cycle);
            perf_crsp_stalls  <= perf_crsp_stalls + `PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
        end
    end

    assign perf_cache_if.reads        = perf_core_reads;
    assign perf_cache_if.writes       = perf_core_writes;
    assign perf_cache_if.read_misses  = '0;
    assign perf_cache_if.write_misses = '0;
    assign perf_cache_if.pipe_stalls  = '0;
    assign perf_cache_if.crsp_stalls  = perf_crsp_stalls;
`endif

endmodule