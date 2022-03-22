`include "VX_cache_define.vh"

module VX_shared_mem #(
    parameter IDNAME                        = "",

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

    // core request tag size
    parameter TAG_WIDTH                     = 8, 

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
    input wire [NUM_REQS-1:0][TAG_WIDTH-1:0]    core_req_tag,
    output wire [NUM_REQS-1:0]                  core_req_ready,

    // Core response
    output wire [NUM_REQS-1:0]                  core_rsp_valid,
    output wire [NUM_REQS-1:0][`WORD_WIDTH-1:0] core_rsp_data,
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   core_rsp_tag,
    input  wire [NUM_REQS-1:0]                  core_rsp_ready
);
    `define __WID_ADDR_OFFSET           `CLOG2(`SMEM_LOCAL_SIZE / `SMEM_WORD_SIZE)
    `define SMEM_LINE_TO_BLOCK_ADDR(x)  {x[BANK_ADDR_OFFSET +: `NW_BITS], x[0 +: `__WID_ADDR_OFFSET]}

    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid value"))
    `UNUSED_PARAM (IDNAME)

    localparam CACHE_LINE_SIZE = WORD_SIZE;
    localparam NUM_WAYS = 1;

    wire [NUM_BANKS-1:0]                    per_bank_core_req_valid_unqual; 
    wire [NUM_BANKS-1:0]                    per_bank_core_req_rw_unqual;  
    wire [NUM_BANKS-1:0][`LINE_ADDR_WIDTH-1:0] per_bank_core_req_addr_unqual;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_core_req_byteen_unqual;
    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]   per_bank_core_req_data_unqual;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]     per_bank_core_req_tag_unqual;
    wire [NUM_BANKS-1:0][`REQS_BITS-1:0]    per_bank_core_req_tid_unqual;
    wire [NUM_BANKS-1:0]                    per_bank_core_req_ready_unqual;
    
    VX_core_req_bank_sel #(
        .CACHE_ID        (IDNAME),
        .CACHE_LINE_SIZE (WORD_SIZE),
        .NUM_BANKS       (NUM_BANKS),
        .NUM_PORTS       (1),
        .WORD_SIZE       (WORD_SIZE),
        .NUM_REQS        (NUM_REQS),
        .CORE_TAG_WIDTH  (TAG_WIDTH),
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
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]     per_bank_core_req_tag;
    wire [NUM_BANKS-1:0][`REQS_BITS-1:0]    per_bank_core_req_tid;

    wire creq_out_valid, creq_out_ready;
    wire creq_in_valid, creq_in_ready;

    wire creq_in_fire = creq_in_valid && creq_in_ready;
    `UNUSED_VAR (creq_in_fire)

    wire creq_out_fire = creq_out_valid && creq_out_ready;
    `UNUSED_VAR (creq_out_fire)

    assign creq_in_valid = (| core_req_valid);
    assign per_bank_core_req_ready_unqual = {NUM_BANKS{creq_in_ready}};

    VX_elastic_buffer #(
        .DATAW   (NUM_BANKS * (1 + 1 + `LINE_ADDR_WIDTH + WORD_SIZE + `WORD_WIDTH + TAG_WIDTH + `REQS_BITS)), 
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
                      per_bank_core_req_tid_unqual}),
        .data_out   ({per_bank_core_req_valid,
                      per_bank_core_req_rw, 
                      per_bank_core_req_addr, 
                      per_bank_core_req_byteen, 
                      per_bank_core_req_data, 
                      per_bank_core_req_tag,
                      per_bank_core_req_tid}),
        .ready_out  (creq_out_ready),
        .valid_out  (creq_out_valid)
    );        

    wire [NUM_BANKS-1:0]                     per_bank_core_rsp_valid;
    wire [NUM_BANKS-1:0][0:0]                per_bank_core_rsp_pmask;
    wire [NUM_BANKS-1:0][0:0][`WORD_WIDTH-1:0] per_bank_core_rsp_data;
    wire [NUM_BANKS-1:0][0:0][`REQS_BITS-1:0] per_bank_core_rsp_tid; 
    wire [NUM_BANKS-1:0][0:0][TAG_WIDTH-1:0] per_bank_core_rsp_tag;   
    wire [NUM_BANKS-1:0]                     per_bank_core_rsp_ready;

    wire crsq_last_read;

    wire [NUM_BANKS-1:0] core_req_read_mask = per_bank_core_req_valid & ~per_bank_core_req_rw;
    
    assign creq_out_ready = ~(| core_req_read_mask)
                         || ((| per_bank_core_rsp_ready) && crsq_last_read);

    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0] per_bank_core_rsp_data; 

    wire [NUM_BANKS-1:0][`LINE_SELECT_BITS-1:0] per_bank_blk_addr;

    for (genvar i = 0; i < NUM_BANKS; i++) begin

        assign per_bank_blk_addr[i] = `SMEM_LINE_TO_BLOCK_ADDR(per_bank_core_req_addr[i]);

        wire [WORD_SIZE-1:0] wren = per_bank_core_req_byteen[i]
                                  & {WORD_SIZE{per_bank_core_req_valid[i] 
                                            && per_bank_core_req_rw[i]}};
        VX_sp_ram #(
            .DATAW      (`WORD_WIDTH),
            .SIZE       (`LINES_PER_BANK),
            .BYTEENW    (WORD_SIZE),
            .NO_RWCHECK (1)
        ) data_store (
            .clk   (clk),
            .addr  (per_bank_blk_addr[i]),
            .wren  (wren),
            .wdata (per_bank_core_req_data[i]),
            .rdata (per_bank_core_rsp_data[i])
        );
    end

    // output response
    // Stall the input queue until all read results are sent

    reg [NUM_BANKS-1:0] bank_rsp_sel_r, bank_rsp_sel_n;

    wire [NUM_BANKS-1:0] per_bank_core_rsp_fire = per_bank_core_rsp_valid & per_bank_core_rsp_ready;

    assign bank_rsp_sel_n = bank_rsp_sel_r | (core_req_read_mask & per_bank_core_rsp_ready);

    assign crsq_last_read = (bank_rsp_sel_n == core_req_read_mask);

    always @(posedge clk) begin
        if (reset) begin
            bank_rsp_sel_r <= 0;
        end else begin
            if (| per_bank_core_rsp_fire) begin
                if (crsq_last_read) begin
                    bank_rsp_sel_r <= 0;
                end else begin
                    bank_rsp_sel_r <= bank_rsp_sel_n;
                end
            end
        end
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign per_bank_core_rsp_valid[i] = creq_out_valid & per_bank_core_req_valid[i];
        assign per_bank_core_rsp_pmask[i] = 'x;
        assign per_bank_core_rsp_data[i]  = per_bank_core_req_data[i];
        assign per_bank_core_rsp_tag[i]   = per_bank_core_req_tag[i];
        assign per_bank_core_rsp_tid[i]   = per_bank_core_req_tid[i];
    end

    VX_core_rsp_merge #(
        .CACHE_ID  (IDNAME),
        .NUM_BANKS (NUM_BANKS),
        .NUM_PORTS (1),
        .WORD_SIZE (WORD_SIZE),
        .NUM_REQS  (NUM_REQS),
        .CORE_TAG_WIDTH (TAG_WIDTH)
    ) core_rsp_merge (
        .clk                     (clk),
        .reset                   (reset),                    
        .per_bank_core_rsp_valid (per_bank_core_rsp_valid),   
        .per_bank_core_rsp_pmask (per_bank_core_rsp_pmask),   
        .per_bank_core_rsp_data  (per_bank_core_rsp_data),
        .per_bank_core_rsp_tag   (per_bank_core_rsp_tag),
        .per_bank_core_rsp_tid   (per_bank_core_rsp_tid),   
        .per_bank_core_rsp_ready (per_bank_core_rsp_ready),
        .core_rsp_valid          (core_rsp_valid),
        .core_rsp_tag            (core_rsp_tag),
        .core_rsp_data           (core_rsp_data),  
        .core_rsp_ready          (core_rsp_ready)
    );

`ifdef DBG_TRACE_CACHE_BANK

    wire [NUM_BANKS-1:0][`DBG_CACHE_REQ_IDW-1:0]  req_id_st0, req_id_st1;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign req_id_st0[i] = per_bank_core_req_tag_unqual[i][`CACHE_REQ_ID_RNG];
        assign req_id_st1[i] = per_bank_core_req_tag[i][`CACHE_REQ_ID_RNG];
    end
  
    reg is_multi_tag_req;
`IGNORE_UNUSED_BEGIN
    reg [TAG_WIDTH-1:0] core_req_tag_sel;
`IGNORE_UNUSED_END

    VX_find_first #(
        .N     (NUM_BANKS),
        .DATAW (TAG_WIDTH)
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
        if (|(per_bank_core_rsp_valid & ~per_bank_core_rsp_ready)) begin
            dpi_trace("%d: *** cache%0d pipeline-stall\n", $time, IDNAME);        
        end
        if (is_multi_tag_req) begin
            dpi_trace("%d: *** cache%0d multi-tag request!\n", $time, IDNAME);
        end
        if (creq_in_fire) begin
            for (integer i = 0; i < NUM_BANKS; ++i) begin
                if (per_bank_core_req_valid_unqual[i]) begin
                    if (per_bank_core_req_rw_unqual[i]) begin
                        dpi_trace("%d: %s:%0d core-wr-req: addr=0x%0h, tag=0x%0h, byteen=%b, data=0x%0h (#%0d)\n", 
                            $time, IDNAME, i, `LINE_TO_BYTE_ADDRX(per_bank_core_req_addr_unqual[i], i), per_bank_core_req_tag_unqual[i], per_bank_core_req_byteen_unqual[i], per_bank_core_req_data_unqual[i], req_id_st0[i]);
                    end else begin
                        dpi_trace("%d: %s:%0d core-rd-req: addr=0x%0h, tag=0x%0h (#%0d)\n", 
                            $time, IDNAME, i, `LINE_TO_BYTE_ADDRX(per_bank_core_req_addr_unqual[i], i), per_bank_core_req_tag_unqual[i], req_id_st0[i]);
                    end
                end
            end
        end
        if (creq_out_fire) begin
            for (integer i = 0; i < NUM_BANKS; ++i) begin
                if (per_bank_core_req_valid[i]) begin
                    if (per_bank_core_req_rw[i]) begin
                        dpi_trace("%d: %s:%0d core-wr-rsp: addr=0x%0h, blk_addr=%0d, tag=0x%0h, data=0x%0h (#%0d)\n", 
                            $time, IDNAME, i, `LINE_TO_BYTE_ADDRX(per_bank_core_req_addr[i], i), per_bank_blk_addr[i], per_bank_core_req_tag[i], per_bank_core_req_data[i], req_id_st1[i]);
                    end else begin
                        dpi_trace("%d: %s:%0d core-rd-rsp: addr=0x%0h, blk_addr=%0d, tag=0x%0h, data=0x%0h (#%0d)\n", 
                            $time, IDNAME, i, `LINE_TO_BYTE_ADDRX(per_bank_core_req_addr[i], i), per_bank_blk_addr[i], per_bank_core_req_tag[i], per_bank_core_rsp_data[i], req_id_st1[i]);
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

    wire [NUM_REQS-1:0] perf_core_reads_per_mask = core_req_valid & core_req_ready & ~core_req_rw;
    wire [NUM_REQS-1:0] perf_core_writes_per_mask = core_req_valid & core_req_ready & core_req_rw;

    `POP_COUNT(perf_core_reads_per_cycle, perf_core_reads_per_mask);
    `POP_COUNT(perf_core_writes_per_cycle, perf_core_writes_per_mask);
    wire perf_crsp_stall_per_cycle = core_rsp_valid & ~core_rsp_ready;

    reg [`PERF_CTR_BITS-1:0] perf_core_reads;
    reg [`PERF_CTR_BITS-1:0] perf_core_writes;
    reg [`PERF_CTR_BITS-1:0] perf_crsp_stalls;

    always @(posedge clk) begin
        if (reset) begin
            perf_core_reads  <= 0;
            perf_core_writes <= 0;
            perf_crsp_stalls <= 0;
        end else begin
            perf_core_reads  <= perf_core_reads  + `PERF_CTR_BITS'(perf_core_reads_per_cycle);
            perf_core_writes <= perf_core_writes + `PERF_CTR_BITS'(perf_core_writes_per_cycle);
            perf_crsp_stalls <= perf_crsp_stalls + `PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
        end
    end

    assign perf_cache_if.reads        = perf_core_reads;
    assign perf_cache_if.writes       = perf_core_writes;
    assign perf_cache_if.read_misses  = '0;
    assign perf_cache_if.write_misses = '0;
    assign perf_cache_if.mshr_stalls  = '0;
    assign perf_cache_if.mem_stalls   = '0;
    assign perf_cache_if.crsp_stalls  = perf_crsp_stalls;
`endif

endmodule
