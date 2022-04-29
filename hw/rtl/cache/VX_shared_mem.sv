`include "VX_define.vh"

module VX_shared_mem #(
    parameter string  IDNAME                = "",

    // Size of cache in bytes
    parameter SIZE                          = (1024*16), 
    
    // Number of Word requests per cycle
    parameter NUM_REQS                      = 4, 
    // Number of banks
    parameter NUM_BANKS                     = 2,

    // Address width
    parameter ADDR_WIDTH                    = 8,
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 

    // Core Request Queue Size
    parameter CREQ_SIZE                     = 2,
    // Core Response Queue Size
    parameter CRSQ_SIZE                     = 2,

    // Request debug identifier
    parameter REQ_UUID_BITS                 = 0,

    // core request tag size
    parameter TAG_WIDTH                     = REQ_UUID_BITS,

    localparam WORD_WIDTH = WORD_SIZE * 8
 ) (    
    input wire clk,
    input wire reset,   

    // PERF
`ifdef PERF_ENABLE
    VX_perf_cache_if.master perf_cache_if,
`endif

    // Core request    
    input wire [NUM_REQS-1:0]                   req_valid,
    input wire [NUM_REQS-1:0]                   req_rw,
    input wire [NUM_REQS-1:0][ADDR_WIDTH-1:0]   req_addr,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]    req_byteen,
    input wire [NUM_REQS-1:0][WORD_WIDTH-1:0]   req_data,
    input wire [NUM_REQS-1:0][TAG_WIDTH-1:0]    req_tag,
    output wire [NUM_REQS-1:0]                  req_ready,

    // Core response
    output wire [NUM_REQS-1:0]                  rsp_valid,
    output wire [NUM_REQS-1:0][WORD_WIDTH-1:0]  rsp_data,
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   rsp_tag,
    input  wire [NUM_REQS-1:0]                  rsp_ready
);

    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid value"))
    `UNUSED_PARAM (IDNAME)

    localparam REQ_SEL_BITS    = `LOG2UP(NUM_REQS);    
    localparam NUM_WORDS       = SIZE / WORD_SIZE;
    localparam WORDS_PER_BANK  = NUM_WORDS / NUM_BANKS;
    localparam BANK_ADDR_WIDTH = ADDR_WIDTH - `CLOG2(NUM_BANKS);

    wire [NUM_BANKS-1:0]                    per_bank_req_valid_unqual; 
    wire [NUM_BANKS-1:0]                    per_bank_req_rw_unqual;  
    wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0] per_bank_req_addr_unqual;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_req_byteen_unqual;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]   per_bank_req_data_unqual;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]     per_bank_req_tag_unqual;
    wire [NUM_BANKS-1:0][REQ_SEL_BITS-1:0] per_bank_req_idx_unqual;
    wire [NUM_BANKS-1:0]                    per_bank_req_ready_unqual;
    
    VX_core_req_bank_sel #(
        .LINE_SIZE  (WORD_SIZE),
        .WORD_SIZE  (WORD_SIZE),
        .ADDR_WIDTH (ADDR_WIDTH),
        .NUM_REQS   (NUM_REQS),
        .NUM_BANKS  (NUM_BANKS),
        .NUM_PORTS  (1),      
        .TAG_WIDTH  (TAG_WIDTH)
    ) core_req_bank_sel (        
        .clk        (clk),
        .reset      (reset),
    `ifdef PERF_ENABLE        
        .bank_stalls(perf_cache_if.bank_stalls),
    `endif     
        .core_req_valid          (req_valid),
        .core_req_rw             (req_rw),
        .core_req_addr           (req_addr),
        .core_req_byteen         (req_byteen),
        .core_req_data           (req_data),
        .core_req_tag            (req_tag),
        .core_req_ready          (req_ready),
        .per_bank_core_req_valid (per_bank_req_valid_unqual),
        .per_bank_core_req_idx   (per_bank_req_idx_unqual),
        .per_bank_core_req_rw    (per_bank_req_rw_unqual),
        .per_bank_core_req_addr  (per_bank_req_addr_unqual),
        .per_bank_core_req_byteen(per_bank_req_byteen_unqual),
        .per_bank_core_req_tag   (per_bank_req_tag_unqual),
        .per_bank_core_req_data  (per_bank_req_data_unqual),
        .per_bank_core_req_ready (per_bank_req_ready_unqual),
        `UNUSED_PIN (per_bank_core_req_pmask),
        `UNUSED_PIN (per_bank_core_req_wsel)
    );

    wire [NUM_BANKS-1:0]                    per_bank_req_valid; 
    wire [NUM_BANKS-1:0]                    per_bank_req_rw;      
    wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0] per_bank_req_addr;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_req_byteen;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]   per_bank_req_data;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]     per_bank_req_tag;
    wire [NUM_BANKS-1:0][REQ_SEL_BITS-1:0] per_bank_req_idx;

    wire creq_out_valid, creq_out_ready;
    wire creq_in_valid, creq_in_ready;

    wire creq_in_fire = creq_in_valid && creq_in_ready;
    `UNUSED_VAR (creq_in_fire)

    wire creq_out_fire = creq_out_valid && creq_out_ready;
    `UNUSED_VAR (creq_out_fire)

    assign creq_in_valid = (| req_valid);
    assign per_bank_req_ready_unqual = {NUM_BANKS{creq_in_ready}};

    VX_elastic_buffer #(
        .DATAW   (NUM_BANKS * (1 + 1 + BANK_ADDR_WIDTH + WORD_SIZE + WORD_WIDTH + TAG_WIDTH + REQ_SEL_BITS)), 
        .SIZE    (CREQ_SIZE),
        .OUT_REG (1)   // output should be registered for the data_store addr port
    ) req_queue (
        .clk        (clk),
        .reset      (reset),
        .ready_in   (creq_in_ready),
        .valid_in   (creq_in_valid),
        .data_in    ({per_bank_req_valid_unqual,
                      per_bank_req_rw_unqual, 
                      per_bank_req_addr_unqual, 
                      per_bank_req_byteen_unqual, 
                      per_bank_req_data_unqual, 
                      per_bank_req_tag_unqual,
                      per_bank_req_idx_unqual}),
        .data_out   ({per_bank_req_valid,
                      per_bank_req_rw, 
                      per_bank_req_addr, 
                      per_bank_req_byteen, 
                      per_bank_req_data, 
                      per_bank_req_tag,
                      per_bank_req_idx}),
        .ready_out  (creq_out_ready),
        .valid_out  (creq_out_valid)
    );        

    wire [NUM_BANKS-1:0]                     per_bank_rsp_valid;
    wire [NUM_BANKS-1:0][0:0]                per_bank_rsp_pmask;
    wire [NUM_BANKS-1:0][0:0][WORD_WIDTH-1:0] per_bank_rsp_data;
    wire [NUM_BANKS-1:0][0:0][REQ_SEL_BITS-1:0] per_bank_rsp_idx; 
    wire [NUM_BANKS-1:0][0:0][TAG_WIDTH-1:0] per_bank_rsp_tag;   
    wire [NUM_BANKS-1:0]                     per_bank_rsp_ready;

    wire crsq_last_read;

    wire [NUM_BANKS-1:0] req_read_mask = per_bank_req_valid & ~per_bank_req_rw;

    wire write_only = ~(| req_read_mask);

    wire bank_rsp_read_ready = | (req_read_mask & per_bank_rsp_ready);
    
    assign creq_out_ready = write_only || (bank_rsp_read_ready && crsq_last_read);

    for (genvar i = 0; i < NUM_BANKS; i++) begin
        wire [WORD_SIZE-1:0] wren = per_bank_req_byteen[i]
                                  & {WORD_SIZE{per_bank_req_valid[i] && per_bank_req_rw[i]}};
        VX_sp_ram #(
            .DATAW      (WORD_WIDTH),
            .SIZE       (WORDS_PER_BANK),
            .BYTEENW    (WORD_SIZE),
            .NO_RWCHECK (1)
        ) data_store (
            .clk   (clk),
            .addr  (per_bank_req_addr[i]),
            .wren  (wren),
            .wdata (per_bank_req_data[i]),
            .rdata (per_bank_rsp_data[i])
        );
    end

    // output response
    // Stall the input queue until all read results are sent

    reg [NUM_BANKS-1:0] bank_rsp_sel_r, bank_rsp_sel_n;

    wire [NUM_BANKS-1:0] per_bank_rsp_fire = per_bank_rsp_valid & per_bank_rsp_ready;

    assign bank_rsp_sel_n = bank_rsp_sel_r | (req_read_mask & per_bank_rsp_ready);

    assign crsq_last_read = (bank_rsp_sel_n == req_read_mask);

    always @(posedge clk) begin
        if (reset) begin
            bank_rsp_sel_r <= 0;
        end else begin
            if (| per_bank_rsp_fire) begin
                if (crsq_last_read) begin
                    bank_rsp_sel_r <= 0;
                end else begin
                    bank_rsp_sel_r <= bank_rsp_sel_n;
                end
            end
        end
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign per_bank_rsp_valid[i] = creq_out_valid & req_read_mask[i];
        assign per_bank_rsp_pmask[i] = 'x;
        assign per_bank_rsp_tag[i]   = per_bank_req_tag[i];
        assign per_bank_rsp_idx[i]   = per_bank_req_idx[i];
    end

    VX_core_rsp_merge #(
        .NUM_REQS  (NUM_REQS),
        .NUM_BANKS (NUM_BANKS),
        .NUM_PORTS (1),
        .WORD_SIZE (WORD_SIZE),        
        .TAG_WIDTH (TAG_WIDTH),
        .OUT_REG   (NUM_BANKS >= 2)
    ) rsp_merge (
        .clk                     (clk),
        .reset                   (reset),                    
        .per_bank_core_rsp_valid (per_bank_rsp_valid),   
        .per_bank_core_rsp_pmask (per_bank_rsp_pmask),   
        .per_bank_core_rsp_data  (per_bank_rsp_data),
        .per_bank_core_rsp_tag   (per_bank_rsp_tag),
        .per_bank_core_rsp_idx   (per_bank_rsp_idx),   
        .per_bank_core_rsp_ready (per_bank_rsp_ready),
        .core_rsp_valid          (rsp_valid),
        .core_rsp_tag            (rsp_tag),
        .core_rsp_data           (rsp_data),  
        .core_rsp_ready          (rsp_ready)
    );

`ifdef DBG_TRACE_CACHE_BANK

    wire [NUM_BANKS-1:0][`UP(REQ_UUID_BITS)-1:0] req_id_st0, req_id_st1;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        if (REQ_UUID_BITS != 0) begin
            assign req_uuid_st0[i] = per_bank_req_tag_unqual[i][TAG_WIDTH-1 -: REQ_UUID_BITS];
            assign req_uuid_st1[i] = per_bank_req_tag[i][TAG_WIDTH-1 -: REQ_UUID_BITS];
        end else begin
            assign req_uuid_st0[i] = 0;
            assign req_uuid_st1[i] = 0;
        end
    end

    always @(posedge clk) begin        
        if (creq_in_fire) begin
            for (integer i = 0; i < NUM_BANKS; ++i) begin
                if (per_bank_req_valid_unqual[i]) begin
                    if (per_bank_req_rw_unqual[i]) begin
                        dpi_trace(1, "%d: %s:%0d core-wr-req: addr=0x%0h, tag=0x%0h, byteen=%b, data=0x%0h (#%0d)\n", 
                            $time, IDNAME, i, per_bank_req_addr_unqual[i], per_bank_req_tag_unqual[i], per_bank_req_byteen_unqual[i], per_bank_req_data_unqual[i], req_uuid_st0[i]);
                    end else begin
                        dpi_trace(1, "%d: %s:%0d core-rd-req: addr=0x%0h, tag=0x%0h (#%0d)\n", 
                            $time, IDNAME, i, per_bank_req_addr_unqual[i], per_bank_req_tag_unqual[i], req_uuid_st0[i]);
                    end
                end
            end
        end
        if (creq_out_fire) begin
            for (integer i = 0; i < NUM_BANKS; ++i) begin
                if (per_bank_req_valid[i]) begin
                    if (per_bank_req_rw[i]) begin
                        dpi_trace(1, "%d: %s:%0d core-wr-rsp: addr=0x%0h, tag=0x%0h, data=0x%0h (#%0d)\n", 
                            $time, IDNAME, i, per_bank_req_addr[i], per_bank_req_tag[i], per_bank_req_data[i], req_uuid_st1[i]);
                    end else begin
                        dpi_trace(1, "%d: %s:%0d core-rd-rsp: addr=0x%0h, tag=0x%0h, data=0x%0h (#%0d)\n", 
                            $time, IDNAME, i, per_bank_req_addr[i], per_bank_req_tag[i], per_bank_rsp_data[i], req_uuid_st1[i]);
                    end
                end
            end
        end
    end    
`endif

`ifdef PERF_ENABLE
    // per cycle: reads, writes
    wire [$clog2(NUM_REQS+1)-1:0] perf_reads_per_cycle;
    wire [$clog2(NUM_REQS+1)-1:0] perf_writes_per_cycle;
    wire [$clog2(NUM_REQS+1)-1:0] perf_crsp_stall_per_cycle;

    wire [NUM_REQS-1:0] perf_reads_per_req = req_valid & req_ready & ~req_rw;
    wire [NUM_REQS-1:0] perf_writes_per_req = req_valid & req_ready & req_rw;
    wire [NUM_REQS-1:0] perf_crsp_stall_per_req = rsp_valid & ~rsp_ready;

    `POP_COUNT(perf_reads_per_cycle, perf_reads_per_req);
    `POP_COUNT(perf_writes_per_cycle, perf_writes_per_req);
    `POP_COUNT(perf_crsp_stall_per_cycle, perf_crsp_stall_per_req);

    reg [`PERF_CTR_BITS-1:0] perf_reads;
    reg [`PERF_CTR_BITS-1:0] perf_writes;
    reg [`PERF_CTR_BITS-1:0] perf_crsp_stalls;

    always @(posedge clk) begin
        if (reset) begin
            perf_reads  <= 0;
            perf_writes <= 0;
            perf_crsp_stalls <= 0;
        end else begin
            perf_reads  <= perf_reads  + `PERF_CTR_BITS'(perf_reads_per_cycle);
            perf_writes <= perf_writes + `PERF_CTR_BITS'(perf_writes_per_cycle);
            perf_crsp_stalls <= perf_crsp_stalls + `PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
        end
    end

    assign perf_cache_if.reads        = perf_reads;
    assign perf_cache_if.writes       = perf_writes;
    assign perf_cache_if.read_misses  = '0;
    assign perf_cache_if.write_misses = '0;
    assign perf_cache_if.mshr_stalls  = '0;
    assign perf_cache_if.mem_stalls   = '0;
    assign perf_cache_if.crsp_stalls  = perf_crsp_stalls;
`endif

endmodule
