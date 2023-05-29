`include "VX_define.vh"

module VX_shared_mem #(
    parameter `STRING  INSTANCE_ID = "",

    // Size of cache in bytes
    parameter SIZE              = (1024*16*8), 
    
    // Number of Word requests per cycle
    parameter NUM_REQS          = 4, 
    // Number of banks
    parameter NUM_BANKS         = 4,

    // Address width
    parameter ADDR_WIDTH        = 22,
    // Size of a word in bytes
    parameter WORD_SIZE         = `XLEN/8,

    // Request debug identifier
    parameter UUID_WIDTH        = 0,

    // Request tag size
    parameter TAG_WIDTH         = 16,

    // Response output register
    parameter OUT_REG           = 1
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
    input wire [NUM_REQS-1:0][WORD_SIZE*8-1:0]  req_data,
    input wire [NUM_REQS-1:0][TAG_WIDTH-1:0]    req_tag,
    output wire [NUM_REQS-1:0]                  req_ready,

    // Core response
    output wire [NUM_REQS-1:0]                  rsp_valid,
    output wire [NUM_REQS-1:0][WORD_SIZE*8-1:0] rsp_data,
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   rsp_tag,
    input  wire [NUM_REQS-1:0]                  rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (UUID_WIDTH)

    localparam WORD_WIDTH      = WORD_SIZE * 8;
    localparam REQ_SEL_BITS    = `CLOG2(NUM_REQS);
    localparam NUM_WORDS       = SIZE / WORD_SIZE;
    localparam WORDS_PER_BANK  = NUM_WORDS / NUM_BANKS;
    localparam BANK_ADDR_WIDTH = `CLOG2(WORDS_PER_BANK);

    `STATIC_ASSERT(ADDR_WIDTH == (BANK_ADDR_WIDTH + `CLOG2(NUM_BANKS)), ("invalid parameter"))

    wire [NUM_BANKS-1:0]                    per_bank_req_valid_unqual; 
    wire [NUM_BANKS-1:0]                    per_bank_req_rw_unqual;  
    wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0] per_bank_req_addr_unqual;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_req_byteen_unqual;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]    per_bank_req_data_unqual;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]     per_bank_req_tag_unqual;
    wire [NUM_BANKS-1:0][`UP(REQ_SEL_BITS)-1:0] per_bank_req_idx_unqual;
    wire [NUM_BANKS-1:0]                    per_bank_req_ready_unqual;
    
    VX_cache_req_dispatch #(
        .LINE_SIZE  (WORD_SIZE),
        .WORD_SIZE  (WORD_SIZE),
        .ADDR_WIDTH (ADDR_WIDTH),
        .NUM_REQS   (NUM_REQS),
        .NUM_BANKS  (NUM_BANKS),
        .NUM_PORTS  (1),      
        .TAG_WIDTH  (TAG_WIDTH)
    ) req_dispatch (        
        .clk        (clk),
        .reset      (reset),
    `ifdef PERF_ENABLE        
        .bank_stalls (perf_cache_if.bank_stalls),
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
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]    per_bank_req_data;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]     per_bank_req_tag;
    wire [NUM_BANKS-1:0][`UP(REQ_SEL_BITS)-1:0] per_bank_req_idx;
    wire [NUM_BANKS-1:0]                    per_bank_req_ready;

    `RESET_RELAY_EX (req_sbuf_reset, reset, 1, (NUM_BANKS > 1) ? 0 : -1);

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        VX_skid_buffer #(
            .DATAW   (1 + BANK_ADDR_WIDTH + `UP(REQ_SEL_BITS)),
            .OUT_REG (1) // output should be registered for the data_store addressing
        ) req_sbuf0 (
            .clk       (clk),
            .reset     (req_sbuf_reset),
            .valid_in  (per_bank_req_valid_unqual[i]),
            .ready_in  (per_bank_req_ready_unqual[i]),
            .data_in   ({per_bank_req_rw_unqual[i], per_bank_req_addr_unqual[i], per_bank_req_idx_unqual[i]}),
            .data_out  ({per_bank_req_rw[i],        per_bank_req_addr[i],        per_bank_req_idx[i]}),
            .valid_out (per_bank_req_valid[i]),
            .ready_out (per_bank_req_ready[i])
        );
        
        VX_skid_buffer #(
            .DATAW (WORD_SIZE + WORD_WIDTH + TAG_WIDTH)
        ) req_sbuf1 (
            .clk      (clk),
            .reset    (req_sbuf_reset),
            .valid_in (per_bank_req_valid_unqual[i]),
            `UNUSED_PIN (ready_in),            
            .data_in  ({per_bank_req_byteen_unqual[i], per_bank_req_data_unqual[i], per_bank_req_tag_unqual[i]}),
            .data_out ({per_bank_req_byteen[i],        per_bank_req_data[i],        per_bank_req_tag[i]}),
            `UNUSED_PIN (valid_out),
            .ready_out (per_bank_req_ready[i])
        );
    end 

    wire [NUM_BANKS-1:0]                     per_bank_rsp_valid;
    wire [NUM_BANKS-1:0][0:0]                per_bank_rsp_pmask;
    wire [NUM_BANKS-1:0][0:0][WORD_WIDTH-1:0] per_bank_rsp_data;
    wire [NUM_BANKS-1:0][0:0][`UP(REQ_SEL_BITS)-1:0] per_bank_rsp_idx; 
    wire [NUM_BANKS-1:0][0:0][TAG_WIDTH-1:0] per_bank_rsp_tag;   
    wire [NUM_BANKS-1:0]                     per_bank_rsp_ready;

    // Generate memory banks
    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        VX_sp_ram #(
            .DATAW (WORD_WIDTH),
            .SIZE  (WORDS_PER_BANK),
            .WRENW (WORD_SIZE)
        ) data_store (
            .clk   (clk),
            .write (per_bank_req_valid[i] && per_bank_req_rw[i]),
            .wren  (per_bank_req_byteen[i]),
            .addr  (per_bank_req_addr[i]),            
            .wdata (per_bank_req_data[i]),
            .rdata (per_bank_rsp_data[i])
        );
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign per_bank_rsp_valid[i] = per_bank_req_valid[i] && ~per_bank_req_rw[i];
        assign per_bank_rsp_pmask[i] = '0;
        assign per_bank_rsp_tag[i]   = per_bank_req_tag[i];
        assign per_bank_rsp_idx[i]   = per_bank_req_idx[i];
        assign per_bank_req_ready[i] = per_bank_req_rw[i] || per_bank_rsp_ready[i];
    end

    wire [NUM_REQS-1:0]                 rsp_valid_s;
    wire [NUM_REQS-1:0][WORD_WIDTH-1:0] rsp_data_s;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]  rsp_tag_s;
    wire [NUM_REQS-1:0]                 rsp_ready_s;

    VX_cache_rsp_merge #(
        .NUM_REQS  (NUM_REQS),
        .NUM_BANKS (NUM_BANKS),
        .NUM_PORTS (1),
        .WORD_SIZE (WORD_SIZE),        
        .TAG_WIDTH (TAG_WIDTH)
    ) rsp_merge (
        .clk                     (clk),
        .reset                   (reset),
        .per_bank_core_rsp_valid (per_bank_rsp_valid),   
        .per_bank_core_rsp_pmask (per_bank_rsp_pmask),   
        .per_bank_core_rsp_data  (per_bank_rsp_data),
        .per_bank_core_rsp_tag   (per_bank_rsp_tag),
        .per_bank_core_rsp_idx   (per_bank_rsp_idx),   
        .per_bank_core_rsp_ready (per_bank_rsp_ready),
        .core_rsp_valid          (rsp_valid_s),
        .core_rsp_tag            (rsp_tag_s),
        .core_rsp_data           (rsp_data_s),  
        .core_rsp_ready          (rsp_ready_s)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        VX_generic_buffer #(
            .DATAW   (WORD_WIDTH + TAG_WIDTH),
            .SKID    (OUT_REG >> 1),
            .OUT_REG (OUT_REG & 1)
        ) rsp_sbuf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (rsp_valid_s[i]),
            .ready_in  (rsp_ready_s[i]),
            .data_in   ({rsp_data_s[i], rsp_tag_s[i]}),
            .data_out  ({rsp_data[i],   rsp_tag[i]}), 
            .valid_out (rsp_valid[i]),
            .ready_out (rsp_ready[i])
        );
    end

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
            perf_reads       <= '0;
            perf_writes      <= '0;
            perf_crsp_stalls <= '0;
        end else begin
            perf_reads       <= perf_reads  + `PERF_CTR_BITS'(perf_reads_per_cycle);
            perf_writes      <= perf_writes + `PERF_CTR_BITS'(perf_writes_per_cycle);
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

`ifdef DBG_TRACE_CACHE_BANK

    wire [NUM_BANKS-1:0][`UP(UUID_WIDTH)-1:0] req_uuid_st0, req_uuid_st1;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        if (UUID_WIDTH != 0) begin
            assign req_uuid_st0[i] = per_bank_req_tag_unqual[i][TAG_WIDTH-1 -: UUID_WIDTH];
            assign req_uuid_st1[i] = per_bank_req_tag[i][TAG_WIDTH-1 -: UUID_WIDTH];
        end else begin
            assign req_uuid_st0[i] = '0;
            assign req_uuid_st1[i] = '0;
        end
    end

    always @(posedge clk) begin        
        for (integer i = 0; i < NUM_BANKS; ++i) begin
            if (per_bank_req_valid_unqual[i] && per_bank_req_ready_unqual[i]) begin
                if (per_bank_req_rw_unqual[i]) begin
                    `TRACE(1, ("%d: %s:%0d core-wr-req: addr=0x%0h, tag=0x%0h, byteen=%b, data=0x%0h (#%0d)\n", 
                        $time, INSTANCE_ID, i, per_bank_req_addr_unqual[i], per_bank_req_tag_unqual[i], per_bank_req_byteen_unqual[i], per_bank_req_data_unqual[i], req_uuid_st0[i]));
                end else begin
                    `TRACE(1, ("%d: %s:%0d core-rd-req: addr=0x%0h, tag=0x%0h (#%0d)\n", 
                        $time, INSTANCE_ID, i, per_bank_req_addr_unqual[i], per_bank_req_tag_unqual[i], req_uuid_st0[i]));
                end
            end

            if (per_bank_rsp_valid[i] && per_bank_rsp_ready[i]) begin
                `TRACE(1, ("%d: %s:%0d core-rd-rsp: tag=0x%0h, data=0x%0h (#%0d)\n", 
                    $time, INSTANCE_ID, i, per_bank_rsp_tag[i], per_bank_rsp_data[i], req_uuid_st1[i]));
            end
        end
    end

`endif

endmodule
