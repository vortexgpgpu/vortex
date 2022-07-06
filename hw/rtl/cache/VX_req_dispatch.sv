`include "VX_define.vh"

module VX_req_dispatch #(  
    // Size of a line in bytes
    parameter LINE_SIZE     = 4, 
    // Size of a word in bytes
    parameter WORD_SIZE     = 4,
    // Address width
    parameter ADDR_WIDTH    = 8,
    // Number of Word requests per cycle
    parameter NUM_REQS      = 4,
    // Number of banks
    parameter NUM_BANKS     = 4,
    // Number of ports per banks
    parameter NUM_PORTS     = 1,
    // core request tag size
    parameter TAG_WIDTH     = 3,

    localparam WORDS_PER_LINE   = LINE_SIZE / WORD_SIZE,
    localparam WORD_WIDTH       = WORD_SIZE * 8,
    localparam REQ_SEL_BITS     = `CLOG2(NUM_REQS),    
    localparam WORD_SEL_BITS    = `CLOG2(WORDS_PER_LINE),
    localparam BANK_SEL_BITS    = `CLOG2(NUM_BANKS),
    localparam LINE_ADDR_WIDTH  = ADDR_WIDTH - BANK_SEL_BITS - WORD_SEL_BITS
) (
    input wire                                      clk,
    input wire                                      reset,

`ifdef PERF_ENABLE
    output wire [`PERF_CTR_BITS-1:0]                bank_stalls,
`endif

    input wire [NUM_REQS-1:0]                       core_req_valid,
    input wire [NUM_REQS-1:0]                       core_req_rw,
    input wire [NUM_REQS-1:0][ADDR_WIDTH-1:0]       core_req_addr,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]        core_req_byteen,
    input wire [NUM_REQS-1:0][WORD_WIDTH-1:0]       core_req_data,
    input wire [NUM_REQS-1:0][TAG_WIDTH-1:0]        core_req_tag,
    output wire [NUM_REQS-1:0]                      core_req_ready,

    output wire [NUM_BANKS-1:0]                     per_bank_core_req_valid,
    output wire [NUM_BANKS-1:0][NUM_PORTS-1:0]      per_bank_core_req_pmask,
    output wire [NUM_BANKS-1:0]                     per_bank_core_req_rw,  
    output wire [NUM_BANKS-1:0][LINE_ADDR_WIDTH-1:0] per_bank_core_req_addr,
    output wire [NUM_BANKS-1:0][NUM_PORTS-1:0][`UP(WORD_SEL_BITS)-1:0] per_bank_core_req_wsel,
    output wire [NUM_BANKS-1:0][NUM_PORTS-1:0][WORD_SIZE-1:0] per_bank_core_req_byteen,
    output wire [NUM_BANKS-1:0][NUM_PORTS-1:0][WORD_WIDTH-1:0] per_bank_core_req_data,  
    output wire [NUM_BANKS-1:0][NUM_PORTS-1:0][`UP(REQ_SEL_BITS)-1:0] per_bank_core_req_idx,
    output wire [NUM_BANKS-1:0][NUM_PORTS-1:0][TAG_WIDTH-1:0] per_bank_core_req_tag,
    input  wire [NUM_BANKS-1:0]                     per_bank_core_req_ready
);
    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid parameter"))    
    `STATIC_ASSERT(NUM_BANKS == (1 << $clog2(NUM_BANKS)), ("invalid parameter"))
    `STATIC_ASSERT(NUM_PORTS <= NUM_REQS, ("invalid parameter"))
    `STATIC_ASSERT(NUM_PORTS <= WORDS_PER_LINE, ("invalid parameter"))

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    wire [NUM_REQS-1:0][LINE_ADDR_WIDTH-1:0]    core_req_line_addr;
    wire [NUM_REQS-1:0][`UP(WORD_SEL_BITS)-1:0] core_req_wsel;

    for (genvar i = 0; i < NUM_REQS; ++i) begin            
        if (WORDS_PER_LINE > 1) begin
            assign core_req_wsel[i] = core_req_addr[i][0 +: WORD_SEL_BITS];
        end else begin
            assign core_req_wsel[i] = 0;
        end
        assign core_req_line_addr[i] = core_req_addr[i][(BANK_SEL_BITS + WORD_SEL_BITS) +: LINE_ADDR_WIDTH];
    end

    if ((NUM_BANKS > 1) || (NUM_PORTS > 1)) begin

        wire [NUM_REQS-1:0][`UP(BANK_SEL_BITS)-1:0] core_req_bid;

        for (genvar i = 0; i < NUM_REQS; ++i) begin            
            if (NUM_BANKS > 1) begin
                assign core_req_bid[i] = core_req_addr[i][WORD_SEL_BITS +: BANK_SEL_BITS];
            end else begin
                assign core_req_bid[i] = 0;
            end
        end

        reg [NUM_BANKS-1:0][LINE_ADDR_WIDTH-1:0] per_bank_line_addr_r;
        reg [NUM_BANKS-1:0]                      per_bank_rw_r;
        wire [NUM_REQS-1:0]                      core_req_line_select;
        
        always @(*) begin
            per_bank_line_addr_r = 'x;
            for (integer i = NUM_REQS-1; i >= 0; --i) begin
                if (core_req_valid[i]) begin
                    per_bank_line_addr_r[core_req_bid[i]] = core_req_line_addr[i];
                    per_bank_rw_r[core_req_bid[i]] = core_req_rw[i];
                end     
            end
        end
        
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_req_line_select[i] = (core_req_line_addr[i] == per_bank_line_addr_r[core_req_bid[i]]) 
                                          && (core_req_rw[i] == per_bank_rw_r[core_req_bid[i]]);
        end  

        logic [NUM_BANKS-1:0]                       per_bank_core_req_valid_r;
        logic [NUM_BANKS-1:0][NUM_PORTS-1:0]        per_bank_core_req_pmask_r;
        logic [NUM_BANKS-1:0][NUM_PORTS-1:0][`UP(WORD_SEL_BITS)-1:0] per_bank_core_req_wsel_r;
        logic [NUM_BANKS-1:0][NUM_PORTS-1:0][WORD_SIZE-1:0] per_bank_core_req_byteen_r;
        logic [NUM_BANKS-1:0][NUM_PORTS-1:0][WORD_WIDTH-1:0] per_bank_core_req_data_r;
        logic [NUM_BANKS-1:0][NUM_PORTS-1:0][`UP(REQ_SEL_BITS)-1:0] per_bank_core_req_idx_r;
        logic [NUM_BANKS-1:0][NUM_PORTS-1:0][TAG_WIDTH-1:0] per_bank_core_req_tag_r;
        logic [NUM_BANKS-1:0]                       per_bank_core_req_rw_r;
        logic [NUM_BANKS-1:0][LINE_ADDR_WIDTH-1:0]  per_bank_core_req_addr_r;
        logic [NUM_REQS-1:0]                        core_req_ready_r;
        logic [NUM_BANKS-1:0][NUM_PORTS-1:0][NUM_REQS-1:0] req_select_table_r;

        always @(*) begin
            per_bank_core_req_valid_r = 0;
            per_bank_core_req_pmask_r = 0;
            per_bank_core_req_rw_r    = 'x;
            per_bank_core_req_addr_r  = 'x;
            per_bank_core_req_wsel_r  = 'x;
            per_bank_core_req_byteen_r= 'x;
            per_bank_core_req_data_r  = 'x;
            per_bank_core_req_tag_r   = 'x;
            per_bank_core_req_idx_r   = 'x;

            for (integer i = NUM_REQS-1; i >= 0; --i) begin
                if (core_req_valid[i]) begin
                    per_bank_core_req_valid_r[core_req_bid[i]]                 = 1;
                    per_bank_core_req_pmask_r[core_req_bid[i]][i % NUM_PORTS]  = (1 == NUM_PORTS) || core_req_line_select[i];
                    per_bank_core_req_wsel_r[core_req_bid[i]][i % NUM_PORTS]   = core_req_wsel[i];
                    per_bank_core_req_byteen_r[core_req_bid[i]][i % NUM_PORTS] = core_req_byteen[i];
                    per_bank_core_req_data_r[core_req_bid[i]][i % NUM_PORTS]   = core_req_data[i];
                    per_bank_core_req_idx_r[core_req_bid[i]][i % NUM_PORTS]    = `UP(REQ_SEL_BITS)'(i);
                    per_bank_core_req_tag_r[core_req_bid[i]][i % NUM_PORTS]    = core_req_tag[i];
                    per_bank_core_req_rw_r[core_req_bid[i]]                    = core_req_rw[i];
                    per_bank_core_req_addr_r[core_req_bid[i]]                  = core_req_line_addr[i];                    
                end
            end
        end

        if (NUM_PORTS > 1) begin            
            always @(*) begin
                req_select_table_r = 'x;
                for (integer i = NUM_REQS-1; i >= 0; --i) begin
                    if (core_req_valid[i]) begin
                        req_select_table_r[core_req_bid[i]][i % NUM_PORTS] = (1 << i);
                    end 
                end                
            end
            
            for (genvar i = 0; i < NUM_REQS; ++i) begin
                assign core_req_ready_r[i] = per_bank_core_req_ready[core_req_bid[i]]
                                          && core_req_line_select[i]
                                          && ((NUM_REQS <= NUM_PORTS) 
                                           || req_select_table_r[core_req_bid[i]][i % NUM_PORTS][i]);
            end
        end else begin
            always @(*) begin
                core_req_ready_r   = '0;
                req_select_table_r = '0;

                for (integer i = NUM_REQS-1; i >= 0; --i) begin
                    if (core_req_valid[i]) begin
                        req_select_table_r[core_req_bid[i]][0] = (1 << i);
                    end 
                end
            end

            for (genvar i = 0; i < NUM_REQS; ++i) begin
                assign core_req_ready_r[i] = per_bank_core_req_ready[core_req_bid[i]] 
                                          && req_select_table_r[core_req_bid[i]][0][i];
            end
        end

        assign per_bank_core_req_valid  = per_bank_core_req_valid_r;
        assign per_bank_core_req_pmask  = per_bank_core_req_pmask_r;
        assign per_bank_core_req_rw     = per_bank_core_req_rw_r;
        assign per_bank_core_req_addr   = per_bank_core_req_addr_r;
        assign per_bank_core_req_wsel   = per_bank_core_req_wsel_r;
        assign per_bank_core_req_byteen = per_bank_core_req_byteen_r;
        assign per_bank_core_req_data   = per_bank_core_req_data_r;
        assign per_bank_core_req_tag    = per_bank_core_req_tag_r;
        assign per_bank_core_req_idx    = per_bank_core_req_idx_r;
        assign core_req_ready = core_req_ready_r;

    end else begin

        assign per_bank_core_req_valid  = core_req_valid;
        assign per_bank_core_req_pmask  = 1;
        assign per_bank_core_req_rw     = core_req_rw;
        assign per_bank_core_req_addr   = core_req_line_addr;
        assign per_bank_core_req_wsel   = core_req_wsel;
        assign per_bank_core_req_byteen = core_req_byteen;
        assign per_bank_core_req_data   = core_req_data;
        assign per_bank_core_req_tag    = core_req_tag;
        assign per_bank_core_req_idx    = 0;
        assign core_req_ready = per_bank_core_req_ready;

    end
    
`ifdef PERF_ENABLE
    reg [NUM_REQS-1:0] core_req_sel_r;

    always @(*) begin
        core_req_sel_r = 0;
        for (integer i = 0; i < NUM_REQS; ++i) begin
            if (core_req_valid[i]) begin
                core_req_sel_r[i] = per_bank_core_req_ready[core_req_bid[i]];
            end
        end
    end

    reg [`PERF_CTR_BITS-1:0] bank_stalls_r;
    wire [$clog2(NUM_REQS+1)-1:0] bank_stall_cnt;

    wire [NUM_REQS-1:0] bank_stall_mask = core_req_sel_r & ~core_req_ready;
    `POP_COUNT(bank_stall_cnt, bank_stall_mask);

    always @(posedge clk) begin
        if (reset) begin
            bank_stalls_r <= 0;
        end else begin
            bank_stalls_r <= bank_stalls_r + `PERF_CTR_BITS'(bank_stall_cnt);
        end
    end

    assign bank_stalls = bank_stalls_r;
`endif

endmodule
