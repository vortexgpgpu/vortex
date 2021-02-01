`include "VX_cache_config.vh"

module VX_cache_core_req_bank_sel #(  
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE   = 64, 
    // Size of a word in bytes
    parameter WORD_SIZE         = 4, 
    // Number of banks
    parameter NUM_BANKS         = 4,
    // Number of Word requests per cycle
    parameter NUM_REQS          = 4,
    // core request tag size
    parameter CORE_TAG_WIDTH    = 3,
    // bank offset from beginning of index range
    parameter BANK_ADDR_OFFSET  = 0
) (
    input wire                                      clk,
    input wire                                      reset,

`ifdef PERF_ENABLE
    output wire [63:0]                              bank_stalls,
`endif

    input wire [NUM_REQS-1:0]                       core_req_valid,
    input wire [NUM_REQS-1:0]                       core_req_rw,
    input wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0] core_req_addr,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]        core_req_byteen,
    input wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]      core_req_data,
    input wire [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0]   core_req_tag,
    output wire [NUM_REQS-1:0]                      core_req_ready,

    output wire [NUM_BANKS-1:0]                     per_bank_core_req_valid, 
    output wire [NUM_BANKS-1:0]                     per_bank_core_req_rw,  
    output wire [NUM_BANKS-1:0][`LINE_ADDR_WIDTH-1:0] per_bank_core_req_addr,
    output wire [NUM_BANKS-1:0][`UP(`WORD_SELECT_BITS)-1:0] per_bank_core_req_wsel,
    output wire [NUM_BANKS-1:0][WORD_SIZE-1:0]      per_bank_core_req_byteen,
    output wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]    per_bank_core_req_data,
    output wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0] per_bank_core_req_tag,
    output wire [NUM_BANKS-1:0][`REQS_BITS-1:0]     per_bank_core_req_tid,
    input  wire [NUM_BANKS-1:0]                     per_bank_core_req_ready
);
    `STATIC_ASSERT (NUM_REQS >= NUM_BANKS, ("invalid number of banks"));

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)
        
    wire [NUM_REQS-1:0][`LINE_ADDR_WIDTH-1:0] core_req_line_addr;
    wire [NUM_REQS-1:0][`UP(`WORD_SELECT_BITS)-1:0] core_req_wsel;
    wire [NUM_REQS-1:0][`UP(`BANK_SELECT_BITS)-1:0] core_req_bid;

    for (genvar i = 0; i < NUM_REQS; i++) begin    
        if (BANK_ADDR_OFFSET == 0) begin
            assign core_req_line_addr[i] = `LINE_SELECT_ADDR0(core_req_addr[i]);
        end else begin
            assign core_req_line_addr[i] = `LINE_SELECT_ADDRX(core_req_addr[i]);
        end
        assign core_req_wsel[i] = core_req_addr[i][`UP(`WORD_SELECT_BITS)-1:0];
    end   

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        if (NUM_BANKS > 1) begin
            assign core_req_bid[i] = `BANK_SELECT_ADDR(core_req_addr[i]);
        end else begin
            assign core_req_bid[i] = 0;
        end
    end

    if (NUM_REQS > 1) begin

        reg [NUM_BANKS-1:0]                         per_bank_core_req_valid_r;
        reg [NUM_BANKS-1:0]                         per_bank_core_req_rw_r;
        reg [NUM_BANKS-1:0][`LINE_ADDR_WIDTH-1:0]   per_bank_core_req_addr_r;
        reg [NUM_BANKS-1:0][`UP(`WORD_SELECT_BITS)-1:0] per_bank_core_req_wsel_r;
        reg [NUM_BANKS-1:0][WORD_SIZE-1:0]          per_bank_core_req_byteen_r;
        reg [NUM_BANKS-1:0][`WORD_WIDTH-1:0]        per_bank_core_req_data_r;
        reg [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]     per_bank_core_req_tag_r;        
        reg [NUM_BANKS-1:0][`REQS_BITS-1:0]         per_bank_core_req_tid_r;

        reg [NUM_REQS-1:0]                          core_req_ready_r;        

        always @(*) begin
            per_bank_core_req_valid_r = 0;            
            per_bank_core_req_rw_r    = 'x;
            per_bank_core_req_addr_r  = 'x;
            per_bank_core_req_wsel_r  = 'x;
            per_bank_core_req_byteen_r= 'x;
            per_bank_core_req_data_r  = 'x;
            per_bank_core_req_tag_r   = 'x;
            per_bank_core_req_tid_r   = 'x;

            for (integer i = NUM_REQS-1; i >= 0; --i) begin                                    
                if (core_req_valid[i]) begin                    
                    per_bank_core_req_valid_r[core_req_bid[i]] = 1;
                    per_bank_core_req_rw_r[core_req_bid[i]]    = core_req_rw[i];
                    per_bank_core_req_addr_r[core_req_bid[i]]  = core_req_line_addr[i];
                    per_bank_core_req_wsel_r[core_req_bid[i]]  = core_req_wsel[i];
                    per_bank_core_req_byteen_r[core_req_bid[i]]= core_req_byteen[i];
                    per_bank_core_req_data_r[core_req_bid[i]]  = core_req_data[i];
                    per_bank_core_req_tag_r[core_req_bid[i]]   = core_req_tag[i];
                    per_bank_core_req_tid_r[core_req_bid[i]]   = `REQS_BITS'(i);  
                end
            end
        end

        always @(*) begin
            core_req_ready_r = 0;            
            for (integer j = 0; j < NUM_BANKS; ++j) begin
                for (integer i = 0; i < NUM_REQS; ++i) begin
                    if (core_req_valid[i] && (core_req_bid[i] == `BANK_SELECT_BITS'(j))) begin
                        core_req_ready_r[i] = per_bank_core_req_ready[j];
                        break;
                    end
                end
            end
        end

        assign per_bank_core_req_valid  = per_bank_core_req_valid_r;        
        assign per_bank_core_req_rw     = per_bank_core_req_rw_r;        
        assign per_bank_core_req_addr   = per_bank_core_req_addr_r;
        assign per_bank_core_req_wsel   = per_bank_core_req_wsel_r;
        assign per_bank_core_req_byteen = per_bank_core_req_byteen_r;        
        assign per_bank_core_req_data   = per_bank_core_req_data_r;
        assign per_bank_core_req_tag    = per_bank_core_req_tag_r;
        assign per_bank_core_req_tid    = per_bank_core_req_tid_r;
        assign core_req_ready = core_req_ready_r;
        
    end else begin

        assign per_bank_core_req_valid     = core_req_valid[0];        
        assign per_bank_core_req_rw[0]     = core_req_rw[0];        
        assign per_bank_core_req_addr[0]   = core_req_line_addr[0];
        assign per_bank_core_req_wsel[0]   = core_req_wsel[0];
        assign per_bank_core_req_byteen[0] = core_req_byteen[0];        
        assign per_bank_core_req_data[0]   = core_req_data[0];
        assign per_bank_core_req_tag[0]    = core_req_tag[0];
        assign per_bank_core_req_tid[0]    = 0;
        assign core_req_ready[0] = per_bank_core_req_ready;
    end       

`ifdef PERF_ENABLE
    reg [NUM_REQS-1:0] core_req_sel_r;

    always @(*) begin
        core_req_sel_r = 0;
        for (integer j = 0; j < NUM_BANKS; ++j) begin
            for (integer i = 0; i < NUM_REQS; ++i) begin
                if (core_req_valid[i] && (core_req_bid[i] == `UP(`BANK_SELECT_BITS)'(j))) begin
                    core_req_sel_r[i] = per_bank_core_req_ready[j];
                end
            end
        end
    end

    reg [63:0] bank_stalls_r;

    always @(posedge clk) begin
        if (reset) begin
            bank_stalls_r <= 0;
        end else begin
            bank_stalls_r <= bank_stalls_r + 64'($countones(core_req_sel_r & ~core_req_ready));
        end
    end

    assign bank_stalls = bank_stalls_r;
`endif

endmodule