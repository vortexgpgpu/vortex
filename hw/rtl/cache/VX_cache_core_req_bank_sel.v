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
    parameter BANK_ADDR_OFFSET  = 0,
    // buffer the output
    parameter BUFFERED          = 0
) (
    input wire                                      clk,
    input wire                                      reset,

    output wire [63:0]                              bank_stalls,

    input wire [NUM_REQS-1:0]                       core_req_valid,
    input wire [NUM_REQS-1:0]                       core_req_rw,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]        core_req_byteen,
    input wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0] core_req_addr,
    input wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]      core_req_data,
    input wire [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0]   core_req_tag,
    output wire [NUM_REQS-1:0]                      core_req_ready,

    output wire [NUM_BANKS-1:0]                     per_bank_core_req_valid, 
    output wire [NUM_BANKS-1:0][`REQS_BITS-1:0]     per_bank_core_req_tid,
    output wire [NUM_BANKS-1:0]                     per_bank_core_req_rw,  
    output wire [NUM_BANKS-1:0][WORD_SIZE-1:0]      per_bank_core_req_byteen,
    output wire [NUM_BANKS-1:0][`WORD_ADDR_WIDTH-1:0] per_bank_core_req_addr,
    output wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0] per_bank_core_req_tag,
    output wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]    per_bank_core_req_data,
    input  wire [NUM_BANKS-1:0]                     per_bank_core_req_ready
);
    if (NUM_BANKS > 1) begin

        reg [NUM_BANKS-1:0]                         per_bank_core_req_valid_r;
        reg [NUM_BANKS-1:0][`REQS_BITS-1:0]         per_bank_core_req_tid_r;
        reg [NUM_BANKS-1:0]                         per_bank_core_req_rw_r;
        reg [NUM_BANKS-1:0][WORD_SIZE-1:0]          per_bank_core_req_byteen_r;
        reg [NUM_BANKS-1:0][`WORD_ADDR_WIDTH-1:0]   per_bank_core_req_addr_r;
        reg [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]     per_bank_core_req_tag_r;
        reg [NUM_BANKS-1:0][`WORD_WIDTH-1:0]        per_bank_core_req_data_r;
        reg [NUM_BANKS-1:0]                         per_bank_core_req_stall;

        reg [NUM_REQS-1:0]                          core_req_ready_r;
        reg [NUM_BANKS-1:0]                         core_req_sel_r;
        wire [NUM_REQS-1:0][`BANK_SELECT_BITS-1:0]  core_req_bid;

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_req_bid[i] = `BANK_SELECT_ADDR(core_req_addr[i]);
        end

        always @(*) begin
            per_bank_core_req_valid_r = 0;            
            per_bank_core_req_tid_r   = 'x;
            per_bank_core_req_rw_r    = 'x;
            per_bank_core_req_byteen_r= 'x;
            per_bank_core_req_addr_r  = 'x;
            per_bank_core_req_tag_r   = 'x;
            per_bank_core_req_data_r  = 'x;

            for (integer i = NUM_REQS-1; i >= 0; --i) begin                                                
                if (core_req_valid[i]) begin                    
                    per_bank_core_req_valid_r[core_req_bid[i]] = 1;
                    per_bank_core_req_tid_r[core_req_bid[i]]   = `REQS_BITS'(i);                    
                    per_bank_core_req_rw_r[core_req_bid[i]]    = core_req_rw[i];
                    per_bank_core_req_byteen_r[core_req_bid[i]]= core_req_byteen[i];
                    per_bank_core_req_addr_r[core_req_bid[i]]  = core_req_addr[i];
                    per_bank_core_req_tag_r[core_req_bid[i]]   = core_req_tag[i];
                    per_bank_core_req_data_r[core_req_bid[i]]  = core_req_data[i];
                end
            end
        end

        always @(*) begin
            core_req_ready_r = 0;
            core_req_sel_r   = 0;
            
            for (integer j = 0; j < NUM_BANKS; ++j) begin
                for (integer i = 0; i < NUM_REQS; ++i) begin
                    if (core_req_valid[i] && (core_req_bid[i] == `BANK_SELECT_BITS'(j))) begin
                        core_req_ready_r[i] = ~per_bank_core_req_stall[j];                        
                        core_req_sel_r[i]   = 1;
                        break;
                    end
                end
            end
        end

        reg [63:0] bank_stalls_r;
        always @(posedge clk) begin
            if (reset) begin
                bank_stalls_r <= 0;
            end else begin
                bank_stalls_r <= bank_stalls_r + 64'($countones(core_req_valid & ~core_req_sel_r));
            end
        end

        for (genvar i = 0; i < NUM_BANKS; ++i) begin
            assign per_bank_core_req_stall[i] = ~per_bank_core_req_ready[i] & per_bank_core_req_valid[i];
            VX_pipe_register #(
                .DATAW  (1 + `REQS_BITS + 1 + WORD_SIZE + `WORD_ADDR_WIDTH + CORE_TAG_WIDTH + `WORD_WIDTH),
                .RESETW (1),
                .DEPTH  (BUFFERED)
            ) pipe_reg (
                .clk      (clk),
                .reset    (reset),
                .enable   (~per_bank_core_req_stall[i]),
                .data_in  ({per_bank_core_req_valid_r[i], per_bank_core_req_tid_r[i], per_bank_core_req_rw_r[i], per_bank_core_req_byteen_r[i], per_bank_core_req_addr_r[i], per_bank_core_req_tag_r[i], per_bank_core_req_data_r[i]}),
                .data_out ({per_bank_core_req_valid[i],   per_bank_core_req_tid[i],   per_bank_core_req_rw[i],   per_bank_core_req_byteen[i],  per_bank_core_req_addr[i],    per_bank_core_req_tag[i],   per_bank_core_req_data[i]})
            );
        end

        assign core_req_ready = core_req_ready_r;
        assign bank_stalls    = bank_stalls_r;
        
    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign bank_stalls                 = 0;
        assign per_bank_core_req_valid     = core_req_valid;
        assign per_bank_core_req_tid[0]    = 0;
        assign per_bank_core_req_rw[0]     = core_req_rw;
        assign per_bank_core_req_byteen[0] = core_req_byteen;
        assign per_bank_core_req_addr[0]   = core_req_addr;
        assign per_bank_core_req_tag[0]    = core_req_tag;
        assign per_bank_core_req_data[0]   = core_req_data;
        assign core_req_ready[0]           = per_bank_core_req_ready;

    end   

endmodule