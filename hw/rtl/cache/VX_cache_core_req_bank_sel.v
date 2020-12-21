`include "VX_cache_config.vh"

module VX_cache_core_req_bank_sel #(
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE      = 1, 
    // Number of banks
    parameter NUM_BANKS      = 1, 
    // Number of Word requests per cycle
    parameter NUM_REQS       = 1
) (
    input  wire [NUM_REQS-1:0]                       core_req_valid,
    input  wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0] core_req_addr, 
    output wire [NUM_REQS-1:0]                       core_req_ready,
    output wire [NUM_BANKS-1:0]                      per_bank_valid,
    output wire [NUM_BANKS-1:0][`REQS_BITS-1:0]      per_bank_tid, 
    input  wire [NUM_BANKS-1:0]                      per_bank_ready
);     
    if (NUM_BANKS > 1) begin        
        reg [NUM_BANKS-1:0]                 per_bank_valid_r;
        reg [NUM_BANKS-1:0][`REQS_BITS-1:0] per_bank_tid_r;
        reg [NUM_REQS-1:0]                  core_req_ready_r;
        wire [NUM_REQS-1:0][`BANK_BITS-1:0] core_req_bid;

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_req_bid[i] = core_req_addr[i][`BANK_SELECT_ADDR_RNG];
        end

        always @(*) begin
            per_bank_valid_r = 0;            
            per_bank_tid_r   = 'x;
            for (integer i = NUM_REQS-1; i >= 0; --i) begin                                                
                if (core_req_valid[i]) begin                    
                    per_bank_valid_r[core_req_bid[i]] = 1;
                    per_bank_tid_r[core_req_bid[i]]   = `REQS_BITS'(i);                    
                end
            end
        end

        always @(*) begin
            core_req_ready_r = 0;
            for (integer j = 0; j < NUM_BANKS; ++j) begin
                for (integer i = 0; i < NUM_REQS; ++i) begin
                    if (core_req_valid[i] && (core_req_bid[i] == `BANK_BITS'(j))) begin
                        core_req_ready_r[i] = per_bank_ready[j];
                        break;
                    end
                end
            end
        end

        assign per_bank_valid = per_bank_valid_r;
        assign per_bank_tid   = per_bank_tid_r;
        assign core_req_ready = core_req_ready_r;

    end else begin   

        `UNUSED_VAR (core_req_valid)
        `UNUSED_VAR (core_req_addr)
        assign per_bank_valid = core_req_valid;
        assign per_bank_tid = 0;
        assign core_req_ready[0] = per_bank_ready;

    end   

endmodule