
`include "VX_cache_config.vh"

module VX_cache_core_req_bank_sel #(
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 0, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 0, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 0, 
    // Number of Word requests per cycle {1, 2, 4, 8, ...}
    parameter NUM_REQUESTS                  = 0
) (
    input  wire [NUM_REQUESTS-1:0]                   core_req_valid,
    input  wire [NUM_REQUESTS-1:0][31:0]             core_req_addr,
    
    output reg  [NUM_BANKS-1:0][NUM_REQUESTS-1:0]    per_bank_valids
);

    generate
        integer i;
        always @(*) begin
            per_bank_valids = 0;
            for (i = 0; i < NUM_REQUESTS; i++) begin
                if (NUM_BANKS == 1) begin
                    // If there is only one bank, then only map requests to that bank
                    per_bank_valids[0][i] = core_req_valid[i];
                end else begin
                    per_bank_valids[core_req_addr[i][`BANK_SELECT_ADDR_RNG]][i] = core_req_valid[i];
                end
            end
        end
    endgenerate

endmodule