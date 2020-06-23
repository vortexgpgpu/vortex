
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
    input  wire [NUM_REQUESTS-1:0]                       core_req_valid,
`IGNORE_WARNINGS_BEGIN    
    input  wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0] core_req_addr,    
`IGNORE_WARNINGS_END
    input wire  [NUM_BANKS-1:0]                          per_bank_ready,
    output reg  [NUM_BANKS-1:0][NUM_REQUESTS-1:0]        per_bank_valid,
    output wire                                          core_req_ready 
);     
    integer i;

    if (NUM_BANKS == 1) begin
        always @(*) begin            
            per_bank_valid = 0;
            for (i = 0; i < NUM_REQUESTS; i++) begin                    
                per_bank_valid[0][i] = core_req_valid[i];
            end
        end        
        assign core_req_ready = per_bank_ready;
    end else begin            
        reg [NUM_BANKS-1:0] per_bank_ready_sel;
        always @(*) begin
            per_bank_valid = 0;
            per_bank_ready_sel = {NUM_BANKS{1'b1}};
            for (i = 0; i < NUM_REQUESTS; i++) begin
                per_bank_valid[core_req_addr[i][`BANK_SELECT_ADDR_RNG]][i] = core_req_valid[i];
                per_bank_ready_sel[core_req_addr[i][`BANK_SELECT_ADDR_RNG]] = 0;
            end
        end        
        assign core_req_ready = & (per_bank_ready | per_bank_ready_sel);
    end

endmodule