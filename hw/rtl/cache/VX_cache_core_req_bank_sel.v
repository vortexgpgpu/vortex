`include "VX_cache_config.vh"

module VX_cache_core_req_bank_sel #(
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE    = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE         = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1, 
    // Number of Word requests per cycle
    parameter NUM_REQS          = 1,
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 1
) (
    input  wire [NUM_REQS-1:0]                       core_req_valid,
    input  wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0] core_req_addr, 
    output wire [`CORE_REQ_TAG_COUNT-1:0]            core_req_ready, 

    output wire [NUM_BANKS-1:0][NUM_REQS-1:0]        per_bank_valid,
    input  wire [NUM_BANKS-1:0]                      per_bank_ready    
);     
    if (NUM_BANKS > 1) begin

        reg [NUM_BANKS-1:0][NUM_REQS-1:0] per_bank_valid_r;                

        always @(*) begin
            per_bank_valid_r = 0;
            for (integer i = 0; i < NUM_REQS; i++) begin
                per_bank_valid_r[core_req_addr[i][`BANK_SELECT_ADDR_RNG]][i] = core_req_valid[i];
            end
        end        

        if (CORE_TAG_ID_BITS != 0) begin
            
            reg [NUM_BANKS-1:0] per_bank_ready_other, per_bank_ready_ignore;
            
            always @(*) begin
                per_bank_ready_other  = {NUM_BANKS{1'b1}};
                per_bank_ready_ignore = {NUM_BANKS{1'b1}}; 

                for (integer i = 0; i < NUM_REQS; i++) begin
                    per_bank_ready_ignore[core_req_addr[i][`BANK_SELECT_ADDR_RNG]] = 1'b0;
                end           
                
                for (integer i = 0; i < NUM_BANKS; i++) begin
                    for (integer j = 0; j < NUM_BANKS; j++) begin
                        if (i != j) begin
                            per_bank_ready_other[i] &= (per_bank_ready[j] | per_bank_ready_ignore[j]);
                        end
                    end
                end
            end    

            for (genvar i = 0; i < NUM_BANKS; i++) begin
                for (genvar j = 0; j < NUM_REQS; j++) begin
                    assign per_bank_valid[i][j] = per_bank_valid_r[i][j] && per_bank_ready_other[i];
                end
            end

            assign core_req_ready[0] = & (per_bank_ready | per_bank_ready_ignore);        
        
        end else begin

            assign per_bank_valid = per_bank_valid_r;

            for (genvar i = 0; i < NUM_REQS; i++) begin
                assign core_req_ready[i] = per_bank_ready[core_req_addr[i][`BANK_SELECT_ADDR_RNG]];
            end

        end

    end else begin   

        `UNUSED_VAR (core_req_addr)
        assign per_bank_valid = core_req_valid;
        assign core_req_ready[0] = per_bank_ready;

    end   

endmodule