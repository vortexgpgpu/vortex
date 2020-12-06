`include "VX_platform.vh"

module VX_fixed_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOCK_ENABLE  = 0,
    parameter LOG_NUM_REQS = $clog2(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire [NUM_REQS-1:0]      requests,           
    input  wire                     enable,
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,   
    output wire                     grant_valid
  );

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (enable)

    if (NUM_REQS == 1)  begin        
        
        assign grant_index  = 0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin
    
        reg [LOG_NUM_REQS-1:0] grant_index_r;
        reg [NUM_REQS-1:0] grant_onehot_r;

        always @(*) begin
            grant_index_r  = 'x;
            grant_onehot_r = 'x;
            for (integer i = 0; i < NUM_REQS; ++i) begin            
                if (requests[i]) begin
                    grant_index_r  = LOG_NUM_REQS'(i);
                    grant_onehot_r = NUM_REQS'(0);
                    grant_onehot_r[i] = 1;
                    break;
                end
            end
        end      

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);
    end
    
endmodule