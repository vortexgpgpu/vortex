`include "VX_platform.vh"

module VX_rr_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOCK_ENABLE  = 0,
    parameter LOG_NUM_REQS = $clog2(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,          
    input  wire                     enable,
    input  wire [NUM_REQS-1:0]      requests, 
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,   
    output wire                     grant_valid
  );

    if (NUM_REQS == 1)  begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        
        assign grant_index  = 0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin

        reg [LOG_NUM_REQS-1:0] grant_table [NUM_REQS-1:0];
        reg [LOG_NUM_REQS-1:0] state;  
        
        always @(*) begin
            for (integer i = 0; i < NUM_REQS; i++) begin  
                grant_table[i] = LOG_NUM_REQS'(i);    
                for (integer j = 0; j < NUM_REQS; j++) begin    
                    if (requests[(i+j) % NUM_REQS]) begin                        
                        grant_table[i] = LOG_NUM_REQS'((i+j) % NUM_REQS);
                    end
                end
            end
        end  

        always @(posedge clk) begin                       
            if (reset) begin         
                state <= 0;
            end else if (!LOCK_ENABLE || enable) begin
                state <= grant_table[state];
            end
        end      

        reg [NUM_REQS-1:0] grant_onehot_r;
        always @(*) begin
            grant_onehot_r = NUM_REQS'(0);
            grant_onehot_r[grant_table[state]] = 1;
        end

        assign grant_index  = grant_table[state];
        assign grant_onehot = grant_onehot_r; 
        assign grant_valid  = (| requests);

    end
    
endmodule