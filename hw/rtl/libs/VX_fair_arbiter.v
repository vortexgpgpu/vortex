`include "VX_define.vh"

module VX_fair_arbiter #(
    parameter N = 0
) (
    input  wire                  clk,
    input  wire                  reset,
    input  wire [N-1:0]          requests,           
    output wire [`LOG2UP(N)-1:0] grant_index,
    output wire [N-1:0]          grant_onehot,   
    output wire                  grant_valid
  );


    if (N == 1)  begin        
        
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign grant_index  = 0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin
    

       reg  [N-1:0] requests_use;
       wire [N-1:0] update_value;
       wire [N-1:0] late_value;

       wire         refill;
       wire [N-1:0] refill_value;
       reg  [N-1:0] refill_original;

       always @(posedge clk) begin
            if (reset) begin
                requests_use    <= 0;
                refill_original <= 0;
            end else if (refill) begin
                requests_use    <= refill_value;
                refill_original <= refill_value;
            end else begin
                requests_use <= update_value;
            end
       end

       assign refill       = (requests_use == 0);
       assign refill_value = requests;
        
        reg [N-1:0] grant_onehot_r; 

        VX_priority_encoder # (
            .N(N)
        ) priority_encoder (
            .data_in   (requests_use),
            .data_out  (grant_index ),
            .valid_out (grant_valid )
        );

        always @(*) begin
            grant_onehot_r = N'(0);
            grant_onehot_r[grant_index] = 1;
        end
        assign grant_onehot = grant_onehot_r;    
        assign late_value   =  ((refill_original ^ requests) & ~refill_original);
        assign update_value = (requests_use & ~grant_onehot_r) | late_value;

    end
    
endmodule
