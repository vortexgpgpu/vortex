`include "VX_platform.vh"

module VX_rr_arbiter #(
    parameter N = 1
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

        reg [`CLOG2(N)-1:0] grant_table [0:N-1];
        reg [`CLOG2(N)-1:0] state;  
        reg [N-1:0] grant_onehot_r;

        always @(*) begin
            for (integer i = 0; i < N; i++) begin  
                grant_table[i] = `CLOG2(N)'(i);    
                for (integer j = 0; j < N; j++) begin    
                    if (requests[(i+j) % N]) begin                        
                        grant_table[i] = `CLOG2(N)'((i+j) % N);
                    end
                end
            end
            grant_onehot_r = N'(0);
            grant_onehot_r[grant_table[state]] = 1;
        end  

        always @(posedge clk) begin                       
            if (reset) begin         
                state <= 0;
            end else begin
                state <= grant_table[state];
            end
        end      

        assign grant_index  = grant_table[state];
        assign grant_onehot = grant_onehot_r; 
        assign grant_valid  = (| requests); 
    end
    
endmodule