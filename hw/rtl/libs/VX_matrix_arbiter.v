`include "VX_define.vh"

module VX_matrix_arbiter #(
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

        reg [N-1:1] state [0:N-1];  
        wire [N-1:0] pri [0:N-1];

        genvar i, j;           

        for (i = 0; i < N; i++) begin      
            for (j = 0; j < N; j++) begin
                if (j > i) begin
                    assign pri[j][i] = requests[i] && state[i][j];
                end 
                else if (j < i) begin
                    assign pri[j][i] = requests[i] && !state[j][i];
                end 
                else begin
                    assign pri[j][i] = 0;            
                end
            end

            assign grant_onehot[i] = requests[i] && !(| pri[i]);
        end
        
        for (i = 0; i < N; i++) begin      
            for (j = i + 1; j < N; j++) begin
                always @(posedge clk) begin                       
                    if (reset) begin         
                        state[i][j] <= 0;
                    end 
                    else begin
                        state[i][j] <= (state[i][j] || grant_onehot[j]) && !grant_onehot[i];
                    end
                end
            end
        end

        VX_encoder_onehot #(
            .N(N)
        ) encoder (
            .onehot (grant_onehot),
            `UNUSED_PIN (valid),
            .value  (grant_index)
        );

        assign grant_valid = (| requests);

    end
    
endmodule