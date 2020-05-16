`include "VX_define.vh"

module VX_matrix_arbiter #(
    parameter N = 0
) (
    input  wire                  clk,
    input  wire                  reset,
    input  wire [N-1:0]          requests,   
    output wire                  grant_valid,  
    output wire [N-1:0]          grant_onehot,   
    output wire [`LOG2UP(N)-1:0] grant_index
  );

    reg [N-1:0] state [0:N-1];  
    wire [N-1:0] dis [0:N-1];

    genvar i, j;
    
    for (i = 0; i < N; ++i) begin      
        for (j = i + 1; j < N; ++j) begin
            always @(posedge clk) begin                       
                if (reset) begin         
                    state[i][j] <= 0;
                end else begin
                    state[i][j] <= (state[i][j] || grant_onehot[j]) && ~grant_onehot[i];
                end
            end
        end
    end           

    for (i = 0; i < N; ++i) begin      
        for (j = 0; j < N; ++j) begin
            if (j > i) begin
                assign dis[j][i] = requests[i] & state[i][j];
            end else if (j < i) begin
                assign dis[j][i] = requests[i] & ~state[j][i];
            end else begin
                assign dis[j][i] = 0;            
            end
        end

        assign grant_onehot[i] = requests[i] & ~(| dis[i]);
    end

    VX_encoder_onehot #(
        .N(N)
    ) encoder (
        .onehot(grant_onehot),
        .valid(grant_valid),
        .value(grant_index)
    );
    
endmodule