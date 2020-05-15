`include "VX_define.vh"

module VX_matrix_arbiter #(
    parameter N = 0
) (
    input  wire         clk,
    input  wire         reset,
    input  wire [N-1:0] inputs,
    output wire [N-1:0] grant
  );

    reg [N-1:1][N-1:0] pri;    

    always @(posedge clk) begin
        if (reset) begin            
            integer i, j;
            for (i = 0; i < N; ++i) begin      
                for (j = 0; j < N; ++j) begin
                    pri[i][j] <= 1;
                end
            end
        end else begin
            integer i, j;
            for (i = 0; i < N; ++i) begin                        
                if (grant[i]) begin
                    for (j = 0; j < N; ++j) begin
                        if (j > i)
                            pri[j][i] <= 1;
                        else if (j < i)
                            pri[i][j] <= 0;
                    end
                end
            end        
        end
    end    

    genvar i, j;    

    for (i = 0; i < N; ++i) begin      

        wire [N-1:0] dis;

        for (j = 0; j < N; ++j) begin
            if (j > i) begin
                assign dis[j] = inputs[j] & pri[j][i];
            end else if (j < i) begin
                assign dis[j] = inputs[j] & ~pri[i][j];
            end else begin
                assign dis[j] = 0;            
            end
        end

        assign grant[i] = inputs[i] & ~(| dis);
    end
    
endmodule