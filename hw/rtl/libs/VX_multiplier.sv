`include "VX_platform.vh"

`TRACING_OFF
module VX_multiplier #(
    parameter A_WIDTH = 1,
    parameter B_WIDTH = A_WIDTH,
    parameter R_WIDTH = A_WIDTH + B_WIDTH,
    parameter SIGNED  = 0,
    parameter LATENCY = 0
) (
    input wire clk,    
    input wire enable,
    input wire [A_WIDTH-1:0]  dataa,
    input wire [B_WIDTH-1:0]  datab,
    output wire [R_WIDTH-1:0] result
);
    `STATIC_ASSERT ((LATENCY <= 3), ("invalid parameter"))

    wire [A_WIDTH-1:0] dataa_w;
    wire [B_WIDTH-1:0] datab_w;
    wire [A_WIDTH+B_WIDTH-1:0] result_w;
    `UNUSED_VAR (result_w)

    if (SIGNED != 0) begin
        assign result_w = $signed(dataa_w) * $signed(datab_w);
    end else begin
        assign result_w = dataa_w * datab_w;
    end
    
    if (LATENCY == 0) begin
        assign dataa_w = dataa;
        assign datab_w = datab;
        assign result  = R_WIDTH'(result_w);
    end else begin        
        if (LATENCY >= 2) begin
            reg [A_WIDTH-1:0] dataa_p [LATENCY-2:0];
            reg [B_WIDTH-1:0] datab_p [LATENCY-2:0];
            always @(posedge clk) begin
                if (enable) begin
                    dataa_p[0] <= dataa;
                    datab_p[0] <= datab;
                end
            end
            for (genvar i = 2; i < LATENCY; ++i) begin
                always @(posedge clk) begin
                    if (enable) begin
                        dataa_p[i-1] <= dataa_p[i-2];
                        datab_p[i-1] <= datab_p[i-2];
                    end
                end
            end
            assign dataa_w = dataa_p[LATENCY-2];
            assign datab_w = datab_p[LATENCY-2];
        end else begin
            assign dataa_w = dataa;
            assign datab_w = datab;
        end
        reg [R_WIDTH-1:0] result_r;
        always @(posedge clk) begin
            if (enable) begin
                result_r <= R_WIDTH'(result_w);
            end
        end
        assign result = result_r; 
    end

endmodule
`TRACING_ON
