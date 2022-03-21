`include "VX_platform.vh"

`TRACING_OFF
module VX_pipe_register #( 
    parameter DATAW  = 1, 
    parameter RESETW = DATAW, 
    parameter DEPTH  = 1
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire [DATAW-1:0]  data_in,
    output wire [DATAW-1:0] data_out
);

    if (DEPTH == 0) begin        
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (enable)
        assign data_out = data_in;  
    end else if (DEPTH == 1) begin        
        if (RESETW == 0) begin
            `UNUSED_VAR (reset)
            reg [DATAW-1:0] value;

            always @(posedge clk) begin
                if (enable) begin
                    value <= data_in;
                end
            end
            assign data_out = value;
        end else if (RESETW == DATAW) begin
            reg [DATAW-1:0] value;

            always @(posedge clk) begin
                if (reset) begin
                    value <= RESETW'(0);
                end else if (enable) begin
                    value <= data_in;
                end
            end
            assign data_out = value;
        end else begin
            reg [DATAW-RESETW-1:0] value_d;
            reg [RESETW-1:0]       value_r;

            always @(posedge clk) begin
                if (reset) begin
                    value_r <= RESETW'(0);
                end else if (enable) begin
                    value_r <= data_in[DATAW-1:DATAW-RESETW];
                end
            end

            always @(posedge clk) begin
                if (enable) begin
                    value_d <= data_in[DATAW-RESETW-1:0];
                end
            end        
            assign data_out = {value_r, value_d};
        end
    end else begin
        VX_shift_register #(
            .DATAW  (DATAW),
            .RESETW (RESETW),
            .DEPTH  (DEPTH)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (data_in),
            .data_out (data_out)
        );
    end

endmodule
`TRACING_ON
