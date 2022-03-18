`include "VX_rop_define.vh"

module VX_rop_compare #(
    parameter NUM_LANES = 4,
    parameter DATAW = 32
) (
    input wire clk,
    input wire reset,

    // Inputs
    input wire [NUM_LANES-1:0][`ROP_DEPTH_FUNC_BITS-1:0] func,
    input wire [NUM_LANES-1:0][DATAW-1:0] a,
    input wire [NUM_LANES-1:0][DATAW-1:0] b,

    // Outputs
    output reg [NUM_LANES-1:0]result
);

    always @(*) begin
        for (integer i = 0; i < NUM_LANES; i = i + 1) begin
            case (func[i])
            `ROP_DEPTH_FUNC_NEVER    : result[i] = 0;
            `ROP_DEPTH_FUNC_LESS     : result[i] = (a[i] < b[i]);
            `ROP_DEPTH_FUNC_EQUAL    : result[i] = (a[i] == b[i]);
            `ROP_DEPTH_FUNC_LEQUAL   : result[i] = (a[i] <= b[i]);
            `ROP_DEPTH_FUNC_GREATER  : result[i] = (a[i] > b[i]);
            `ROP_DEPTH_FUNC_NOTEQUAL : result[i] = (a[i] != b[i]);
            `ROP_DEPTH_FUNC_GEQUAL   : result[i] = (a[i] >= b[i]);
            `ROP_DEPTH_FUNC_ALWAYS   : result[i] = 1;
            default                  : result[i] = 'x;
            endcase
        end
    end

endmodule