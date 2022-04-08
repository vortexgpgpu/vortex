`include "VX_platform.vh"

module VX_tex_sat #(
    parameter IN_W  = 1,
    parameter OUT_W = 1,
    parameter MODEL = 1
) (
    input wire [IN_W-1:0] data_in,   
    output wire [OUT_W-1:0] data_out
); 
    `STATIC_ASSERT(((OUT_W+1) < IN_W), ("invalid parameter"))

    if (MODEL == 1) begin
        wire [OUT_W-1:0] underflow_mask = {OUT_W{~data_in[IN_W-1]}};        
        wire [OUT_W-1:0] overflow_mask = {OUT_W{(| data_in[IN_W-2:OUT_W])}};
        assign data_out = (data_in[OUT_W-1:0] | overflow_mask) & underflow_mask;        
    end else begin
        assign data_out = data_in[IN_W-1] ? OUT_W'(0) : ((data_in > {OUT_W{1'b1}}) ? {OUT_W{1'b1}} : OUT_W'(data_in));
    end

endmodule
