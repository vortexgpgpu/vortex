`include "VX_tex_define.vh"

module VX_tex_lerp #(
) (
    input wire [`BLEND_FRAC-1:0] blend, 
    input wire  [31:0] in1,
    input wire  [31:0] in2,
    output wire [31:0] out
);  
    wire [63:0] in1_w, in2_w;
    wire [63:0] lerp1, lerp2;

    `UNUSED_VAR (lerp1)
    `UNUSED_VAR (lerp2)

    assign in1_w[15:00] = {8'h00, in1[07:00]};
    assign in1_w[31:16] = {8'h00, in1[15:08]};
    assign in1_w[47:32] = {8'h00, in1[23:16]};
    assign in1_w[63:48] = {8'h00, in1[31:24]};

    assign in2_w[15:00] = {8'h00, in2[07:00]};
    assign in2_w[31:16] = {8'h00, in2[15:08]};
    assign in2_w[47:32] = {8'h00, in2[23:16]};
    assign in2_w[63:48] = {8'h00, in2[31:24]};

    assign lerp1 = (in2_w - in1_w) * blend;
    
    assign lerp2 = in1_w + {8'h00,lerp1[63:56], 8'h00,lerp1[47:40], 8'h00,lerp1[31:24], 8'h00,lerp1[15:8]};
    
    assign out[07:00] = lerp2[07:00];
    assign out[15:08] = lerp2[23:16];
    assign out[23:16] = lerp2[39:32];
    assign out[31:24] = lerp2[55:48];

endmodule