`include "VX_platform.vh"
`include "VX_define.vh"

module VX_tex_pt_addr #(
    parameter FRAC_BITS = 20,
    parameter INT_BITS = 32 - FRAC_BITS
) (
    input wire  clk,
    input wire  reset,

    input wire  valid_in,
    output wire ready_out,

    input wire [`CSR_WIDTH-1:0] tex_addr,
    input wire [`CSR_WIDTH-1:0] tex_width,
    input wire [`CSR_WIDTH-1:0] tex_height,

    input wire [31:0]     tex_u,
    input wire [31:0]     tex_v,

    output wire [31:0]    pt_addr,

    output wire           valid_out,
    input wire            ready_in
);

  `UNUSED_VAR (clk)
  `UNUSED_VAR (reset)

  reg [31:0]  x_offset;
  reg [31:0]  y_offset;

  assign  x_offset = tex_u >> (32'(FRAC_BITS) - tex_width); 
  assign  y_offset = tex_v >> (32'(FRAC_BITS) - tex_height); 
  assign  pt_addr = (tex_addr << (32 - `CSR_WIDTH)) + x_offset + (y_offset << tex_width);  

  assign valid_out = valid_in;
  assign ready_out = ready_in;

endmodule