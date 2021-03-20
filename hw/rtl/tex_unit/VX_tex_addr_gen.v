`include "VX_define.vh"

module VX_tex_addr_gen #(
    parameter CORE_ID   = 0,
    parameter REQ_TAG_WIDTH = 1,
    parameter FRAC_BITS = 20,
    parameter INT_BITS  = 32 - FRAC_BITS
) (
    input wire  clk,
    input wire  reset,

    // handshake
    
    input wire  valid_in,
    output wire ready_in,

    // inputs

    output wire [REQ_TAG_WIDTH-1:0]   req_tag,
    input wire [`TEX_FILTER_BITS-1:0] filter,
    input wire [`TEX_WRAP_BITS-1:0]   wrap_u,
    input wire [`TEX_WRAP_BITS-1:0]   wrap_v,

    input wire [`TEX_ADDR_BITS-1:0]   base_addr,
    input wire [1:0]                  log2_stride,
    input wire [`TEX_WIDTH_BITS-1:0]  log2_width,
    input wire [`TEX_HEIGHT_BITS-1:0] log2_height,
    input wire [3:0]                  lod,

    input wire [31:0] coord_u,
    input wire [31:0] coord_v,

    // outputs

    output wire [3:0] mem_req_valid,  
    output wire [REQ_TAG_WIDTH-1:0] mem_req_tag,  
    output wire [3:0][31:0] mem_req_addr,
    input wire mem_req_ready
);

    `UNUSED_VAR (filter)
    `UNUSED_VAR (lod)

    wire [31:0]  u, y;
    wire [31:0]  x_offset, y_offset;
    wire [31:0]  addr0;

    // addressing mode    

    assign x_offset = u >> (5'(FRAC_BITS) - log2_width); 
    assign y_offset = v >> (5'(FRAC_BITS) - log2_height); 
    assign addr0 = base_addr + (x_offset + (y_offset << log2_width)) << log2_stride;

    wire [3:0]       req_valids  = 4'(valid_in);
    wire [3:0][31:0] req_address = {4{addr0}};

    VX_pipe_register #(
        .DATAW  (1 + 4 + 4 * 32 + REQ_TAG_WIDTH),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({req_valids,    req_address,  req_tag}),
        .data_out ({mem_req_valid, mem_req_addr, mem_req_tag})
    );

    assign ready_in = ~stall_out;

endmodule