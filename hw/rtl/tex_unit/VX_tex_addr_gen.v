`include "VX_tex_define.vh"

module VX_tex_addr_gen #(
    parameter CORE_ID = 0,
    parameter REQ_TAG_WIDTH = 1
) (
    input wire  clk,
    input wire  reset,

    // handshake
    
    input wire  valid_in,
    output wire ready_in,

    // inputs

    input wire [`NUM_THREADS-1:0]       req_tmask,
    input wire [REQ_TAG_WIDTH-1:0]      req_tag,

    input wire [`TEX_FILTER_BITS-1:0]   filter,
    input wire [`TEX_WRAP_BITS-1:0]     wrap_u,
    input wire [`TEX_WRAP_BITS-1:0]     wrap_v,

    input wire [`TEX_ADDR_BITS-1:0]     base_addr,
    input wire [`TEX_STRIDE_BITS-1:0]   log2_stride,
    input wire [`TEX_WIDTH_BITS-1:0]    log2_width,
    input wire [`TEX_HEIGHT_BITS-1:0]   log2_height,
    
    input wire [`NUM_THREADS-1:0][31:0] coord_u,
    input wire [`NUM_THREADS-1:0][31:0] coord_v,
    input wire [`NUM_THREADS-1:0][31:0] lod,

    // outputs

    output wire mem_req_valid,  
    output wire [`NUM_THREADS-1:0] mem_req_tmask,
    output wire [`TEX_FILTER_BITS-1:0] mem_req_filter,
    output wire [`NUM_THREADS-1:0][`FIXED_FRAC-1:0] mem_req_u,
    output wire [`NUM_THREADS-1:0][`FIXED_FRAC-1:0] mem_req_v,
    output wire [REQ_TAG_WIDTH-1:0] mem_req_tag,  
    output wire [`NUM_THREADS-1:0][3:0][31:0] mem_req_addr,
    input wire mem_req_ready
);

    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (lod)

    wire [`NUM_THREADS-1:0][1:0][`FIXED_FRAC-1:0] u;
    wire [`NUM_THREADS-1:0][1:0][`FIXED_FRAC-1:0] v;

    // addressing mode

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin

        wire [31:0] fu[1:0];
        wire [31:0] fv[1:0];

        assign fu[0] = coord_u[i] - (filter ? (`FIXED_HALF >> log2_width) : 0);        
        assign fv[0] = coord_v[i] - (filter ? (`FIXED_HALF >> log2_height) : 0);
        assign fu[1] = coord_u[i] + (filter ? (`FIXED_HALF >> log2_width) : 0);
        assign fv[1] = coord_v[i] + (filter ? (`FIXED_HALF >> log2_height) : 0);        

        VX_tex_wrap #(
            .CORE_ID (CORE_ID)
        ) tex_wrap_u0 (
            .wrap_i  (wrap_u),
            .coord_i (fu[0]),
            .coord_o (u[i][0])
        );

        VX_tex_wrap #(
            .CORE_ID (CORE_ID)
        ) tex_wrap_v0 (
            .wrap_i  (wrap_v),
            .coord_i (fv[0]),
            .coord_o (v[i][0])
        );

        VX_tex_wrap #(
            .CORE_ID (CORE_ID)
        ) tex_wrap_u1 (
            .wrap_i  (wrap_u),
            .coord_i (fu[1]),
            .coord_o (u[i][1])
        );

        VX_tex_wrap #(
            .CORE_ID (CORE_ID)
        ) tex_wrap_v1 (
            .wrap_i  (wrap_v),
            .coord_i (fv[1]),
            .coord_o (v[i][1])
        );
    end
    
    // addresses generation

    wire [`NUM_THREADS-1:0][3:0][31:0] addr;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin

        wire [`FIXED_FRAC-1:0] x [1:0];
        wire [`FIXED_FRAC-1:0] y [1:0];        

        assign x[0] = u[i][0] >> ((`FIXED_FRAC) - log2_width); 
        assign x[1] = u[i][1] >> ((`FIXED_FRAC) - log2_width); 
        assign y[0] = v[i][0] >> ((`FIXED_FRAC) - log2_height);         
        assign y[1] = v[i][1] >> ((`FIXED_FRAC) - log2_height); 

        assign addr[i][0] = base_addr + (32'(x[0]) + (32'(y[0]) << log2_width)) << log2_stride;
        assign addr[i][1] = base_addr + (32'(x[1]) + (32'(y[0]) << log2_width)) << log2_stride;
        assign addr[i][2] = base_addr + (32'(x[0]) + (32'(y[1]) << log2_width)) << log2_stride;
        assign addr[i][3] = base_addr + (32'(x[1]) + (32'(y[1]) << log2_width)) << log2_stride;
    end

    wire stall_out = mem_req_valid && ~mem_req_ready;

    VX_pipe_register #(
        .DATAW  (1 + `NUM_THREADS + `TEX_FILTER_BITS + REQ_TAG_WIDTH + (`NUM_THREADS * 4 * 32) + (`NUM_THREADS  * `FIXED_FRAC)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_in,      req_tmask,     filter,         req_tag,     addr,         u[0],      v[0]}),
        .data_out ({mem_req_valid, mem_req_tmask, mem_req_filter, mem_req_tag, mem_req_addr, mem_req_u, mem_req_v})
    );

    assign ready_in = ~stall_out;

endmodule