`include "VX_define.vh"

module VX_tex_sampler #(
    parameter CORE_ID = 0    
) (
    input wire clk,
    input wire reset,

    // inputs
    input wire                          req_valid,
    input wire [`NW_BITS-1:0]           req_wid,
    input wire [`NUM_THREADS-1:0]       req_tmask,
    input wire [31:0]                   req_PC,
    input wire [`NR_BITS-1:0]           req_rd,   
    input wire                          req_wb,
    input wire [`TEX_FILTER_BITS-1:0]   req_filter,
    input wire [`TEX_FORMAT_BITS-1:0]   req_format,
    input wire [`NUM_THREADS-1:0][3:0][31:0] req_texels,
    output wire                         req_ready,

    // ouputs
    output wire                          rsp_valid,
    output wire [`NW_BITS-1:0]           rsp_wid,
    output wire [`NUM_THREADS-1:0]       rsp_tmask,
    output wire [31:0]                   rsp_PC,
    output wire [`NR_BITS-1:0]           rsp_rd,   
    output wire                          rsp_wb,
    output wire [`NUM_THREADS-1:0][31:0] rsp_data,
    input wire                           rsp_ready
);
    
    `UNUSED_PARAM (CORE_ID)
    
    /*
    assign tex_req_if.ready = (& pt_addr_ready);

    assign lsu_req_if.valid = (& pt_addr_valid);

    assign lsu_req_if.wid   = tex_req_if.wid;
    assign lsu_req_if.tmask = tex_req_if.tmask;
    assign lsu_req_if.PC    = tex_req_if.PC;
    assign lsu_req_if.rd    = tex_req_if.rd;
    assign lsu_req_if.wb    = tex_req_if.wb;
    assign lsu_req_if.offset = 32'h0000;
    assign lsu_req_if.op_type = `OP_BITS'({1'b0, 3'b000}); //func3 for word load??
    assign lsu_req_if.store_data = {`NUM_THREADS{32'h0000}};

    // wait buffer for fragments  / replace with cache/state fragment fifo for bilerp
    // no filtering for point sampling -> directly from dcache to output response

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({rsp_valid,        rsp_wid,        rsp_tmask,        rsp_PC,        rsp_rd,        rsp_wb,        rsp_data}),
        .data_out ({tex_rsp_if.valid, tex_rsp_if.wid, tex_rsp_if.tmask, tex_rsp_if.PC, tex_rsp_if.rd, tex_rsp_if.wb, tex_rsp_if.data})
    );

    // output
    assign stall_out = ~tex_rsp_if.ready && tex_rsp_if.valid;

    // can accept new request?
    assign stall_in  = stall_out;

    assign ld_commit_if.ready = ~stall_in;*/

endmodule