`include "VX_define.vh"

module VX_mufu_unit_top import VX_gpu_pkg::*; (
    input  wire clk,
    input  wire reset,

    input  wire [`VX_CFG_ISSUE_WIDTH-1:0]                   dispatch_valid,
    output wire [`VX_CFG_ISSUE_WIDTH-1:0]                   dispatch_ready,
    input  wire [`VX_CFG_ISSUE_WIDTH-1:0][$bits(dispatch_t)-1:0] dispatch_data,

    output wire [`VX_CFG_ISSUE_WIDTH-1:0]                   commit_valid,
    input  wire [`VX_CFG_ISSUE_WIDTH-1:0]                   commit_ready,
    output wire [`VX_CFG_ISSUE_WIDTH-1:0][$bits(commit_t)-1:0]   commit_data
);
    VX_dispatch_if dispatch_if[`VX_CFG_ISSUE_WIDTH]();
    VX_commit_if   commit_if  [`VX_CFG_ISSUE_WIDTH]();

    for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_ports
        assign dispatch_if[i].valid = dispatch_valid[i];
        assign dispatch_if[i].data  = dispatch_data[i];
        assign dispatch_ready[i]    = dispatch_if[i].ready;

        assign commit_valid[i]      = commit_if[i].valid;
        assign commit_data[i]       = commit_if[i].data;
        assign commit_if[i].ready   = commit_ready[i];
    end

    VX_mufu_unit #(
        .INSTANCE_ID ("mufu_unit_top")
    ) mufu_unit (
        .clk         (clk),
        .reset       (reset),
        .dispatch_if (dispatch_if),
        .commit_if   (commit_if)
    );

endmodule
