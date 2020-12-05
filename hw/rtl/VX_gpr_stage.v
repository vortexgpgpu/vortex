`include "VX_define.vh"

module VX_gpr_stage #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    // inputs    
    VX_writeback_if writeback_if,  
    VX_gpr_req_if   gpr_req_if,

    // outputs
    VX_gpr_rsp_if   gpr_rsp_if
);
    `UNUSED_VAR (reset)

    wire [`NUM_THREADS-1:0][31:0] rdata1, rdata2, rdata3;
    wire [`NW_BITS+`NR_BITS-1:0] waddr, raddr1, raddr2, raddr3;

`ifdef EXT_F_ENABLE
    assign waddr  = {writeback_if.rd[`NR_BITS-1], writeback_if.wid, writeback_if.rd[`NR_BITS-2:0]};
    assign raddr1 = {gpr_req_if.rs1[`NR_BITS-1],  gpr_req_if.wid,   gpr_req_if.rs1[`NR_BITS-2:0]};
    assign raddr2 = {gpr_req_if.rs2[`NR_BITS-1],  gpr_req_if.wid,   gpr_req_if.rs2[`NR_BITS-2:0]};
    assign raddr3 = {gpr_req_if.rs3[`NR_BITS-1],  gpr_req_if.wid,   gpr_req_if.rs3[`NR_BITS-2:0]};
`else
    assign waddr  = {writeback_if.wid, writeback_if.rd};
    assign raddr1 = {gpr_req_if.wid, gpr_req_if.rs1};
    assign raddr2 = {gpr_req_if.wid, gpr_req_if.rs2};
    assign raddr3 = {gpr_req_if.wid, gpr_req_if.rs3};
`endif

    VX_gpr_ram gpr_ram (
        .clk    (clk),
        .wren   (writeback_if.valid),
        .tmask  (writeback_if.tmask),              
        .waddr  (waddr),
        .wdata  (writeback_if.data),
        .raddr1 (raddr1),
        .raddr2 (raddr2),
        .raddr3 (raddr3),
        .rdata1 (rdata1),
        .rdata2 (rdata2),
        .rdata3 (rdata3)
    );  
    
    assign gpr_rsp_if.rs1_data = rdata1;
    assign gpr_rsp_if.rs2_data = rdata2;
    assign gpr_rsp_if.rs3_data = rdata3;

    assign writeback_if.ready = 1'b1;

endmodule