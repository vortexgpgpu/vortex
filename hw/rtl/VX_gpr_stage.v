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

    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (reset)    

    // ensure r0 never gets written, which can happen before the reset
    wire write_enable = writeback_if.valid && (writeback_if.rd != 0);
    
`ifdef EXT_F_ENABLE
    localparam RAM_SIZE = `NUM_WARPS * `NUM_REGS;
    wire [`NUM_THREADS-1:0][31:0] rdata1, rdata2, rdata3;
    wire [$clog2(RAM_SIZE)-1:0] waddr, raddr1, raddr2, raddr3;
    
    assign waddr  = {writeback_if.wid, writeback_if.rd};
    assign raddr1 = {gpr_req_if.wid,   gpr_req_if.rs1};
    assign raddr2 = {gpr_req_if.wid,   gpr_req_if.rs2};
    assign raddr3 = {gpr_req_if.wid,   gpr_req_if.rs3};

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        VX_dp_ram #(
            .RD_PORTS    (3),
            .DATAW       (32),
            .SIZE        (RAM_SIZE),
            .INIT_ENABLE (1),
            .INIT_VALUE  (0)
        ) dp_ram (
            .clk   (clk),
            .wren  (write_enable && writeback_if.tmask[i]),
            .waddr (waddr),
            .wdata (writeback_if.data[i]),
            .rden  (3'b111),
            .raddr ({raddr3, raddr2, raddr1}),
            .rdata ({rdata3[i], rdata2[i], rdata1[i]})
        );      
    end
    
    assign gpr_rsp_if.rs1_data = rdata1;
    assign gpr_rsp_if.rs2_data = rdata2;
    assign gpr_rsp_if.rs3_data = rdata3;
`else
    localparam RAM_SIZE = `NUM_WARPS * `NUM_REGS;
    wire [`NUM_THREADS-1:0][31:0] rdata1, rdata2;
    wire [$clog2(RAM_SIZE)-1:0] waddr, raddr1, raddr2;
    
    assign waddr  = {writeback_if.wid, writeback_if.rd};
    assign raddr1 = {gpr_req_if.wid,   gpr_req_if.rs1};
    assign raddr2 = {gpr_req_if.wid,   gpr_req_if.rs2};  
    `UNUSED_VAR (gpr_req_if.rs3)  

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        VX_dp_ram #(
            .RD_PORTS    (2),
            .DATAW       (32),
            .SIZE        (RAM_SIZE),
            .INIT_ENABLE (1),
            .INIT_VALUE  (0)
        ) dp_ram (
            .clk   (clk),
            .wren  (write_enable && writeback_if.tmask[i]),
            .waddr (waddr),
            .wdata (writeback_if.data[i]),
            .rden  (2'b11),
            .raddr ({raddr2, raddr1}),
            .rdata ({rdata2[i], rdata1[i]})
        );
    end

    assign gpr_rsp_if.rs1_data = rdata1;
    assign gpr_rsp_if.rs2_data = rdata2;
    assign gpr_rsp_if.rs3_data = 0;
`endif
    
    assign writeback_if.ready = 1'b1;

endmodule