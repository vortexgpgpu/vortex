`include "VX_define.vh"

module VX_gpr_stage #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    // inputs    
    VX_writeback_if.slave   writeback_if,  
    VX_gpr_req_if.slave     gpr_req_if,

    // outputs
    VX_gpr_rsp_if.master    gpr_rsp_if
);

    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (reset)    

    localparam RAM_SIZE = `NUM_WARPS * `NUM_REGS;

    // ensure r0 never gets written, which can happen before the reset
    wire write_enable = writeback_if.valid && (writeback_if.rd != 0);
    
    wire [`NUM_THREADS-1:0] wren;
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign wren[i] = write_enable && writeback_if.tmask[i];
    end

    wire [$clog2(RAM_SIZE)-1:0] waddr, raddr1, raddr2;
    assign waddr  = {writeback_if.wid, writeback_if.rd};
    assign raddr1 = {gpr_req_if.wid, gpr_req_if.rs1};
    assign raddr2 = {gpr_req_if.wid, gpr_req_if.rs2};

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        VX_dp_ram #(
            .DATAW       (32),
            .SIZE        (RAM_SIZE),
            .INIT_ENABLE (1),
            .INIT_VALUE  (0)
        ) dp_ram1 (
            .clk   (clk),
            .wren  (wren[i]),
            .waddr (waddr),
            .wdata (writeback_if.data[i]),
            .raddr (raddr1),
            .rdata (gpr_rsp_if.rs1_data[i])
        );

        VX_dp_ram #(
            .DATAW       (32),
            .SIZE        (RAM_SIZE),
            .INIT_ENABLE (1),
            .INIT_VALUE  (0)
        ) dp_ram2 (
            .clk   (clk),
            .wren  (wren[i]),
            .waddr (waddr),
            .wdata (writeback_if.data[i]),
            .raddr (raddr2),
            .rdata (gpr_rsp_if.rs2_data[i])
        );
    end
    
`ifdef EXT_F_ENABLE
    wire [$clog2(RAM_SIZE)-1:0] raddr3;
    assign raddr3 = {gpr_req_if.wid, gpr_req_if.rs3};    

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        VX_dp_ram #(
            .DATAW       (32),
            .SIZE        (RAM_SIZE),
            .INIT_ENABLE (1),
            .INIT_VALUE  (0)
        ) dp_ram3 (
            .clk   (clk),
            .wren  (wren[i]),
            .waddr (waddr),
            .wdata (writeback_if.data[i]),
            .raddr (raddr3),
            .rdata (gpr_rsp_if.rs3_data[i])
        );
    end
`else    
    `UNUSED_VAR (gpr_req_if.rs3)    
    assign gpr_rsp_if.rs3_data = 'x;
`endif
    
    assign writeback_if.ready = 1'b1;

endmodule