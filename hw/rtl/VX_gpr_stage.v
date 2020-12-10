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
    
`ifdef EXT_F_ENABLE
    localparam RAM_DEPTH = `NUM_WARPS * (`NUM_REGS / 2);
    wire [`NUM_THREADS-1:0][31:0] rdata1_i, rdata2_i, rdata1_f, rdata2_f, rdata3_f;
    wire [$clog2(RAM_DEPTH)-1:0] waddr, raddr1, raddr2, raddr3;

    wire waddr_is_fp  = writeback_if.rd[`NR_BITS-1];
    wire raddr1_is_fp = gpr_req_if.rs1[`NR_BITS-1];
    wire raddr2_is_fp = gpr_req_if.rs2[`NR_BITS-1];
    wire raddr3_is_fp = gpr_req_if.rs3[`NR_BITS-1];
    `UNUSED_VAR (raddr3_is_fp)
    
    assign waddr  = {writeback_if.wid, writeback_if.rd[`NR_BITS-2:0]};
    assign raddr1 = {gpr_req_if.wid,   gpr_req_if.rs1[`NR_BITS-2:0]};
    assign raddr2 = {gpr_req_if.wid,   gpr_req_if.rs2[`NR_BITS-2:0]};
    assign raddr3 = {gpr_req_if.wid,   gpr_req_if.rs3[`NR_BITS-2:0]};

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        VX_gpr_ram_i #(
            .DATAW (32),
            .DEPTH (RAM_DEPTH)
        ) gpr_ram_i (
            .clk    (clk),
            .wren   (writeback_if.valid && writeback_if.tmask[i] && !waddr_is_fp),
            .waddr  (waddr),
            .wdata  (writeback_if.data[i]),
            .raddr1 (raddr1),
            .raddr2 (raddr2),
            .rdata1 (rdata1_i[i]),
            .rdata2 (rdata2_i[i])
        );
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        VX_gpr_ram_f #(
            .DATAW (32),
            .DEPTH (RAM_DEPTH)
        ) gpr_ram_f (
            .clk    (clk),
            .wren   (writeback_if.valid && writeback_if.tmask[i] && waddr_is_fp),
            .waddr  (waddr),
            .wdata  (writeback_if.data[i]),
            .raddr1 (raddr1),
            .raddr2 (raddr2),
            .raddr3 (raddr3),
            .rdata1 (rdata1_f[i]),
            .rdata2 (rdata2_f[i]),
            .rdata3 (rdata3_f[i])
        );      
    end
    
    assign gpr_rsp_if.rs1_data = raddr1_is_fp ? rdata1_f : rdata1_i;
    assign gpr_rsp_if.rs2_data = raddr2_is_fp ? rdata2_f : rdata2_i;
    assign gpr_rsp_if.rs3_data = rdata3_f;
`else
    localparam RAM_DEPTH = `NUM_WARPS * `NUM_REGS;
    wire [`NUM_THREADS-1:0][31:0] rdata1_i, rdata2_i;
    wire [$clog2(RAM_DEPTH)-1:0] waddr, raddr1, raddr2;
    
    assign waddr  = {writeback_if.wid, writeback_if.rd};
    assign raddr1 = {gpr_req_if.wid, gpr_req_if.rs1};
    assign raddr2 = {gpr_req_if.wid, gpr_req_if.rs2};  
    `UNUSED_VAR (gpr_req_if.rs3)  

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        VX_gpr_ram_i #(
            .DATAW (32),
            .DEPTH (RAM_DEPTH)
        ) gpr_ram_i (
            .clk    (clk),
            .wren   (writeback_if.valid && writeback_if.tmask[i]),
            .waddr  (waddr),
            .wdata  (writeback_if.data[i]),
            .raddr1 (raddr1),
            .raddr2 (raddr2),
            .rdata1 (rdata1_i[i]),
            .rdata2 (rdata2_i[i])
        );
    end

    assign gpr_rsp_if.rs1_data = rdata1_i;
    assign gpr_rsp_if.rs2_data = rdata2_i;
    assign gpr_rsp_if.rs3_data = 0;
`endif
    
    assign writeback_if.ready = 1'b1;

endmodule