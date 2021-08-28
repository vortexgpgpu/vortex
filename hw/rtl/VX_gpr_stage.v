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

    localparam RAM_SIZE = `NUM_WARPS * `NUM_REGS;

    // ensure r0 never gets written, which can happen before the reset
    wire write_enable = writeback_if.valid && (writeback_if.rd != 0);
    
    wire [(`NUM_THREADS * 4)-1:0] wren;
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign wren [i * 4 +: 4] = {4{write_enable && writeback_if.tmask[i]}};
    end

    reg [`NUM_THREADS-1:0][31:0] last_wdata;
    reg [$clog2(RAM_SIZE)-1:0] last_waddr;
    reg [`NUM_THREADS-1:0] last_wmask;

    always @(posedge clk) begin     
        last_wdata <= writeback_if.data;
        last_wmask <= {`NUM_THREADS{write_enable}} & writeback_if.tmask;
        last_waddr <= waddr;
    end
    
    wire [`NUM_THREADS-1:0][31:0] rdata1, rdata2;
    wire [$clog2(RAM_SIZE)-1:0] waddr, raddr1, raddr2;

    assign waddr  = {writeback_if.wid, writeback_if.rd};
    assign raddr1 = {gpr_req_if.wid, gpr_req_if.rs1};
    assign raddr2 = {gpr_req_if.wid, gpr_req_if.rs2};

    VX_dp_ram #(
        .DATAW       (32 * `NUM_THREADS),
        .SIZE        (RAM_SIZE),
        .BYTEENW     (`NUM_THREADS * 4),
        .INIT_ENABLE (1),
        .INIT_VALUE  (0),
        .NO_RWCHECK  (1)
    ) dp_ram1 (
        .clk   (clk),
        .wren  (wren),
        .waddr (waddr),
        .wdata (writeback_if.data),
        .rden  (1'b1),
        .raddr (raddr1),
        .rdata (rdata1)
    );

    VX_dp_ram #(
        .DATAW       (32 * `NUM_THREADS),
        .SIZE        (RAM_SIZE),
        .BYTEENW     (`NUM_THREADS * 4),
        .INIT_ENABLE (1),
        .INIT_VALUE  (0),
        .NO_RWCHECK  (1)
    ) dp_ram2 (
        .clk   (clk),
        .wren  (wren),
        .waddr (waddr),
        .wdata (writeback_if.data),
        .rden  (1'b1),
        .raddr (raddr2),
        .rdata (rdata2)
    );

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign gpr_rsp_if.rs1_data[i] = (last_wmask[i] && (raddr1 == last_waddr)) ? last_wdata[i] : rdata1[i];
        assign gpr_rsp_if.rs2_data[i] = (last_wmask[i] && (raddr2 == last_waddr)) ? last_wdata[i] : rdata2[i];
    end
    
`ifdef EXT_F_ENABLE
    wire [`NUM_THREADS-1:0][31:0] rdata3;
    wire [$clog2(RAM_SIZE)-1:0] raddr3;
    assign raddr3 = {gpr_req_if.wid, gpr_req_if.rs3};    

    VX_dp_ram #(
        .DATAW       (32 * `NUM_THREADS),
        .SIZE        (RAM_SIZE),
        .BYTEENW     (`NUM_THREADS * 4),
        .INIT_ENABLE (1),
        .INIT_VALUE  (0),
        .NO_RWCHECK  (1)
    ) dp_ram3 (
        .clk   (clk),
        .wren  (wren),
        .waddr (waddr),
        .wdata (writeback_if.data),
        .rden  (1'b1),
        .raddr (raddr3),
        .rdata (rdata3)
    );
    
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign gpr_rsp_if.rs3_data[i] = (last_wmask[i] && (raddr3 == last_waddr)) ? last_wdata[i] : rdata3[i];
    end
`else    
    `UNUSED_VAR (gpr_req_if.rs3)    
    assign gpr_rsp_if.rs3_data = 'x;
`endif
    
    assign writeback_if.ready = 1'b1;

endmodule