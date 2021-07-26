`include "VX_define.vh"

module VX_writeback #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    // inputs
    VX_commit_if    alu_commit_if,
    VX_commit_if    ld_commit_if,  
    VX_commit_if    csr_commit_if,
    VX_commit_if    fpu_commit_if,

    // outputs
    VX_writeback_if writeback_if
);

    `UNUSED_PARAM (CORE_ID)

    localparam DATAW = `NW_BITS + 32 + `NUM_THREADS + `NR_BITS + (`NUM_THREADS * 32) + 1;

    wire wb_valid;
    wire [`NW_BITS-1:0] wb_wid;
    wire [31:0] wb_PC;
    wire [`NUM_THREADS-1:0] wb_tmask;
    wire [`NR_BITS-1:0] wb_rd;
    wire [`NUM_THREADS-1:0][31:0] wb_data;
    wire wb_eop;

    wire [3:0][DATAW-1:0] rsp_data;
    wire [3:0] rsp_ready;
    wire stall;
    
    wire ld_valid  =  ld_commit_if.valid && ld_commit_if.wb;    
    wire fpu_valid = fpu_commit_if.valid && fpu_commit_if.wb;
    wire csr_valid = csr_commit_if.valid && csr_commit_if.wb;
    wire alu_valid = alu_commit_if.valid && alu_commit_if.wb;

    assign rsp_data[0] = { ld_commit_if.wid,  ld_commit_if.PC,  ld_commit_if.tmask,  ld_commit_if.rd,  ld_commit_if.data,  ld_commit_if.eop};
    assign rsp_data[1] = {fpu_commit_if.wid, fpu_commit_if.PC, fpu_commit_if.tmask, fpu_commit_if.rd, fpu_commit_if.data, fpu_commit_if.eop};
    assign rsp_data[2] = {csr_commit_if.wid, csr_commit_if.PC, csr_commit_if.tmask, csr_commit_if.rd, csr_commit_if.data, csr_commit_if.eop};
    assign rsp_data[3] = {alu_commit_if.wid, alu_commit_if.PC, alu_commit_if.tmask, alu_commit_if.rd, alu_commit_if.data, alu_commit_if.eop};
    
    VX_stream_arbiter #(            
        .NUM_REQS (4),
        .DATAW    (DATAW),        
        .TYPE     ("X")
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({alu_valid, csr_valid, fpu_valid, ld_valid}),
        .data_in   (rsp_data),
        .ready_in  (rsp_ready),
        .valid_out (wb_valid),
        .data_out  ({wb_wid, wb_PC, wb_tmask, wb_rd, wb_data, wb_eop}),
        .ready_out (~stall)
    );

    assign  ld_commit_if.ready = rsp_ready[0] || ~ld_commit_if.wb;
    assign fpu_commit_if.ready = rsp_ready[1] || ~fpu_commit_if.wb;
    assign csr_commit_if.ready = rsp_ready[2] || ~csr_commit_if.wb;
    assign alu_commit_if.ready = rsp_ready[3] || ~alu_commit_if.wb;

    assign stall = ~writeback_if.ready && writeback_if.valid;
    
    VX_pipe_register #(
        .DATAW  (1 + DATAW),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({wb_valid,           wb_wid,           wb_PC,           wb_tmask,           wb_rd,           wb_data,           wb_eop}),
        .data_out ({writeback_if.valid, writeback_if.wid, writeback_if.PC, writeback_if.tmask, writeback_if.rd, writeback_if.data, writeback_if.eop})
    );
    
    // special workaround to get RISC-V tests Pass/Fail status
    reg [31:0] last_wb_value [`NUM_REGS-1:0] /* verilator public */;
    always @(posedge clk) begin
        if (writeback_if.valid && writeback_if.ready) begin
            last_wb_value[writeback_if.rd] <= writeback_if.data[0];
        end
    end

endmodule