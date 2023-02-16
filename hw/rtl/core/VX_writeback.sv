`include "VX_define.vh"

module VX_writeback #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    // inputs
    VX_commit_if.slave  alu_commit_if,
    VX_commit_if.slave  ld_commit_if,  
    VX_commit_if.slave  csr_commit_if,
`ifdef EXT_F_ENABLE
    VX_commit_if.slave  fpu_commit_if,
`endif    
    VX_commit_if.slave  gpu_commit_if,

    // outputs
    VX_writeback_if.master writeback_if,

    // simulation helper signals
    output wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value
);
    `UNUSED_PARAM (CORE_ID)

    localparam NW_WIDTH = `UP(`NW_BITS);
    localparam DATAW    = NW_WIDTH + 32 + `NUM_THREADS + `NR_BITS + (`NUM_THREADS * `XLEN) + 1;
    localparam NUM_RSPS = 4 + `EXT_F_ENABLED;

`ifdef EXT_F_ENABLE
    wire wb_fpu_ready_in;
`endif
    wire wb_gpu_ready_in;
    wire wb_csr_ready_in;
    wire wb_alu_ready_in;
    wire wb_ld_ready_in;

    VX_stream_arb #(
        .NUM_INPUTS (NUM_RSPS),
        .DATAW      (DATAW),
        .ARBITER    ("R"),
        .BUFFERED   (2) // ensure registered output for GPRs addressing     
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({            
        `ifdef EXT_F_ENABLE
            fpu_commit_if.valid && fpu_commit_if.wb,
        `endif
            gpu_commit_if.valid && gpu_commit_if.wb,
            csr_commit_if.valid && csr_commit_if.wb,
            alu_commit_if.valid && alu_commit_if.wb,
            ld_commit_if.valid  && ld_commit_if.wb
        }),
        .ready_in  ({
        `ifdef EXT_F_ENABLE
            wb_fpu_ready_in,
        `endif 
            wb_gpu_ready_in,
            wb_csr_ready_in,
            wb_alu_ready_in,
            wb_ld_ready_in
        }),
        .data_in   ({
        `ifdef EXT_F_ENABLE
            {fpu_commit_if.wid, fpu_commit_if.PC, fpu_commit_if.tmask, fpu_commit_if.rd, fpu_commit_if.data, fpu_commit_if.eop},
        `endif     
            {gpu_commit_if.wid, gpu_commit_if.PC, gpu_commit_if.tmask, gpu_commit_if.rd, gpu_commit_if.data, gpu_commit_if.eop},
            {csr_commit_if.wid, csr_commit_if.PC, csr_commit_if.tmask, csr_commit_if.rd, csr_commit_if.data, csr_commit_if.eop},
            {alu_commit_if.wid, alu_commit_if.PC, alu_commit_if.tmask, alu_commit_if.rd, alu_commit_if.data, alu_commit_if.eop},
            {ld_commit_if.wid,  ld_commit_if.PC,  ld_commit_if.tmask,  ld_commit_if.rd,  ld_commit_if.data,  ld_commit_if.eop}
        }),
        .data_out  ({writeback_if.wid, writeback_if.PC, writeback_if.tmask, writeback_if.rd, writeback_if.data, writeback_if.eop}),
        .valid_out (writeback_if.valid),        
        .ready_out (writeback_if.ready)
    );

`ifdef EXT_F_ENABLE
    assign fpu_commit_if.ready = wb_fpu_ready_in || ~fpu_commit_if.wb;
`endif
    assign gpu_commit_if.ready = wb_gpu_ready_in || ~gpu_commit_if.wb;
    assign csr_commit_if.ready = wb_csr_ready_in || ~csr_commit_if.wb;
    assign alu_commit_if.ready = wb_alu_ready_in || ~alu_commit_if.wb;
    assign ld_commit_if.ready  = wb_ld_ready_in  || ~ld_commit_if.wb;
    
    // simulation helper signal to get RISC-V tests Pass/Fail status
    reg [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value_r;
    always @(posedge clk) begin
        if (writeback_if.valid && writeback_if.ready) begin
            sim_wb_value_r[writeback_if.rd] <= writeback_if.data[0];
        end
    end
    assign sim_wb_value = sim_wb_value_r;

endmodule
