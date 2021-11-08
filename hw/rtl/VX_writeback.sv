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
    VX_writeback_if.master writeback_if
);

    `UNUSED_PARAM (CORE_ID)

    localparam DATAW = `NW_BITS + 32 + `NUM_THREADS + `NR_BITS + (`NUM_THREADS * 32) + 1;
`ifdef EXT_F_ENABLE
`ifdef EXT_TEX_ENABLE
    localparam NUM_RSPS = 5;
`else
    localparam NUM_RSPS = 4;
`endif
`else
`ifdef EXT_TEX_ENABLE
    localparam NUM_RSPS = 4;
`else
    localparam NUM_RSPS = 3;
`endif
`endif

    wire wb_valid;
    wire [`NW_BITS-1:0] wb_wid;
    wire [31:0] wb_PC;
    wire [`NUM_THREADS-1:0] wb_tmask;
    wire [`NR_BITS-1:0] wb_rd;
    wire [`NUM_THREADS-1:0][31:0] wb_data;
    wire wb_eop;

    wire [NUM_RSPS-1:0] rsp_valid;
    wire [NUM_RSPS-1:0][DATAW-1:0] rsp_data;
    wire [NUM_RSPS-1:0] rsp_ready;
    wire stall;

    assign rsp_valid = {            
    `ifdef EXT_TEX_ENABLE
        gpu_commit_if.valid && gpu_commit_if.wb,
    `endif
        csr_commit_if.valid && csr_commit_if.wb,
        alu_commit_if.valid && alu_commit_if.wb,        
    `ifdef EXT_F_ENABLE
        fpu_commit_if.valid && fpu_commit_if.wb,
    `endif
        ld_commit_if.valid  && ld_commit_if.wb
    };

    assign rsp_data = {                                       
    `ifdef EXT_TEX_ENABLE
        {gpu_commit_if.wid, gpu_commit_if.PC, gpu_commit_if.tmask, gpu_commit_if.rd, gpu_commit_if.data, gpu_commit_if.eop},
    `endif         
        {csr_commit_if.wid, csr_commit_if.PC, csr_commit_if.tmask, csr_commit_if.rd, csr_commit_if.data, csr_commit_if.eop},
        {alu_commit_if.wid, alu_commit_if.PC, alu_commit_if.tmask, alu_commit_if.rd, alu_commit_if.data, alu_commit_if.eop},
    `ifdef EXT_F_ENABLE
        {fpu_commit_if.wid, fpu_commit_if.PC, fpu_commit_if.tmask, fpu_commit_if.rd, fpu_commit_if.data, fpu_commit_if.eop},
    `endif
        { ld_commit_if.wid, ld_commit_if.PC,  ld_commit_if.tmask,  ld_commit_if.rd,  ld_commit_if.data,  ld_commit_if.eop}
    };
    
    VX_stream_arbiter #(            
        .NUM_REQS (NUM_RSPS),
        .DATAW    (DATAW),
        .TYPE     ("P")
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_valid),
        .data_in   (rsp_data),
        .ready_in  (rsp_ready),
        .valid_out (wb_valid),
        .data_out  ({wb_wid, wb_PC, wb_tmask, wb_rd, wb_data, wb_eop}),
        .ready_out (~stall)
    );

    assign  ld_commit_if.ready = rsp_ready[0] || ~ld_commit_if.wb;
`ifdef EXT_F_ENABLE
    assign fpu_commit_if.ready = rsp_ready[1] || ~fpu_commit_if.wb;
    assign alu_commit_if.ready = rsp_ready[2] || ~alu_commit_if.wb;
    assign csr_commit_if.ready = rsp_ready[3] || ~csr_commit_if.wb;
`else
    assign alu_commit_if.ready = rsp_ready[1] || ~alu_commit_if.wb;
    assign csr_commit_if.ready = rsp_ready[2] || ~csr_commit_if.wb;
`ifdef EXT_TEX_ENABLE
    assign gpu_commit_if.ready = rsp_ready[3] || ~gpu_commit_if.wb;
`endif
`endif    

`ifdef EXT_TEX_ENABLE
`ifdef EXT_F_ENABLE
    assign gpu_commit_if.ready = rsp_ready[4] || ~gpu_commit_if.wb;
`else
    assign gpu_commit_if.ready = rsp_ready[3] || ~gpu_commit_if.wb;
`endif
`else
    assign gpu_commit_if.ready = 1;
`endif
    
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