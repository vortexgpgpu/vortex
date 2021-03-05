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
    
    wire ld_valid  = ld_commit_if.valid && ld_commit_if.wb;    
    wire fpu_valid = fpu_commit_if.valid && fpu_commit_if.wb;
    wire csr_valid = csr_commit_if.valid && csr_commit_if.wb;
    wire alu_valid = alu_commit_if.valid && alu_commit_if.wb;

    wire wb_valid;
    wire [`NW_BITS-1:0] wb_wid;
    wire [31:0] wb_PC;
    wire [`NUM_THREADS-1:0] wb_tmask;
    wire [`NR_BITS-1:0] wb_rd;
    wire [`NUM_THREADS-1:0][31:0] wb_data;
    wire wb_eop;
    
    assign wb_valid =   ld_valid  | 
                        fpu_valid | 
                        csr_valid | 
                        alu_valid;

    assign wb_wid =     ld_valid  ? ld_commit_if.wid :
                        fpu_valid ? fpu_commit_if.wid :
                        csr_valid ? csr_commit_if.wid :                        
                        /*alu_valid ?*/ alu_commit_if.wid;

    assign wb_PC =      ld_valid  ? ld_commit_if.PC :
                        fpu_valid ? fpu_commit_if.PC :
                        csr_valid ? csr_commit_if.PC :                        
                        /*alu_valid ?*/ alu_commit_if.PC;
    
    assign wb_tmask =   ld_valid  ? ld_commit_if.tmask :
                        fpu_valid ? fpu_commit_if.tmask :
                        csr_valid ? csr_commit_if.tmask :                        
                        /*alu_valid ?*/ alu_commit_if.tmask;

    assign wb_rd =      ld_valid  ? ld_commit_if.rd :
                        fpu_valid ? fpu_commit_if.rd :
                        csr_valid ? csr_commit_if.rd :                        
                        /*alu_valid ?*/ alu_commit_if.rd;

    assign wb_data =    ld_valid  ? ld_commit_if.data :
                        fpu_valid ? fpu_commit_if.data :
                        csr_valid ? csr_commit_if.data :                        
                        /*alu_valid ?*/ alu_commit_if.data;

    assign wb_eop =     ld_valid  ? ld_commit_if.eop :
                        fpu_valid ? fpu_commit_if.eop :
                        csr_valid ? csr_commit_if.eop :                        
                        /*alu_valid ?*/ alu_commit_if.eop;

    wire stall = ~writeback_if.ready && writeback_if.valid;
    
    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + 32 + `NUM_THREADS + `NR_BITS + (`NUM_THREADS * 32) + 1),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({wb_valid,           wb_wid,           wb_PC,           wb_tmask,           wb_rd,           wb_data,           wb_eop}),
        .data_out ({writeback_if.valid, writeback_if.wid, writeback_if.PC, writeback_if.tmask, writeback_if.rd, writeback_if.data, writeback_if.eop})
    );
    
    assign ld_commit_if.ready  = !stall;    
    assign fpu_commit_if.ready = !stall && !ld_valid;       
    assign csr_commit_if.ready = !stall && !ld_valid && !fpu_valid;
    assign alu_commit_if.ready = !stall && !ld_valid && !fpu_valid && !csr_valid;  
    
    // special workaround to get RISC-V tests Pass/Fail status
    reg [31:0] last_wb_value [`NUM_REGS-1:0] /* verilator public */;
    always @(posedge clk) begin
        if (writeback_if.valid && writeback_if.ready) begin
            last_wb_value[writeback_if.rd] <= writeback_if.data[0];
        end
    end

endmodule