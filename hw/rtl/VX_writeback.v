`include "VX_define.vh"

module VX_writeback #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    // inputs
    VX_commit_if    alu_commit_if,
    VX_commit_if    lsu_commit_if,  
    VX_commit_if    mul_commit_if,
    VX_commit_if    fpu_commit_if,    
    VX_commit_if    csr_commit_if,

    // outputs
    VX_wb_if        writeback_if
);

    wire alu_valid = (| alu_commit_if.valid) && alu_commit_if.wb;
    wire lsu_valid = (| lsu_commit_if.valid) && lsu_commit_if.wb;
    wire csr_valid = (| csr_commit_if.valid) && csr_commit_if.wb;
    wire mul_valid = (| mul_commit_if.valid) && mul_commit_if.wb;
    wire fpu_valid = (| fpu_commit_if.valid) && fpu_commit_if.wb;

    VX_wb_if writeback_tmp_if();    

    assign writeback_tmp_if.valid = lsu_valid ? lsu_commit_if.valid :
                                    mul_valid ? mul_commit_if.valid :             
                                    alu_valid ? alu_commit_if.valid :                            
                                    csr_valid ? csr_commit_if.valid :                                                 
                                                0;     

    assign writeback_tmp_if.warp_num = lsu_valid ? lsu_commit_if.warp_num :
                                    mul_valid ? mul_commit_if.warp_num :   
                                    alu_valid ? alu_commit_if.warp_num :                            
                                    csr_valid ? csr_commit_if.warp_num :  
                                                0; 

    assign writeback_tmp_if.data =  lsu_valid ? lsu_commit_if.data :
                                    mul_valid ? mul_commit_if.data :                           
                                    alu_valid ? alu_commit_if.data :                            
                                    csr_valid ? csr_commit_if.data :                                                               
                                                0;

    assign writeback_tmp_if.rd =    lsu_valid ? lsu_commit_if.rd :
                                    mul_valid ? mul_commit_if.rd :                           
                                    alu_valid ? alu_commit_if.rd :                            
                                    csr_valid ? csr_commit_if.rd :                                                               
                                                0;

    assign writeback_tmp_if.is_fp = fpu_valid && fpu_commit_if.ready;

    wire stall = ~writeback_if.ready && (| writeback_if.valid);

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + `NR_BITS + (`NUM_THREADS * 32) + 1)
    ) wb_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({writeback_tmp_if.valid, writeback_tmp_if.warp_num, writeback_tmp_if.rd, writeback_tmp_if.data, writeback_tmp_if.is_fp}),
        .out   ({writeback_if.valid,     writeback_if.warp_num,     writeback_if.rd,     writeback_if.data,     writeback_if.is_fp})
    );

    assign lsu_commit_if.ready = !stall;    
    assign fpu_commit_if.ready = !stall && !lsu_valid;   
    assign mul_commit_if.ready = !stall && !lsu_valid && !fpu_valid;
    assign alu_commit_if.ready = !stall && !lsu_valid && !fpu_valid && !mul_valid;    
    assign csr_commit_if.ready = !stall && !lsu_valid && !fpu_valid && !mul_valid && !alu_valid;    
    
    // special workaround to control RISC-V benchmarks termination on Verilator
    reg [31:0] last_data_wb /* verilator public */;
    always @(posedge clk) begin
        if ((| writeback_tmp_if.valid) && ~stall && (writeback_tmp_if.rd == 28)) begin
            last_data_wb <= writeback_tmp_if.data[0];
        end
    end

endmodule