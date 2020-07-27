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
    VX_commit_is_if commit_is_if,

    // outputs
    VX_wb_if        writeback_if
);

    wire alu_valid = alu_commit_if.valid && commit_is_if.alu_data.wb;
    wire lsu_valid = lsu_commit_if.valid && commit_is_if.lsu_data.wb;
    wire csr_valid = csr_commit_if.valid && commit_is_if.csr_data.wb;
    wire mul_valid = mul_commit_if.valid && commit_is_if.mul_data.wb;
    wire fpu_valid = fpu_commit_if.valid && commit_is_if.fpu_data.wb;

    VX_wb_if writeback_tmp_if();    

    assign writeback_tmp_if.valid = alu_valid ? alu_commit_if.valid :
                                    lsu_valid ? lsu_commit_if.valid :
                                    csr_valid ? csr_commit_if.valid :             
                                    mul_valid ? mul_commit_if.valid :                            
                                    fpu_valid ? fpu_commit_if.valid :                                                 
                                                0;     

    assign writeback_tmp_if.warp_num = alu_valid ? commit_is_if.alu_data.warp_num :
                                    lsu_valid ? commit_is_if.lsu_data.warp_num :   
                                    csr_valid ? commit_is_if.csr_data.warp_num :   
                                    mul_valid ? commit_is_if.mul_data.warp_num :                            
                                    fpu_valid ? commit_is_if.fpu_data.warp_num :  
                                                0;

    assign writeback_tmp_if.curr_PC = alu_valid ? commit_is_if.alu_data.curr_PC :
                                    lsu_valid ? commit_is_if.lsu_data.curr_PC :   
                                    csr_valid ? commit_is_if.csr_data.curr_PC :   
                                    mul_valid ? commit_is_if.mul_data.curr_PC :                            
                                    fpu_valid ? commit_is_if.fpu_data.curr_PC :  
                                                0;
    
    assign writeback_tmp_if.thread_mask = alu_valid ? commit_is_if.alu_data.thread_mask :
                                    lsu_valid ? commit_is_if.lsu_data.thread_mask :   
                                    csr_valid ? commit_is_if.csr_data.thread_mask :   
                                    mul_valid ? commit_is_if.mul_data.thread_mask :                            
                                    fpu_valid ? commit_is_if.fpu_data.thread_mask :  
                                                0;

    assign writeback_tmp_if.rd =    alu_valid ? commit_is_if.alu_data.rd :
                                    lsu_valid ? commit_is_if.lsu_data.rd :                           
                                    csr_valid ? commit_is_if.csr_data.rd :                           
                                    mul_valid ? commit_is_if.mul_data.rd :                            
                                    fpu_valid ? commit_is_if.fpu_data.rd :                                                               
                                                0;

    assign writeback_tmp_if.rd_is_fp = alu_valid ? 0 :
                                       lsu_valid ? commit_is_if.lsu_data.rd_is_fp :                            
                                       csr_valid ? 0 :                                                               
                                       mul_valid ? 0 :                           
                                       fpu_valid ? commit_is_if.fpu_data.rd_is_fp :                          
                                                   0; 

    assign writeback_tmp_if.data =  alu_valid ? alu_commit_if.data :
                                    lsu_valid ? lsu_commit_if.data :                           
                                    csr_valid ? csr_commit_if.data :                           
                                    mul_valid ? mul_commit_if.data :                            
                                    fpu_valid ? fpu_commit_if.data :                                                               
                                                0;

    wire stall = ~writeback_if.ready && writeback_if.valid;

    VX_generic_register #(
        .N(1 + `NW_BITS + 32 + `NUM_THREADS + `NR_BITS + (`NUM_THREADS * 32) + 1)
    ) wb_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({writeback_tmp_if.valid, writeback_tmp_if.warp_num, writeback_tmp_if.curr_PC, writeback_tmp_if.thread_mask, writeback_tmp_if.rd, writeback_tmp_if.rd_is_fp, writeback_tmp_if.data}),
        .out   ({writeback_if.valid,     writeback_if.warp_num,     writeback_if.curr_PC,     writeback_if.thread_mask,     writeback_if.rd,     writeback_if.rd_is_fp,     writeback_if.data})
    );

    assign alu_commit_if.ready = !stall;    
    assign lsu_commit_if.ready = !stall && !alu_valid;   
    assign csr_commit_if.ready = !stall && !alu_valid && !lsu_valid;
    assign mul_commit_if.ready = !stall && !alu_valid && !lsu_valid && !csr_valid;    
    assign fpu_commit_if.ready = !stall && !alu_valid && !lsu_valid && !csr_valid && !mul_valid;    
    
    // special workaround to control RISC-V benchmarks termination on Verilator
    reg [31:0] last_data_wb /* verilator public */;
    always @(posedge clk) begin
        if (writeback_tmp_if.valid && ~stall && (writeback_tmp_if.rd == 28)) begin
            last_data_wb <= writeback_tmp_if.data[0];
        end
    end

endmodule