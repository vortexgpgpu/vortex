`include "VX_define.vh"

module VX_writeback #(
    parameter CORE_ID = 0
) (
    input wire  clk,
    input wire  reset,

    // inputs
    VX_wb_if    alu_wb_if,
    VX_wb_if    branch_wb_if,
    VX_wb_if    lsu_wb_if,  
    VX_wb_if    mul_wb_if,    
    VX_wb_if    csr_wb_if,

    // outputs
    VX_wb_if    writeback_if,
    output wire notify_commit
);

    wire br_valid  = (| branch_wb_if.valid);
    wire lsu_valid = (| lsu_wb_if.valid);
    wire mul_valid = (| mul_wb_if.valid);
    wire alu_valid = (| alu_wb_if.valid);
    wire csr_valid = (| csr_wb_if.valid);

    VX_wb_if writeback_tmp_if();    

    assign writeback_tmp_if.valid =  br_valid ? branch_wb_if.valid :
                                    lsu_valid ? lsu_wb_if.valid :
                                    mul_valid ? mul_wb_if.valid :             
                                    alu_valid ? alu_wb_if.valid :                            
                                    csr_valid ? csr_wb_if.valid :                                                 
                                                0;     

    assign writeback_tmp_if.warp_num = br_valid ? branch_wb_if.warp_num :
                                    lsu_valid ? lsu_wb_if.warp_num :
                                    mul_valid ? mul_wb_if.warp_num :   
                                    alu_valid ? alu_wb_if.warp_num :                            
                                    csr_valid ? csr_wb_if.warp_num :                           
                                    
                                                0;   

    assign writeback_tmp_if.curr_PC = br_valid ? branch_wb_if.curr_PC :
                                    lsu_valid ? lsu_wb_if.curr_PC :
                                    mul_valid ? mul_wb_if.curr_PC :    
                                    alu_valid ? alu_wb_if.curr_PC :                            
                                    csr_valid ? csr_wb_if.curr_PC :                                                                                      
                                                0;

    assign writeback_tmp_if.data =   br_valid ? branch_wb_if.data :
                                    lsu_valid ? lsu_wb_if.data :
                                    mul_valid ? mul_wb_if.data :                           
                                    alu_valid ? alu_wb_if.data :                            
                                    csr_valid ? csr_wb_if.data :                                                               
                                                0;

    assign writeback_tmp_if.rd =     br_valid ? branch_wb_if.rd :
                                    lsu_valid ? lsu_wb_if.rd :
                                    mul_valid ? mul_wb_if.rd :                           
                                    alu_valid ? alu_wb_if.rd :                            
                                    csr_valid ? csr_wb_if.rd :                                                               
                                                0;

    assign writeback_tmp_if.wb =     br_valid ? branch_wb_if.wb :
                                    lsu_valid ? lsu_wb_if.wb :
                                    alu_valid ? alu_wb_if.wb :                            
                                    csr_valid ? csr_wb_if.wb :                           
                                    mul_valid ? mul_wb_if.wb :                           
                                                0; 

    wire stall = ~writeback_if.ready && (| writeback_if.valid);

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + `NR_BITS + (`NUM_THREADS * 32) + `WB_BITS)
    ) wb_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({writeback_tmp_if.valid, writeback_tmp_if.warp_num, writeback_tmp_if.curr_PC, writeback_tmp_if.rd, writeback_tmp_if.data, writeback_tmp_if.wb}),
        .out   ({writeback_if.valid,     writeback_if.warp_num,     writeback_if.curr_PC,     writeback_if.rd,     writeback_if.data,     writeback_if.wb})
    );

    assign branch_wb_if.ready = !stall;    
    assign lsu_wb_if.ready    = !stall && !br_valid;    
    assign mul_wb_if.ready    = !stall && !br_valid && !lsu_valid;
    assign alu_wb_if.ready    = !stall && !br_valid && !lsu_valid && !mul_valid;    
    assign csr_wb_if.ready    = !stall && !br_valid && !lsu_valid && !mul_valid && !alu_valid;    
    
    assign notify_commit = (| writeback_tmp_if.valid) && ~stall;

    // special workaround to control RISC-V benchmarks termination on Verilator
    reg [31:0] last_data_wb /* verilator public */;
    always @(posedge clk) begin
        if (notify_commit && (writeback_tmp_if.wb != 0) && (writeback_tmp_if.rd == 28)) begin
            last_data_wb <= writeback_tmp_if.data[0];
        end
    end

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if ((| writeback_tmp_if.valid) && ~stall) begin
            $display("%t: Core%0d-WB: warp=%0d, PC=%0h, rd=%0d, wb=%0d, data=%0h", $time, CORE_ID, writeback_tmp_if.warp_num, writeback_tmp_if.curr_PC, writeback_tmp_if.rd, writeback_tmp_if.wb, writeback_tmp_if.data);
        end 
    end
`endif

endmodule







