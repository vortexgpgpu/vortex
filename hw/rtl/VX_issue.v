`include "VX_define.vh"

module VX_issue #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_decode_if        decode_if,
    VX_wb_if            writeback_if,
    
    VX_alu_req_if       alu_req_if,
    VX_lsu_req_if       lsu_req_if,    
    VX_csr_req_if       csr_req_if,
    VX_mul_req_if       mul_req_if,    
    VX_gpu_req_if       gpu_req_if
);
    VX_execute_if execute_if();

    VX_scheduler #(
        .CORE_ID(CORE_ID)
    ) scheduler (
        .clk            (clk),
        .reset          (reset), 
        .decode_if      (decode_if),
        .writeback_if   (writeback_if),        
        .execute_if     (execute_if),        
        `UNUSED_PIN     (is_empty)
    );

    VX_gpr_stage #(
        .CORE_ID(CORE_ID)
    ) gpr_stage (
        .clk            (clk),
        .reset          (reset),        
        
        .execute_if     (execute_if),        
        .writeback_if   (writeback_if),

        .alu_req_if     (alu_req_if),
        .lsu_req_if     (lsu_req_if),        
        .csr_req_if     (csr_req_if),
        .mul_req_if     (mul_req_if),
        .gpu_req_if     (gpu_req_if)
    );

endmodule