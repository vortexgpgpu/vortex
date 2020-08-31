`include "VX_define.vh"

module VX_issue #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_decode_if        decode_if,
    VX_writeback_if     writeback_if,   
    VX_csr_to_issue_if  csr_to_issue_if, 
    
    VX_alu_req_if       alu_req_if,
    VX_lsu_req_if       lsu_req_if,    
    VX_csr_req_if       csr_req_if,
    VX_mul_req_if       mul_req_if,    
    VX_fpu_req_if       fpu_req_if,    
    VX_gpu_req_if       gpu_req_if
);
    VX_decode_if    ibuf_deq_if();
    VX_decode_if    execute_if();
    VX_gpr_read_if  gpr_read_if();

    wire scoreboard_delay;
    wire [`NW_BITS-1:0] deq_wid_next;

    VX_ibuffer #(
        .CORE_ID(CORE_ID)
    ) ibuffer (
        .clk            (clk),
        .reset          (reset), 
        .freeze         (~gpr_read_if.ready_in),
        .ibuf_enq_if    (decode_if),
        .deq_wid_next   (deq_wid_next),
        .ibuf_deq_if    (ibuf_deq_if)      
    );

    VX_scoreboard #(
        .CORE_ID(CORE_ID)
    ) scoreboard (
        .clk            (clk),
        .reset          (reset), 
        .ibuf_deq_if    (ibuf_deq_if),
        .writeback_if   (writeback_if),
        .deq_wid_next   (deq_wid_next),
        .exe_delay      (~execute_if.ready),
        .gpr_delay      (~gpr_read_if.ready_in),
        .delay          (scoreboard_delay)
    );
        
    assign gpr_read_if.valid   = ibuf_deq_if.valid && ~scoreboard_delay;
    assign gpr_read_if.wid     = ibuf_deq_if.wid;
    assign gpr_read_if.rs1     = ibuf_deq_if.rs1;
    assign gpr_read_if.rs2     = ibuf_deq_if.rs2;
    assign gpr_read_if.rs3     = ibuf_deq_if.rs3;
    assign gpr_read_if.use_rs3 = ibuf_deq_if.use_rs3;
    assign gpr_read_if.ready_out = execute_if.ready;

    VX_gpr_stage #(
        .CORE_ID(CORE_ID)
    ) gpr_stage (
        .clk            (clk),      
        .reset          (reset),          
        .writeback_if   (writeback_if),
        .gpr_read_if    (gpr_read_if)
    );
    
    assign execute_if.valid     = ibuf_deq_if.valid && gpr_read_if.ready_in && ~scoreboard_delay;
    assign execute_if.wid       = ibuf_deq_if.wid;
    assign execute_if.thread_mask = ibuf_deq_if.thread_mask;
    assign execute_if.curr_PC   = ibuf_deq_if.curr_PC;
    assign execute_if.ex_type   = ibuf_deq_if.ex_type;    
    assign execute_if.op_type   = ibuf_deq_if.op_type; 
    assign execute_if.op_mod    = ibuf_deq_if.op_mod;    
    assign execute_if.wb        = ibuf_deq_if.wb;
    assign execute_if.rd        = ibuf_deq_if.rd;
    assign execute_if.rs1       = ibuf_deq_if.rs1;
    assign execute_if.imm       = ibuf_deq_if.imm;        
    assign execute_if.rs1_is_PC = ibuf_deq_if.rs1_is_PC;
    assign execute_if.rs2_is_imm = ibuf_deq_if.rs2_is_imm;

    VX_instr_demux instr_demux (
        .clk            (clk),      
        .reset          (reset),
        .execute_if     (execute_if),
        .gpr_read_if    (gpr_read_if),
        .csr_to_issue_if(csr_to_issue_if),
        .alu_req_if     (alu_req_if),
        .lsu_req_if     (lsu_req_if),        
        .csr_req_if     (csr_req_if),
        .mul_req_if     (mul_req_if),
        .fpu_req_if     (fpu_req_if),
        .gpu_req_if     (gpu_req_if)
    );      

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (alu_req_if.valid && alu_req_if.ready) begin
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=ALU, tmask=%b, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, alu_req_if.wid, alu_req_if.curr_PC, alu_req_if.thread_mask, alu_req_if.rs1_data, alu_req_if.rs2_data);   
        end
        if (lsu_req_if.valid && lsu_req_if.ready) begin
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=LSU, tmask=%b, rw=%b, byteen=%b, baddr=%0h, offset=%0h, data=%0h", $time, CORE_ID, lsu_req_if.wid, lsu_req_if.curr_PC, lsu_req_if.thread_mask, lsu_req_if.rw, lsu_req_if.byteen, lsu_req_if.base_addr, lsu_req_if.offset, lsu_req_if.store_data);   
        end
        if (csr_req_if.valid && csr_req_if.ready) begin
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=CSR, tmask=%b, addr=%0h, mask=%0h", $time, CORE_ID, csr_req_if.wid, csr_req_if.curr_PC, csr_req_if.thread_mask, csr_req_if.csr_addr, csr_req_if.csr_mask);   
        end
        if (mul_req_if.valid && mul_req_if.ready) begin
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=MUL, tmask=%b, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, mul_req_if.wid, mul_req_if.curr_PC, mul_req_if.thread_mask, mul_req_if.rs1_data, mul_req_if.rs2_data);   
        end
        if (fpu_req_if.valid && fpu_req_if.ready) begin
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=FPU, tmask=%b, rs1_data=%0h, rs2_data=%0h, rs3_data=%0h", $time, CORE_ID, fpu_req_if.wid, fpu_req_if.curr_PC, fpu_req_if.thread_mask, fpu_req_if.rs1_data, fpu_req_if.rs2_data, fpu_req_if.rs3_data);   
        end
        if (gpu_req_if.valid && gpu_req_if.ready) begin
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=GPU, tmask=%b, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, gpu_req_if.wid, gpu_req_if.curr_PC, gpu_req_if.thread_mask, gpu_req_if.rs1_data, gpu_req_if.rs2_data);   
        end
    end
`endif

endmodule