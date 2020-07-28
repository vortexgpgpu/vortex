`include "VX_define.vh"

module VX_writeback #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    // inputs
    VX_exu_to_cmt_if    alu_commit_if,
    VX_exu_to_cmt_if    lsu_commit_if,  
    VX_exu_to_cmt_if    csr_commit_if,
    VX_exu_to_cmt_if    mul_commit_if,
    VX_fpu_to_cmt_if    fpu_commit_if,        
    VX_exu_to_cmt_if    gpu_commit_if,
    VX_cmt_to_issue_if  cmt_to_issue_if,

    // outputs
    VX_wb_if            writeback_if
);

    reg [`NUM_THREADS-1:0][31:0] wb_data [`ISSUEQ_SIZE-1:0];
    reg [`NW_BITS-1:0] wb_warp_num [`ISSUEQ_SIZE-1:0];
    reg [`NUM_THREADS-1:0] wb_thread_mask [`ISSUEQ_SIZE-1:0];
    reg [31:0] wb_curr_PC [`ISSUEQ_SIZE-1:0];
    reg [`NR_BITS-1:0] wb_rd [`ISSUEQ_SIZE-1:0];
    reg wb_rd_is_fp [`ISSUEQ_SIZE-1:0];
    reg [`ISSUEQ_SIZE-1:0] wb_pending;

    wire [`ISTAG_BITS-1:0] wb_index;
    wire wb_valid, wb_valid_unqual;

    VX_priority_encoder #(
        .N(`ISSUEQ_SIZE)
    ) free_slots_encoder (
        .data_in   (wb_pending),
        .data_out  (wb_index),
        .valid_out (wb_valid_unqual)
    );

    assign wb_valid = wb_valid_unqual && writeback_if.ready;

    always @(posedge clk) begin
        if (reset) begin
            wb_pending <= 0;
        end else begin
            if (alu_commit_if.valid) begin
                wb_data [alu_commit_if.issue_tag]       <= alu_commit_if.data;
                wb_warp_num [alu_commit_if.issue_tag]   <= cmt_to_issue_if.alu_data.warp_num;
                wb_thread_mask [alu_commit_if.issue_tag] <= cmt_to_issue_if.alu_data.thread_mask;
                wb_curr_PC [alu_commit_if.issue_tag]    <= cmt_to_issue_if.alu_data.curr_PC;
                wb_rd [alu_commit_if.issue_tag]         <= cmt_to_issue_if.alu_data.rd;
                wb_rd_is_fp [alu_commit_if.issue_tag]   <= cmt_to_issue_if.alu_data.rd_is_fp;
                wb_pending [alu_commit_if.issue_tag]    <= cmt_to_issue_if.alu_data.wb;
            end
            if (lsu_commit_if.valid) begin
                wb_data [lsu_commit_if.issue_tag]       <= lsu_commit_if.data;
                wb_warp_num [lsu_commit_if.issue_tag]   <= cmt_to_issue_if.lsu_data.warp_num;
                wb_thread_mask [lsu_commit_if.issue_tag] <= cmt_to_issue_if.lsu_data.thread_mask;
                wb_curr_PC [lsu_commit_if.issue_tag]    <= cmt_to_issue_if.lsu_data.curr_PC;
                wb_rd [lsu_commit_if.issue_tag]         <= cmt_to_issue_if.lsu_data.rd;
                wb_rd_is_fp [lsu_commit_if.issue_tag]   <= cmt_to_issue_if.lsu_data.rd_is_fp;
                wb_pending [lsu_commit_if.issue_tag]    <= cmt_to_issue_if.lsu_data.wb;
            end
            if (csr_commit_if.valid) begin
                wb_data [csr_commit_if.issue_tag]       <= csr_commit_if.data;
                wb_warp_num [csr_commit_if.issue_tag]   <= cmt_to_issue_if.csr_data.warp_num;
                wb_thread_mask [csr_commit_if.issue_tag] <= cmt_to_issue_if.csr_data.thread_mask;
                wb_curr_PC [csr_commit_if.issue_tag]    <= cmt_to_issue_if.csr_data.curr_PC;
                wb_rd [csr_commit_if.issue_tag]         <= cmt_to_issue_if.csr_data.rd;
                wb_rd_is_fp [csr_commit_if.issue_tag]   <= cmt_to_issue_if.csr_data.rd_is_fp;
                wb_pending [csr_commit_if.issue_tag]    <= cmt_to_issue_if.csr_data.wb;
            end
            if (mul_commit_if.valid) begin
                wb_data [mul_commit_if.issue_tag]       <= mul_commit_if.data;
                wb_warp_num [mul_commit_if.issue_tag]   <= cmt_to_issue_if.mul_data.warp_num;
                wb_thread_mask [mul_commit_if.issue_tag] <= cmt_to_issue_if.mul_data.thread_mask;
                wb_curr_PC [mul_commit_if.issue_tag]    <= cmt_to_issue_if.mul_data.curr_PC;
                wb_rd [mul_commit_if.issue_tag]         <= cmt_to_issue_if.mul_data.rd;
                wb_rd_is_fp [mul_commit_if.issue_tag]   <= cmt_to_issue_if.mul_data.rd_is_fp;
                wb_pending [mul_commit_if.issue_tag]    <= cmt_to_issue_if.mul_data.wb;
            end
            if (fpu_commit_if.valid) begin
                wb_data [fpu_commit_if.issue_tag]       <= fpu_commit_if.data;
                wb_warp_num [fpu_commit_if.issue_tag]   <= cmt_to_issue_if.fpu_data.warp_num;
                wb_thread_mask [fpu_commit_if.issue_tag] <= cmt_to_issue_if.fpu_data.thread_mask;
                wb_curr_PC [fpu_commit_if.issue_tag]    <= cmt_to_issue_if.fpu_data.curr_PC;
                wb_rd [fpu_commit_if.issue_tag]         <= cmt_to_issue_if.fpu_data.rd;
                wb_rd_is_fp [fpu_commit_if.issue_tag]   <= cmt_to_issue_if.fpu_data.rd_is_fp;
                wb_pending [fpu_commit_if.issue_tag]    <= cmt_to_issue_if.fpu_data.wb;
            end 
            if (wb_valid) begin
                wb_pending [wb_index] <= 0;
            end        
        end        
    end 

    // writeback request
    assign writeback_if.valid       = wb_pending [wb_index];
    assign writeback_if.warp_num    = wb_warp_num [wb_index];
    assign writeback_if.thread_mask = wb_thread_mask [wb_index];
    assign writeback_if.curr_PC     = wb_curr_PC [wb_index];
    assign writeback_if.rd          = wb_rd [wb_index];
    assign writeback_if.rd_is_fp    = wb_rd_is_fp [wb_index];
    assign writeback_if.data        = wb_data [wb_index];    

    // commit back-pressure
    assign alu_commit_if.ready = 1'b1;    
    assign lsu_commit_if.ready = 1'b1;   
    assign csr_commit_if.ready = 1'b1;
    assign mul_commit_if.ready = 1'b1;    
    assign fpu_commit_if.ready = 1'b1;    
    assign gpu_commit_if.ready = 1'b1;   
    
    // special workaround to get RISC-V tests Pass/Fail status
    reg [31:0] last_wb_value [`NUM_REGS-1:0] /* verilator public */;
    always @(posedge clk) begin
        if (writeback_if.valid) begin
            last_wb_value[writeback_if.rd] <= writeback_if.data[0];
        end
    end

endmodule