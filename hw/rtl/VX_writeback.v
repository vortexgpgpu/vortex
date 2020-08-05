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

    reg [`NUM_THREADS-1:0][31:0] wb_data_table [`ISSUEQ_SIZE-1:0];
    reg [`NW_BITS-1:0]           wb_warp_num_table [`ISSUEQ_SIZE-1:0];
    reg [`NUM_THREADS-1:0]       wb_thread_mask_table [`ISSUEQ_SIZE-1:0];
    reg [31:0]                   wb_curr_PC_table [`ISSUEQ_SIZE-1:0];
    reg [`NR_BITS-1:0]           wb_rd_table [`ISSUEQ_SIZE-1:0];

    reg [`ISSUEQ_SIZE-1:0] wb_valid_table;
    reg [`ISSUEQ_SIZE-1:0] wb_valid_table_n;

    reg [`ISTAG_BITS-1:0]  wb_index;
    wire [`ISTAG_BITS-1:0] wb_index_n;
    
    reg wb_valid;
    wire wb_valid_n;

    always @(*) begin
        wb_valid_table_n = wb_valid_table;   

        if (wb_valid) begin
            wb_valid_table_n[wb_index] = 0;    
        end

        if (alu_commit_if.valid) begin
            wb_valid_table_n [alu_commit_if.issue_tag] = cmt_to_issue_if.alu_data.wb;
        end
        if (lsu_commit_if.valid) begin
            wb_valid_table_n [lsu_commit_if.issue_tag] = cmt_to_issue_if.lsu_data.wb;
        end
        if (csr_commit_if.valid) begin
            wb_valid_table_n [csr_commit_if.issue_tag] = cmt_to_issue_if.csr_data.wb;
        end
        if (mul_commit_if.valid) begin
            wb_valid_table_n [mul_commit_if.issue_tag] = cmt_to_issue_if.mul_data.wb;
        end
        if (fpu_commit_if.valid) begin
            wb_valid_table_n [fpu_commit_if.issue_tag] = cmt_to_issue_if.fpu_data.wb;
        end        
    end

    VX_priority_encoder #(
        .N(`ISSUEQ_SIZE)
    ) wb_select (
        .data_in   (wb_valid_table_n),
        .data_out  (wb_index_n),
        .valid_out (wb_valid_n)
    );

    always @(posedge clk) begin
        if (reset) begin
            wb_valid_table <= 0;
            wb_index   <= 0;
            wb_valid   <= 0;  
        end else begin
            if (alu_commit_if.valid) begin
                wb_data_table [alu_commit_if.issue_tag]       <= alu_commit_if.data;
                wb_warp_num_table [alu_commit_if.issue_tag]   <= cmt_to_issue_if.alu_data.warp_num;
                wb_thread_mask_table [alu_commit_if.issue_tag] <= cmt_to_issue_if.alu_data.thread_mask;
                wb_curr_PC_table [alu_commit_if.issue_tag]    <= cmt_to_issue_if.alu_data.curr_PC;
                wb_rd_table [alu_commit_if.issue_tag]         <= cmt_to_issue_if.alu_data.rd;
            end

            if (lsu_commit_if.valid) begin
                wb_data_table [lsu_commit_if.issue_tag]       <= lsu_commit_if.data;
                wb_warp_num_table [lsu_commit_if.issue_tag]   <= cmt_to_issue_if.lsu_data.warp_num;
                wb_thread_mask_table [lsu_commit_if.issue_tag] <= cmt_to_issue_if.lsu_data.thread_mask;
                wb_curr_PC_table [lsu_commit_if.issue_tag]    <= cmt_to_issue_if.lsu_data.curr_PC;
                wb_rd_table [lsu_commit_if.issue_tag]         <= cmt_to_issue_if.lsu_data.rd;
            end

            if (csr_commit_if.valid) begin
                wb_data_table [csr_commit_if.issue_tag]       <= csr_commit_if.data;
                wb_warp_num_table [csr_commit_if.issue_tag]   <= cmt_to_issue_if.csr_data.warp_num;
                wb_thread_mask_table [csr_commit_if.issue_tag] <= cmt_to_issue_if.csr_data.thread_mask;
                wb_curr_PC_table [csr_commit_if.issue_tag]    <= cmt_to_issue_if.csr_data.curr_PC;
                wb_rd_table [csr_commit_if.issue_tag]         <= cmt_to_issue_if.csr_data.rd;
            end

            if (mul_commit_if.valid) begin
                wb_data_table [mul_commit_if.issue_tag]       <= mul_commit_if.data;
                wb_warp_num_table [mul_commit_if.issue_tag]   <= cmt_to_issue_if.mul_data.warp_num;
                wb_thread_mask_table [mul_commit_if.issue_tag] <= cmt_to_issue_if.mul_data.thread_mask;
                wb_curr_PC_table [mul_commit_if.issue_tag]    <= cmt_to_issue_if.mul_data.curr_PC;
                wb_rd_table [mul_commit_if.issue_tag]         <= cmt_to_issue_if.mul_data.rd;
            end
            
            if (fpu_commit_if.valid) begin
                wb_data_table [fpu_commit_if.issue_tag]       <= fpu_commit_if.data;
                wb_warp_num_table [fpu_commit_if.issue_tag]   <= cmt_to_issue_if.fpu_data.warp_num;
                wb_thread_mask_table [fpu_commit_if.issue_tag] <= cmt_to_issue_if.fpu_data.thread_mask;
                wb_curr_PC_table [fpu_commit_if.issue_tag]    <= cmt_to_issue_if.fpu_data.curr_PC;
                wb_rd_table [fpu_commit_if.issue_tag]         <= cmt_to_issue_if.fpu_data.rd;
            end 

            wb_valid_table <= wb_valid_table_n;
            wb_index   <= wb_index_n;
            wb_valid   <= wb_valid_n && writeback_if.ready;
        end        
    end 

    // writeback request
    assign writeback_if.valid       = wb_valid;
    assign writeback_if.warp_num    = wb_warp_num_table [wb_index];
    assign writeback_if.thread_mask = wb_thread_mask_table [wb_index];
    assign writeback_if.curr_PC     = wb_curr_PC_table [wb_index];
    assign writeback_if.rd          = wb_rd_table [wb_index];
    assign writeback_if.data        = wb_data_table [wb_index];    

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