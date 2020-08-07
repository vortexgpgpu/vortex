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
    reg [`ISSUEQ_SIZE-1:0]                         wb_valid_table, wb_valid_table_n;
    reg [`ISSUEQ_SIZE-1:0][`NUM_THREADS-1:0][31:0] wb_data_table, wb_data_table_n;
    reg [`ISSUEQ_SIZE-1:0][`NW_BITS-1:0]           wb_warp_num_table, wb_warp_num_table_n;    
    reg [`ISSUEQ_SIZE-1:0][`NUM_THREADS-1:0]       wb_thread_mask_table, wb_thread_mask_table_n;
    reg [`ISSUEQ_SIZE-1:0][31:0]                   wb_curr_PC_table, wb_curr_PC_table_n;
    reg [`ISSUEQ_SIZE-1:0][`NR_BITS-1:0]           wb_rd_table, wb_rd_table_n;    

    reg [`NUM_THREADS-1:0][31:0] wb_data, wb_data_n;
    reg [`NW_BITS-1:0]           wb_warp_num, wb_warp_num_n;    
    reg [`NUM_THREADS-1:0]       wb_thread_mask, wb_thread_mask_n;
    reg [31:0]                   wb_curr_PC, wb_curr_PC_n;
    reg [`NR_BITS-1:0]           wb_rd, wb_rd_n;

    reg [`ISTAG_BITS-1:0] wb_index;
    reg [`ISTAG_BITS-1:0] wb_index_n;
    
    reg wb_valid;
    reg wb_valid_n;

    always @(*) begin
        wb_valid_table_n        = wb_valid_table;  
        wb_warp_num_table_n     = wb_warp_num_table; 
        wb_thread_mask_table_n  = wb_thread_mask_table;
        wb_curr_PC_table_n      = wb_curr_PC_table;        
        wb_rd_table_n           = wb_rd_table;
        wb_data_table_n         = wb_data_table;

        if (wb_valid) begin
            wb_valid_table_n[wb_index] = 0;    
        end

        if (alu_commit_if.valid) begin
            wb_valid_table_n [alu_commit_if.issue_tag]       = cmt_to_issue_if.alu_data.wb;
            wb_thread_mask_table_n [alu_commit_if.issue_tag] = cmt_to_issue_if.alu_data.thread_mask;
            wb_data_table_n [alu_commit_if.issue_tag]        = alu_commit_if.data;
            wb_warp_num_table_n [alu_commit_if.issue_tag]    = cmt_to_issue_if.alu_data.warp_num;                
            wb_curr_PC_table_n [alu_commit_if.issue_tag]     = cmt_to_issue_if.alu_data.curr_PC;
            wb_rd_table_n [alu_commit_if.issue_tag]          = cmt_to_issue_if.alu_data.rd;
        end

        if (lsu_commit_if.valid) begin
            wb_valid_table_n [lsu_commit_if.issue_tag]       = cmt_to_issue_if.lsu_data.wb;
            wb_thread_mask_table_n [lsu_commit_if.issue_tag] = cmt_to_issue_if.lsu_data.thread_mask;
            wb_data_table_n [lsu_commit_if.issue_tag]        = lsu_commit_if.data;
            wb_warp_num_table_n [lsu_commit_if.issue_tag]    = cmt_to_issue_if.lsu_data.warp_num;                
            wb_curr_PC_table_n [lsu_commit_if.issue_tag]     = cmt_to_issue_if.lsu_data.curr_PC;
            wb_rd_table_n [lsu_commit_if.issue_tag]          = cmt_to_issue_if.lsu_data.rd;
        end

        if (csr_commit_if.valid) begin
            wb_valid_table_n [csr_commit_if.issue_tag]       = cmt_to_issue_if.csr_data.wb;
            wb_thread_mask_table_n [csr_commit_if.issue_tag] = cmt_to_issue_if.csr_data.thread_mask;
            wb_data_table_n [csr_commit_if.issue_tag]        = csr_commit_if.data;
            wb_warp_num_table_n [csr_commit_if.issue_tag]    = cmt_to_issue_if.csr_data.warp_num;                
            wb_curr_PC_table_n [csr_commit_if.issue_tag]     = cmt_to_issue_if.csr_data.curr_PC;
            wb_rd_table_n [csr_commit_if.issue_tag]          = cmt_to_issue_if.csr_data.rd;
        end

        if (mul_commit_if.valid) begin
            wb_valid_table_n [mul_commit_if.issue_tag]       = cmt_to_issue_if.mul_data.wb;
            wb_thread_mask_table_n [mul_commit_if.issue_tag] = cmt_to_issue_if.mul_data.thread_mask;
            wb_data_table_n [mul_commit_if.issue_tag]        = mul_commit_if.data;
            wb_warp_num_table_n [mul_commit_if.issue_tag]    = cmt_to_issue_if.mul_data.warp_num;                
            wb_curr_PC_table_n [mul_commit_if.issue_tag]     = cmt_to_issue_if.mul_data.curr_PC;
            wb_rd_table_n [mul_commit_if.issue_tag]          = cmt_to_issue_if.mul_data.rd;
        end

        if (fpu_commit_if.valid) begin
            wb_valid_table_n [fpu_commit_if.issue_tag]       = cmt_to_issue_if.fpu_data.wb;
            wb_thread_mask_table_n [fpu_commit_if.issue_tag] = cmt_to_issue_if.fpu_data.thread_mask;
            wb_data_table_n [fpu_commit_if.issue_tag]        = fpu_commit_if.data;
            wb_warp_num_table_n [fpu_commit_if.issue_tag]    = cmt_to_issue_if.fpu_data.warp_num;                
            wb_curr_PC_table_n [fpu_commit_if.issue_tag]     = cmt_to_issue_if.fpu_data.curr_PC;
            wb_rd_table_n [fpu_commit_if.issue_tag]          = cmt_to_issue_if.fpu_data.rd;
        end        
    end

    integer i;

    always @(*) begin
        wb_index_n = 0;
        wb_valid_n = 0;
        for (i = `ISSUEQ_SIZE-1; i >= 0; i--) begin
            if (wb_valid_table_n[i]) begin
                wb_index_n      = `ISTAG_BITS'(i);
                wb_valid_n      = 1;
                wb_thread_mask_n= wb_thread_mask_table_n[i]; 
                wb_warp_num_n   = wb_warp_num_table_n[i]; 
                wb_curr_PC_n    = wb_curr_PC_table_n[i]; 
                wb_rd_n         = wb_rd_table_n[i]; 
                wb_data_n       = wb_data_table_n[i]; 
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            wb_valid_table <= 0;
            wb_index       <= 0;
            wb_valid       <= 0;  
        end else begin
            wb_valid_table        <= wb_valid_table_n;      
            wb_thread_mask_table  <= wb_thread_mask_table_n; 
            wb_warp_num_table     <= wb_warp_num_table_n; 
            wb_curr_PC_table      <= wb_curr_PC_table_n; 
            wb_rd_table           <= wb_rd_table_n; 
            wb_data_table         <= wb_data_table_n; 
            
            wb_index        <= wb_index_n;     
            wb_valid        <= wb_valid_n && writeback_if.ready;
            wb_thread_mask  <= wb_thread_mask_n; 
            wb_warp_num     <= wb_warp_num_n; 
            wb_curr_PC      <= wb_curr_PC_n; 
            wb_rd           <= wb_rd_n; 
            wb_data         <= wb_data_n; 
        end        
    end 

    // writeback request
    assign writeback_if.valid       = wb_valid;
    assign writeback_if.thread_mask = wb_thread_mask;
    assign writeback_if.warp_num    = wb_warp_num;
    assign writeback_if.curr_PC     = wb_curr_PC;
    assign writeback_if.rd          = wb_rd;
    assign writeback_if.data        = wb_data;    

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