`include "VX_define.vh"

module VX_writeback #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    // inputs
    VX_exu_to_cmt_if    alu_commit_if,
    VX_exu_to_cmt_if    bru_commit_if,
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
    reg [`ISSUEQ_SIZE-1:0][`NW_BITS-1:0]           wb_wid_table, wb_wid_table_n;    
    reg [`ISSUEQ_SIZE-1:0][`NUM_THREADS-1:0]       wb_thread_mask_table, wb_thread_mask_table_n;
    reg [`ISSUEQ_SIZE-1:0][31:0]                   wb_curr_PC_table, wb_curr_PC_table_n;
    reg [`ISSUEQ_SIZE-1:0][`NR_BITS-1:0]           wb_rd_table, wb_rd_table_n;    

    reg                          wb_valid, wb_valid_n;
    reg [`NUM_THREADS-1:0][31:0] wb_data, wb_data_n;
    reg [`NW_BITS-1:0]           wb_wid, wb_wid_n;    
    reg [`NUM_THREADS-1:0]       wb_thread_mask, wb_thread_mask_n;
    reg [31:0]                   wb_curr_PC, wb_curr_PC_n;
    reg [`NR_BITS-1:0]           wb_rd, wb_rd_n;

    reg [`ISTAG_BITS-1:0] wb_index;
    reg [`ISTAG_BITS-1:0] wb_index_n;    

    always @(*) begin
        wb_valid_table_n        = wb_valid_table;  
        wb_wid_table_n     = wb_wid_table; 
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
            wb_wid_table_n [alu_commit_if.issue_tag]         = cmt_to_issue_if.alu_data.wid;                
            wb_curr_PC_table_n [alu_commit_if.issue_tag]     = cmt_to_issue_if.alu_data.curr_PC;
            wb_rd_table_n [alu_commit_if.issue_tag]          = cmt_to_issue_if.alu_data.rd;
        end

        if (bru_commit_if.valid) begin
            wb_valid_table_n [bru_commit_if.issue_tag]       = cmt_to_issue_if.bru_data.wb;
            wb_thread_mask_table_n [bru_commit_if.issue_tag] = cmt_to_issue_if.bru_data.thread_mask;
            wb_data_table_n [bru_commit_if.issue_tag]        = bru_commit_if.data;
            wb_wid_table_n [bru_commit_if.issue_tag]         = cmt_to_issue_if.bru_data.wid;                
            wb_curr_PC_table_n [bru_commit_if.issue_tag]     = cmt_to_issue_if.bru_data.curr_PC;
            wb_rd_table_n [bru_commit_if.issue_tag]          = cmt_to_issue_if.bru_data.rd;
        end

        if (lsu_commit_if.valid) begin
            wb_valid_table_n [lsu_commit_if.issue_tag]       = cmt_to_issue_if.lsu_data.wb;
            wb_thread_mask_table_n [lsu_commit_if.issue_tag] = cmt_to_issue_if.lsu_data.thread_mask;
            wb_data_table_n [lsu_commit_if.issue_tag]        = lsu_commit_if.data;
            wb_wid_table_n [lsu_commit_if.issue_tag]         = cmt_to_issue_if.lsu_data.wid;                
            wb_curr_PC_table_n [lsu_commit_if.issue_tag]     = cmt_to_issue_if.lsu_data.curr_PC;
            wb_rd_table_n [lsu_commit_if.issue_tag]          = cmt_to_issue_if.lsu_data.rd;
        end

        if (csr_commit_if.valid) begin
            wb_valid_table_n [csr_commit_if.issue_tag]       = cmt_to_issue_if.csr_data.wb;
            wb_thread_mask_table_n [csr_commit_if.issue_tag] = cmt_to_issue_if.csr_data.thread_mask;
            wb_data_table_n [csr_commit_if.issue_tag]        = csr_commit_if.data;
            wb_wid_table_n [csr_commit_if.issue_tag]         = cmt_to_issue_if.csr_data.wid;                
            wb_curr_PC_table_n [csr_commit_if.issue_tag]     = cmt_to_issue_if.csr_data.curr_PC;
            wb_rd_table_n [csr_commit_if.issue_tag]          = cmt_to_issue_if.csr_data.rd;
        end

        if (mul_commit_if.valid) begin
            wb_valid_table_n [mul_commit_if.issue_tag]       = cmt_to_issue_if.mul_data.wb;
            wb_thread_mask_table_n [mul_commit_if.issue_tag] = cmt_to_issue_if.mul_data.thread_mask;
            wb_data_table_n [mul_commit_if.issue_tag]        = mul_commit_if.data;
            wb_wid_table_n [mul_commit_if.issue_tag]         = cmt_to_issue_if.mul_data.wid;                
            wb_curr_PC_table_n [mul_commit_if.issue_tag]     = cmt_to_issue_if.mul_data.curr_PC;
            wb_rd_table_n [mul_commit_if.issue_tag]          = cmt_to_issue_if.mul_data.rd;
        end

        if (fpu_commit_if.valid) begin
            wb_valid_table_n [fpu_commit_if.issue_tag]       = cmt_to_issue_if.fpu_data.wb;
            wb_thread_mask_table_n [fpu_commit_if.issue_tag] = cmt_to_issue_if.fpu_data.thread_mask;
            wb_data_table_n [fpu_commit_if.issue_tag]        = fpu_commit_if.data;
            wb_wid_table_n [fpu_commit_if.issue_tag]         = cmt_to_issue_if.fpu_data.wid;                
            wb_curr_PC_table_n [fpu_commit_if.issue_tag]     = cmt_to_issue_if.fpu_data.curr_PC;
            wb_rd_table_n [fpu_commit_if.issue_tag]          = cmt_to_issue_if.fpu_data.rd;
        end

        if (gpu_commit_if.valid) begin
            wb_valid_table_n [gpu_commit_if.issue_tag]       = cmt_to_issue_if.gpu_data.wb;
            wb_thread_mask_table_n [gpu_commit_if.issue_tag] = cmt_to_issue_if.gpu_data.thread_mask;
            wb_data_table_n [gpu_commit_if.issue_tag]        = gpu_commit_if.data;
            wb_wid_table_n [gpu_commit_if.issue_tag]         = cmt_to_issue_if.gpu_data.wid;                
            wb_curr_PC_table_n [gpu_commit_if.issue_tag]     = cmt_to_issue_if.gpu_data.curr_PC;
            wb_rd_table_n [gpu_commit_if.issue_tag]          = cmt_to_issue_if.gpu_data.rd;
        end        
    end

    always @(*) begin
        wb_index_n       = 0;
        wb_valid_n       = 0;
        wb_thread_mask_n = {`NUM_THREADS{1'bx}};
        wb_wid_n         = {`NW_BITS{1'bx}};
        wb_curr_PC_n     = {32{1'bx}};
        wb_data_n        = {(`NUM_THREADS * 32){1'bx}};
        for (integer i = `ISSUEQ_SIZE-1; i >= 0; i--) begin
            if (wb_valid_table_n[i]) begin
                wb_index_n      = `ISTAG_BITS'(i);
                wb_valid_n      = 1;
                wb_thread_mask_n= wb_thread_mask_table_n[i]; 
                wb_wid_n   = wb_wid_table_n[i]; 
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
            wb_wid_table     <= wb_wid_table_n; 
            wb_curr_PC_table      <= wb_curr_PC_table_n; 
            wb_rd_table           <= wb_rd_table_n; 
            wb_data_table         <= wb_data_table_n; 
            
            wb_index        <= wb_index_n;     
            wb_valid        <= wb_valid_n;
            wb_thread_mask  <= wb_thread_mask_n; 
            wb_wid     <= wb_wid_n; 
            wb_curr_PC      <= wb_curr_PC_n; 
            wb_rd           <= wb_rd_n; 
            wb_data         <= wb_data_n; 
        end        
    end 

    // writeback request
    assign writeback_if.valid       = wb_valid;
    assign writeback_if.thread_mask = wb_thread_mask;
    assign writeback_if.wid         = wb_wid;
    assign writeback_if.curr_PC     = wb_curr_PC;
    assign writeback_if.rd          = wb_rd;
    assign writeback_if.data        = wb_data;    
    
    // special workaround to get RISC-V tests Pass/Fail status
    reg [31:0] last_wb_value [`NUM_REGS-1:0] /* verilator public */;
    always @(posedge clk) begin
        if (writeback_if.valid) begin
            last_wb_value[writeback_if.rd] <= writeback_if.data[0];
        end
    end

endmodule