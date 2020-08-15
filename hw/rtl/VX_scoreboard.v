`include "VX_define.vh"

module VX_scoreboard  #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_decode_if        decode_if,
    VX_wb_if            writeback_if,  
    VX_cmt_to_issue_if  cmt_to_issue_if,    
    input wire          ex_busy,      
    output wire [`ISTAG_BITS-1:0] issue_tag,
    output wire         schedule_delay
);
    reg [`NUM_REGS-1:0] inuse_reg_mask [`NUM_WARPS-1:0];
    
    wire [`NUM_REGS-1:0] inuse_mask = inuse_reg_mask[decode_if.wid] & decode_if.reg_use_mask;
    wire inuse_hazard = (inuse_mask != 0);

    wire issue_buf_full;

    assign schedule_delay = ex_busy || inuse_hazard || issue_buf_full;

    wire issue_fire = decode_if.valid && decode_if.ready;
    
    wire reserve_rd = issue_fire && (decode_if.wb != 0);

    wire release_rd = writeback_if.valid;

    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < `NUM_WARPS; i++) begin
                inuse_reg_mask[i] <= `NUM_REGS'(0);
            end            
        end else begin
            if (reserve_rd) begin
                inuse_reg_mask[decode_if.wid][decode_if.rd] <= 1;                
            end       
            if (release_rd) begin
                assert(inuse_reg_mask[writeback_if.wid][writeback_if.rd] != 0);
                inuse_reg_mask[writeback_if.wid][writeback_if.rd] <= 0;
            end            
        end
    end

    VX_cam_buffer #(
        .DATAW  ($bits(issue_data_t)),
        .SIZE   (`ISSUEQ_SIZE),
        .RPORTS (`NUM_EXS)
    ) issue_table (
        .clk          (clk),
        .reset        (reset),
        .write_data   ({decode_if.wid, decode_if.thread_mask, decode_if.curr_PC, decode_if.rd, decode_if.wb}),    
        .write_addr   (issue_tag),        
        .acquire_slot (issue_fire), 
        .release_slot ({cmt_to_issue_if.alu_valid, cmt_to_issue_if.bru_valid, cmt_to_issue_if.lsu_valid, cmt_to_issue_if.csr_valid, cmt_to_issue_if.mul_valid, cmt_to_issue_if.fpu_valid, cmt_to_issue_if.gpu_valid}),           
        .read_addr    ({cmt_to_issue_if.alu_tag,   cmt_to_issue_if.bru_tag,   cmt_to_issue_if.lsu_tag,   cmt_to_issue_if.csr_tag,   cmt_to_issue_if.mul_tag,   cmt_to_issue_if.fpu_tag,   cmt_to_issue_if.gpu_tag}),
        .read_data    ({cmt_to_issue_if.alu_data,  cmt_to_issue_if.bru_data,  cmt_to_issue_if.lsu_data,  cmt_to_issue_if.csr_data,  cmt_to_issue_if.mul_data,  cmt_to_issue_if.fpu_data,  cmt_to_issue_if.gpu_data}),
        .full         (issue_buf_full)
    );

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (decode_if.valid && ~decode_if.ready) begin
            $display("%t: Core%0d-stall: wid=%0d, PC=%0h, rd=%0d, wb=%0d, ib_full=%b, inuse=%b%b%b%b, ex_busy=%b",
                    $time, CORE_ID, decode_if.wid, decode_if.curr_PC, decode_if.rd, decode_if.wb, issue_buf_full,
                    inuse_mask[decode_if.rd], inuse_mask[decode_if.rs1], inuse_mask[decode_if.rs2], inuse_mask[decode_if.rs3], ex_busy);
        end
    end
`endif

endmodule