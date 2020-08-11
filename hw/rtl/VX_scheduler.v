`include "VX_define.vh"

module VX_scheduler  #(
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
    localparam CTVW = `CLOG2(`NUM_WARPS * `NUM_REGS + 1);
    reg [`NUM_THREADS-1:0] inuse_registers [(`NUM_WARPS * `NUM_REGS)-1:0];    
    reg [`NUM_REGS-1:0] inuse_reg_mask [`NUM_WARPS-1:0];
    
    wire [`NUM_REGS-1:0] inuse_mask = inuse_reg_mask[decode_if.warp_num] & decode_if.reg_use_mask;
    wire inuse_hazard = (inuse_mask != 0);

    wire issue_buf_full;

    assign schedule_delay = ex_busy || inuse_hazard || issue_buf_full;

    wire issue_fire = decode_if.valid && decode_if.ready;

    wire writeback_fire = writeback_if.valid && writeback_if.ready;

    wire acquire_rd = issue_fire && (decode_if.wb != 0);

    wire [`NUM_THREADS-1:0] inuse_registers_n = inuse_registers[{writeback_if.warp_num, writeback_if.rd}] & ~writeback_if.thread_mask;

    always @(posedge clk) begin
        if (reset) begin
            for (integer w = 0; w < `NUM_WARPS; w++) begin
                for (integer i = 0; i < `NUM_REGS; i++) begin
                    inuse_registers[w * `NUM_REGS + i] <= 0;                    
                end
                inuse_reg_mask[w] <= `NUM_REGS'(0);
            end            
        end else begin
            if (acquire_rd) begin
                inuse_registers[{decode_if.warp_num, decode_if.rd}] <= decode_if.thread_mask;
                inuse_reg_mask[decode_if.warp_num][decode_if.rd] <= 1;                
            end       
            if (writeback_fire) begin
                assert(inuse_reg_mask[writeback_if.warp_num][writeback_if.rd] != 0);
                inuse_registers[{writeback_if.warp_num, writeback_if.rd}] <= inuse_registers_n;
                inuse_reg_mask[writeback_if.warp_num][writeback_if.rd] <= (| inuse_registers_n);
            end            
        end
    end

    VX_cam_buffer #(
        .DATAW  ($bits(issue_data_t)),
        .SIZE   (`ISSUEQ_SIZE),
        .RPORTS (`NUM_EXS)
    ) issue_buffer (
        .clk          (clk),
        .reset        (reset),
        .write_data   ({decode_if.warp_num, decode_if.thread_mask, decode_if.curr_PC, decode_if.rd, decode_if.wb}),    
        .write_addr   (issue_tag),        
        .acquire_slot (issue_fire), 
        .release_slot ({cmt_to_issue_if.alu_valid, cmt_to_issue_if.lsu_valid, cmt_to_issue_if.csr_valid, cmt_to_issue_if.mul_valid, cmt_to_issue_if.fpu_valid, cmt_to_issue_if.gpu_valid}),           
        .read_addr    ({cmt_to_issue_if.alu_tag,   cmt_to_issue_if.lsu_tag,   cmt_to_issue_if.csr_tag,   cmt_to_issue_if.mul_tag,   cmt_to_issue_if.fpu_tag,   cmt_to_issue_if.gpu_tag}),
        .read_data    ({cmt_to_issue_if.alu_data,  cmt_to_issue_if.lsu_data,  cmt_to_issue_if.csr_data,  cmt_to_issue_if.mul_data,  cmt_to_issue_if.fpu_data,  cmt_to_issue_if.gpu_data}),
        .full         (issue_buf_full)
    );

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (decode_if.valid && ~decode_if.ready) begin
            $display("%t: Core%0d-stall: warp=%0d, PC=%0h, rd=%0d, wb=%0d, ib_full=%b, inuse=%b%b%b%b, ex_busy=%b",
                    $time, CORE_ID, decode_if.warp_num, decode_if.curr_PC, decode_if.rd, decode_if.wb, issue_buf_full,
                    inuse_mask[decode_if.rd], inuse_mask[decode_if.rs1], inuse_mask[decode_if.rs2], inuse_mask[decode_if.rs3], ex_busy);
        end
    end
`endif

endmodule