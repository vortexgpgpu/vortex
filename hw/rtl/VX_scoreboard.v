`include "VX_define.vh"

module VX_scoreboard  #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_decode_if        ibuf_deq_if,
    VX_writeback_if     writeback_if,  
    input wire          exe_delay,
    input wire          gpr_delay,

    output wire         delay
);
    reg [`NUM_THREADS-1:0] inuse_registers [(`NUM_WARPS * `NUM_REGS)-1:0];  
    reg [`NUM_REGS-1:0] inuse_reg_mask  [`NUM_WARPS-1:0];
    
    wire [`NUM_REGS-1:0] inuse_mask = inuse_reg_mask[ibuf_deq_if.wid] & ibuf_deq_if.used_regs;

    assign delay = (| inuse_mask);
    
    wire reserve_reg = ibuf_deq_if.valid && ibuf_deq_if.ready && (ibuf_deq_if.wb != 0);

    wire release_reg = writeback_if.valid && writeback_if.ready;

    wire [`NUM_THREADS-1:0] inuse_registers_n = inuse_registers[{writeback_if.wid, writeback_if.rd}] & ~writeback_if.thread_mask;

    always @(posedge clk) begin
        if (reset) begin
            for (integer w = 0; w < `NUM_WARPS; w++) begin
                for (integer i = 0; i < `NUM_REGS; i++) begin
                    inuse_registers[w * `NUM_REGS + i] <= 0;                    
                end
                inuse_reg_mask [w] <= `NUM_REGS'(0);
            end            
        end else begin
            if (reserve_reg) begin
                inuse_registers[{ibuf_deq_if.wid, ibuf_deq_if.rd}] <= ibuf_deq_if.thread_mask;
                inuse_reg_mask[ibuf_deq_if.wid][ibuf_deq_if.rd] <= 1;                
            end       
            if (release_reg) begin
                assert(inuse_reg_mask[writeback_if.wid][writeback_if.rd] != 0);
                inuse_registers[{writeback_if.wid, writeback_if.rd}] <= inuse_registers_n;
                inuse_reg_mask[writeback_if.wid][writeback_if.rd] <= (| inuse_registers_n);
            end            
        end
    end

    // issue the instruction
    assign ibuf_deq_if.ready = ~(delay || exe_delay || gpr_delay);

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (ibuf_deq_if.valid && ~ibuf_deq_if.ready) begin
            $display("%t: core%0d-stall: wid=%0d, PC=%0h, rd=%0d, wb=%0d, inuse=%b%b%b%b, exe=%b, gpr=%b",
                    $time, CORE_ID, ibuf_deq_if.wid, ibuf_deq_if.curr_PC, ibuf_deq_if.rd, ibuf_deq_if.wb, 
                    inuse_mask[ibuf_deq_if.rd], inuse_mask[ibuf_deq_if.rs1], inuse_mask[ibuf_deq_if.rs2], inuse_mask[ibuf_deq_if.rs3], exe_delay, gpr_delay);
        end
    end
`endif

endmodule