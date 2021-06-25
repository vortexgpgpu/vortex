`include "VX_define.vh"

module VX_scoreboard  #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    VX_instr_sched_if instr_sched_if,
    VX_writeback_if writeback_if,
    output wire     delay
);
    reg [`NUM_WARPS-1:0][`NUM_REGS-1:0] inuse_regs, inuse_regs_n;

    reg [`NUM_REGS-1:0] deq_inuse_regs;

    assign delay = |(deq_inuse_regs & instr_sched_if.used_regs);

    wire reserve_reg = instr_sched_if.valid && instr_sched_if.ready && instr_sched_if.wb;

    wire release_reg = writeback_if.valid && writeback_if.ready && writeback_if.eop;
    
    always @(*) begin
        inuse_regs_n = inuse_regs;
        if (reserve_reg) begin
            inuse_regs_n[instr_sched_if.wid][instr_sched_if.rd] = 1;                
        end       
        if (release_reg) begin
            inuse_regs_n[writeback_if.wid][writeback_if.rd] = 0;
        end    
    end

    always @(posedge clk) begin
        if (reset) begin
            inuse_regs <= '0;
        end else begin
            inuse_regs <= inuse_regs_n;
        end
        deq_inuse_regs <= inuse_regs_n[instr_sched_if.wid_n];
    end

    reg [31:0] deadlock_ctr;
    wire [31:0] deadlock_timeout = 1000 * (10 ** (`L2_ENABLE + `L3_ENABLE));
    always @(posedge clk) begin
        if (reset) begin
            deadlock_ctr <= 0;
        end else begin
        `ifdef DBG_PRINT_PIPELINE
            if (instr_sched_if.valid && ~instr_sched_if.ready) begin            
                $display("%t: *** core%0d-stall: wid=%0d, PC=%0h, rd=%0d, wb=%0d, inuse=%b%b%b%b",
                        $time, CORE_ID, instr_sched_if.wid, instr_sched_if.PC, instr_sched_if.rd, instr_sched_if.wb, 
                        deq_inuse_regs[instr_sched_if.rd], deq_inuse_regs[instr_sched_if.rs1], deq_inuse_regs[instr_sched_if.rs2], deq_inuse_regs[instr_sched_if.rs3]);            
            end
        `endif
            if (release_reg) begin
                assert(inuse_regs[writeback_if.wid][writeback_if.rd] != 0) 
                    else $error("%t: *** core%0d: invalid writeback register: wid=%0d, PC=%0h, rd=%0d",
                                $time, CORE_ID, writeback_if.wid, writeback_if.PC, writeback_if.rd);
            end
            if (instr_sched_if.valid && ~instr_sched_if.ready) begin            
                deadlock_ctr <= deadlock_ctr + 1;
                assert(deadlock_ctr < deadlock_timeout) else $error("%t: *** core%0d-deadlock: wid=%0d, PC=%0h, rd=%0d, wb=%0d, inuse=%b%b%b%b",
                        $time, CORE_ID, instr_sched_if.wid, instr_sched_if.PC, instr_sched_if.rd, instr_sched_if.wb, 
                        deq_inuse_regs[instr_sched_if.rd], deq_inuse_regs[instr_sched_if.rs1], deq_inuse_regs[instr_sched_if.rs2], deq_inuse_regs[instr_sched_if.rs3]);            
            end else if (instr_sched_if.valid && instr_sched_if.ready) begin
                deadlock_ctr <= 0;
            end
        end
    end

endmodule