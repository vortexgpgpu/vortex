`include "VX_define.vh"

module VX_scoreboard  #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    VX_writeback_if.slave   writeback_if,
    VX_scoreboard_if.slave  scoreboard_if,
    VX_ibuffer_if.scoreboard ibuffer_if
);
    `UNUSED_PARAM (CORE_ID)

    reg [`NUM_WARPS-1:0][`NUM_REGS-1:0] inuse_regs, inuse_regs_n;

    wire reserve_reg = ibuffer_if.valid && ibuffer_if.ready && ibuffer_if.wb;
    wire release_reg = writeback_if.valid && writeback_if.eop;
    
    always @(*) begin
        inuse_regs_n = inuse_regs;
        if (reserve_reg) begin
            inuse_regs_n[ibuffer_if.wid][ibuffer_if.rd] = 1; 
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
    end

    wire used_rd  = inuse_regs_n[scoreboard_if.wid][scoreboard_if.rd];
    wire used_rs1 = inuse_regs_n[scoreboard_if.wid][scoreboard_if.rs1];
    wire used_rs2 = inuse_regs_n[scoreboard_if.wid][scoreboard_if.rs2];
    wire used_rs3 = inuse_regs_n[scoreboard_if.wid][scoreboard_if.rs3];

    assign scoreboard_if.ready = ~(used_rd | used_rs1 | used_rs2 | used_rs3);

    reg [31:0] timeout_ctr;
    
    always @(posedge clk) begin
        if (reset) begin
            timeout_ctr <= '0;
        end else begin        
            if (scoreboard_if.valid && ~scoreboard_if.ready) begin
            `ifdef DBG_TRACE_CORE_PIPELINE
                `TRACE(3, ("%d: *** core%0d-stall: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b%b%b%b (#%0d)\n",
                    $time, CORE_ID, scoreboard_if.wid, scoreboard_if.PC, scoreboard_if.tmask, timeout_ctr,
                    used_rd, used_rs1, used_rs2, used_rs3, scoreboard_if.uuid));
            `endif
                timeout_ctr <= timeout_ctr + 1;
            end else if (scoreboard_if.valid && scoreboard_if.ready) begin
                timeout_ctr <= '0;
            end
        end
    end

    `RUNTIME_ASSERT((timeout_ctr < `STALL_TIMEOUT),
                    ("%t: *** core%0d-issue-timeout: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b%b%b%b (#%0d)",
                        $time, CORE_ID, scoreboard_if.wid, scoreboard_if.PC, scoreboard_if.tmask, timeout_ctr,
                        used_rd, used_rs1, used_rs2, used_rs3, scoreboard_if.uuid));

    always @(posedge clk) begin
        if (release_reg) begin
            `ASSERT(inuse_regs[writeback_if.wid][writeback_if.rd] != 0,
                ("%t: *** core%0d: invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
                            $time, CORE_ID, writeback_if.wid, writeback_if.PC, writeback_if.tmask, writeback_if.rd, writeback_if.uuid));
        end
    end

endmodule
