`include "VX_define.vh"

module VX_scoreboard  #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    VX_scoreboard_if.slave  scoreboard_if,
    VX_writeback_if.slave   writeback_if
);
    reg [`NUM_WARPS-1:0][`NUM_REGS-1:0] inuse_regs, inuse_regs_n;

    wire reserve_reg = scoreboard_if.valid && scoreboard_if.ready && scoreboard_if.wb;
    wire release_reg = writeback_if.valid && writeback_if.ready && writeback_if.eop;
    
    always @(*) begin
        inuse_regs_n = inuse_regs;
        if (reserve_reg) begin
            inuse_regs_n[scoreboard_if.wid][scoreboard_if.rd] = 1; 
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
    
    reg deq_inuse_rd, deq_inuse_rs1, deq_inuse_rs2, deq_inuse_rs3;

    always @(posedge clk) begin
        deq_inuse_rd  <= inuse_regs_n[scoreboard_if.wid_n][scoreboard_if.rd_n];
        deq_inuse_rs1 <= inuse_regs_n[scoreboard_if.wid_n][scoreboard_if.rs1_n];
        deq_inuse_rs2 <= inuse_regs_n[scoreboard_if.wid_n][scoreboard_if.rs2_n];
        deq_inuse_rs3 <= inuse_regs_n[scoreboard_if.wid_n][scoreboard_if.rs3_n];
    end

    assign writeback_if.ready = 1'b1;

    assign scoreboard_if.ready = ~(deq_inuse_rd 
                                 | deq_inuse_rs1 
                                 | deq_inuse_rs2 
                                 | deq_inuse_rs3);

    assign scoreboard_if.used_regs[0] = deq_inuse_rd;
    assign scoreboard_if.used_regs[1] = deq_inuse_rs1;
    assign scoreboard_if.used_regs[2] = deq_inuse_rs2;
    assign scoreboard_if.used_regs[3] = deq_inuse_rs3;

    `UNUSED_VAR (writeback_if.PC)
    `UNUSED_VAR (scoreboard_if.PC)
    `UNUSED_VAR (scoreboard_if.tmask)    
    `UNUSED_VAR (scoreboard_if.uuid)

    always @(posedge clk) begin  
        if (release_reg) begin
            `ASSERT(inuse_regs[writeback_if.wid][writeback_if.rd] != 0,
                ("%t: *** core%0d: invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
                            $time, CORE_ID, writeback_if.wid, writeback_if.PC, writeback_if.tmask, writeback_if.rd, writeback_if.uuid));
        end
    end

endmodule
