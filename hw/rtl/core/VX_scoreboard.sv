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

    localparam NW_WIDTH = `UP(`NW_BITS);

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

    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        assign scoreboard_if.ready[i] = ~(inuse_regs_n[i][scoreboard_if.rd[i]]
                                        | inuse_regs_n[i][scoreboard_if.rs1[i]]
                                        | inuse_regs_n[i][scoreboard_if.rs2[i]]
                                        | inuse_regs_n[i][scoreboard_if.rs3[i]]);
    end

    reg [31:0] timeout_ctr;
    wire [NW_WIDTH-1:0] wid_sel;

    VX_lzc #(
        .N       (`NUM_WARPS),
        .REVERSE (1)
    ) wid_select (
        .data_in  (scoreboard_if.valid),
        .data_out (wid_sel),
        `UNUSED_PIN (valid_out)
    );    

    wire used_rd  = inuse_regs_n[wid_sel][scoreboard_if.rd[wid_sel]];
    wire used_rs1 = inuse_regs_n[wid_sel][scoreboard_if.rs1[wid_sel]];
    wire used_rs2 = inuse_regs_n[wid_sel][scoreboard_if.rs2[wid_sel]];
    wire used_rs3 = inuse_regs_n[wid_sel][scoreboard_if.rs3[wid_sel]];

    wire [`NUM_WARPS-1:0] scoreboard_fire = scoreboard_if.valid & scoreboard_if.ready;
    
    always @(posedge clk) begin
        if (reset) begin
            timeout_ctr <= '0;
        end else begin        
            if (scoreboard_if.valid != 0 && scoreboard_fire == 0) begin
            `ifdef DBG_TRACE_CORE_PIPELINE
                `TRACE(3, ("%d: *** core%0d-stall: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b%b%b%b (#%0d)\n",
                    $time, CORE_ID, wid_sel, scoreboard_if.PC[wid_sel], scoreboard_if.tmask[wid_sel], timeout_ctr,
                    used_rd, used_rs1, used_rs2, used_rs3, scoreboard_if.uuid[wid_sel]));
            `endif
                timeout_ctr <= timeout_ctr + 1;
            end else if (scoreboard_fire != 0) begin
                timeout_ctr <= '0;
            end
        end
    end

    `RUNTIME_ASSERT((timeout_ctr < `STALL_TIMEOUT),
                    ("%t: *** core%0d-issue-timeout: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b%b%b%b (#%0d)",
                        $time, CORE_ID, wid_sel, scoreboard_if.PC[wid_sel], scoreboard_if.tmask[wid_sel], timeout_ctr,
                        used_rd, used_rs1, used_rs2, used_rs3, scoreboard_if.uuid[wid_sel]));

    always @(posedge clk) begin
        if (release_reg) begin
            `ASSERT(inuse_regs[writeback_if.wid][writeback_if.rd] != 0,
                ("%t: *** core%0d: invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
                            $time, CORE_ID, writeback_if.wid, writeback_if.PC, writeback_if.tmask, writeback_if.rd, writeback_if.uuid));
        end
    end

endmodule
