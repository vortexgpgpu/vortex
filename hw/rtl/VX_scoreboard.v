`include "VX_define.vh"

module VX_scoreboard  #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    VX_decode_if    ibuf_deq_if,
    VX_writeback_if writeback_if,
    output wire     delay
);
    reg [`NUM_WARPS-1:0][`NUM_REGS-1:0] inuse_regs;
    wire [`NUM_REGS-1:0] deq_inuse_regs;
    
    assign deq_inuse_regs = inuse_regs[ibuf_deq_if.wid] & ibuf_deq_if.used_regs;

    assign delay = (| deq_inuse_regs);
    
    wire reserve_reg = ibuf_deq_if.valid && ibuf_deq_if.ready && (ibuf_deq_if.wb != 0);

    wire release_reg = writeback_if.valid && writeback_if.ready && writeback_if.eop;

    always @(posedge clk) begin
        if (reset) begin
            inuse_regs <= (`NUM_WARPS*`NUM_REGS)'(0);
        end else begin
            if (reserve_reg) begin
                inuse_regs[ibuf_deq_if.wid][ibuf_deq_if.rd] <= 1;                
            end       
            if (release_reg) begin
                inuse_regs[writeback_if.wid][writeback_if.rd] <= 0;
                assert(inuse_regs[writeback_if.wid][writeback_if.rd] != 0) 
                    else $error("%t: *** core%0d: invalid writeback register: wid=%0d, PC=%0h, rd=%0d",
                                $time, CORE_ID, writeback_if.wid, writeback_if.PC, writeback_if.rd);
            end
        end    
    end

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (ibuf_deq_if.valid && ~ibuf_deq_if.ready) begin            
            $display("%t: *** core%0d-stall: wid=%0d, PC=%0h, rd=%0d, wb=%0d, inuse=%b%b%b%b",
                    $time, CORE_ID, ibuf_deq_if.wid, ibuf_deq_if.PC, ibuf_deq_if.rd, ibuf_deq_if.wb, 
                    deq_inuse_regs[ibuf_deq_if.rd], deq_inuse_regs[ibuf_deq_if.rs1], deq_inuse_regs[ibuf_deq_if.rs2], deq_inuse_regs[ibuf_deq_if.rs3]);            
        end
    end    
`endif

    reg [31:0] deadlock_ctr;
    wire [31:0] deadlock_timeout = 1000 * (10 ** (`L2_ENABLE + `L3_ENABLE));
    always @(posedge clk) begin
        if (reset) begin
            deadlock_ctr <= 0;
        end else if (ibuf_deq_if.valid && ~ibuf_deq_if.ready) begin            
            deadlock_ctr <= deadlock_ctr + 1;
            assert(deadlock_ctr < deadlock_timeout) else $error("%t: *** core%0d-deadlock: wid=%0d, PC=%0h, rd=%0d, wb=%0d, inuse=%b%b%b%b",
                    $time, CORE_ID, ibuf_deq_if.wid, ibuf_deq_if.PC, ibuf_deq_if.rd, ibuf_deq_if.wb, 
                    deq_inuse_regs[ibuf_deq_if.rd], deq_inuse_regs[ibuf_deq_if.rs1], deq_inuse_regs[ibuf_deq_if.rs2], deq_inuse_regs[ibuf_deq_if.rs3]);            
        end else if (ibuf_deq_if.valid && ibuf_deq_if.ready) begin
            deadlock_ctr <= 0;
        end
    end

endmodule