`include "VX_define.vh"

module VX_scoreboard  #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_decode_if        ibuf_deq_if,
    VX_writeback_if     writeback_if,  
    input wire [`NW_BITS-1:0] deq_wid_next,
    input wire          exe_delay,

    output wire         delay
);
    reg [`NUM_THREADS-1:0] inuse_registers [(`NUM_WARPS * `NUM_REGS)-1:0];  
    reg [`NUM_WARPS-1:0][`NUM_REGS-1:0] inuse_reg_mask, inuse_reg_mask_n;
    reg [`NUM_REGS-1:0] deq_used_regs;
    
    wire [`NUM_REGS-1:0] inuse_regs = deq_used_regs & ibuf_deq_if.used_regs;

    assign delay = (| inuse_regs);
    
    wire reserve_reg = ibuf_deq_if.valid && ibuf_deq_if.ready && (ibuf_deq_if.wb != 0);

    wire release_reg = writeback_if.valid && writeback_if.ready;

    wire [`NUM_THREADS-1:0] inuse_registers_n = inuse_registers[{writeback_if.wid, writeback_if.rd}] & ~writeback_if.tmask;

    always @(*) begin
        inuse_reg_mask_n = inuse_reg_mask;
        if (reserve_reg) begin
            inuse_reg_mask_n[ibuf_deq_if.wid][ibuf_deq_if.rd] = 1;                
        end       
        if (release_reg) begin
            inuse_reg_mask_n[writeback_if.wid][writeback_if.rd] = (| inuse_registers_n);
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            for (integer w = 0; w < `NUM_WARPS; w++) begin
                for (integer i = 0; i < `NUM_REGS; i++) begin
                    inuse_registers[w * `NUM_REGS + i] <= 0;                    
                end
                inuse_reg_mask[w] <= `NUM_REGS'(0);
            end            
        end else begin
            if (reserve_reg) begin
                inuse_registers[{ibuf_deq_if.wid, ibuf_deq_if.rd}] <= ibuf_deq_if.tmask;
            end       
            if (release_reg) begin
                assert(inuse_reg_mask[writeback_if.wid][writeback_if.rd] != 0) 
                    else $error("*** %t: core%0d: invalid writeback register: wid=%0d, PC=%0h, rd=%0d",
                        $time, CORE_ID, writeback_if.wid, writeback_if.PC, writeback_if.rd);            
                inuse_registers[{writeback_if.wid, writeback_if.rd}] <= inuse_registers_n;
            end            
            inuse_reg_mask <= inuse_reg_mask_n;
        end 

        deq_used_regs <= inuse_reg_mask_n[deq_wid_next];       
    end

    // issue the instruction
    assign ibuf_deq_if.ready = ~(delay || exe_delay);

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (ibuf_deq_if.valid && ~ibuf_deq_if.ready) begin            
            $display("%t: core%0d-stall: wid=%0d, PC=%0h, rd=%0d, wb=%0d, inuse=%b%b%b%b, exe=%b",
                    $time, CORE_ID, ibuf_deq_if.wid, ibuf_deq_if.PC, ibuf_deq_if.rd, ibuf_deq_if.wb, 
                    inuse_regs[ibuf_deq_if.rd], inuse_regs[ibuf_deq_if.rs1], inuse_regs[ibuf_deq_if.rs2], inuse_regs[ibuf_deq_if.rs3], exe_delay);            
        end
    end    
`endif

    reg [31:0] stall_ctr;
    always @(posedge clk) begin
        if (reset) begin
            stall_ctr <= 0;
        end else if (ibuf_deq_if.valid && ~ibuf_deq_if.ready) begin            
            stall_ctr <= stall_ctr + 1;
            assert(stall_ctr < 100000) else $error("*** %t: core%0d-stalled: wid=%0d, PC=%0h, rd=%0d, wb=%0d, inuse=%b%b%b%b, exe=%b",
                    $time, CORE_ID, ibuf_deq_if.wid, ibuf_deq_if.PC, ibuf_deq_if.rd, ibuf_deq_if.wb, 
                    inuse_regs[ibuf_deq_if.rd], inuse_regs[ibuf_deq_if.rs1], inuse_regs[ibuf_deq_if.rs2], inuse_regs[ibuf_deq_if.rs3], exe_delay);            
        end else if (ibuf_deq_if.valid && ibuf_deq_if.ready) begin
            stall_ctr <= 0;
        end
    end

endmodule