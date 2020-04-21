`include "VX_define.vh"

module VX_d_e_reg (
    input wire              clk,
    input wire              reset,
    input wire              branch_stall,
    input wire              freeze,
    VX_frE_to_bckE_req_if   frE_to_bckE_req_if,
    VX_frE_to_bckE_req_if   bckE_req_if
);

    wire stall = freeze;
    wire flush = (branch_stall == `STALL);

    VX_generic_register #(
        .N(233 + `NW_BITS-1 + 1 + `NUM_THREADS)
    ) d_e_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (flush),
        .in   ({frE_to_bckE_req_if.csr_address, frE_to_bckE_req_if.jalQual, frE_to_bckE_req_if.ebreak, frE_to_bckE_req_if.is_csr, frE_to_bckE_req_if.csr_immed, frE_to_bckE_req_if.csr_mask, frE_to_bckE_req_if.rd, frE_to_bckE_req_if.rs1, frE_to_bckE_req_if.rs2, frE_to_bckE_req_if.alu_op, frE_to_bckE_req_if.wb, frE_to_bckE_req_if.rs2_src, frE_to_bckE_req_if.itype_immed, frE_to_bckE_req_if.mem_read, frE_to_bckE_req_if.mem_write, frE_to_bckE_req_if.branch_type, frE_to_bckE_req_if.upper_immed, frE_to_bckE_req_if.curr_PC, frE_to_bckE_req_if.jal, frE_to_bckE_req_if.jal_offset, frE_to_bckE_req_if.PC_next, frE_to_bckE_req_if.valid, frE_to_bckE_req_if.warp_num, frE_to_bckE_req_if.is_wspawn, frE_to_bckE_req_if.is_tmc, frE_to_bckE_req_if.is_split, frE_to_bckE_req_if.is_barrier}),
        .out  ({bckE_req_if.csr_address       , bckE_req_if.jalQual       , bckE_req_if.ebreak       ,bckE_req_if.is_csr       , bckE_req_if.csr_immed       , bckE_req_if.csr_mask       , bckE_req_if.rd       , bckE_req_if.rs1       , bckE_req_if.rs2       , bckE_req_if.alu_op       , bckE_req_if.wb       , bckE_req_if.rs2_src       , bckE_req_if.itype_immed       , bckE_req_if.mem_read       , bckE_req_if.mem_write       , bckE_req_if.branch_type       , bckE_req_if.upper_immed       , bckE_req_if.curr_PC       , bckE_req_if.jal       , bckE_req_if.jal_offset       , bckE_req_if.PC_next       , bckE_req_if.valid       , bckE_req_if.warp_num        , bckE_req_if.is_wspawn       , bckE_req_if.is_tmc       , bckE_req_if.is_split       , bckE_req_if.is_barrier       })
    );

endmodule




