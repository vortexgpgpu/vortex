`include "VX_define.vh"

module VX_front_end #(
    parameter CORE_ID = 0
) (
    `SCOPE_SIGNALS_ISTAGE_IO

    input wire              clk,
    input wire              reset,

    input wire              schedule_delay,

    VX_warp_ctl_if          warp_ctl_if,

    VX_cache_core_rsp_if    icache_rsp_if,
    VX_cache_core_req_if    icache_req_if,

    VX_jal_rsp_if           jal_rsp_if,
    VX_branch_rsp_if        branch_rsp_if,

    VX_backend_req_if       bckE_req_if,
    output wire             busy
);

    VX_inst_meta_if        fe_inst_meta_fi();
    VX_inst_meta_if        fe_inst_meta_fi2();
    VX_inst_meta_if        fe_inst_meta_id();

    VX_backend_req_if      frE_to_bckE_req_if();
    VX_inst_meta_if        fd_inst_meta_de();

    wire total_freeze = schedule_delay;
    wire icache_stage_delay;

    wire[`NW_BITS-1:0] icache_stage_wid;
    wire               icache_stage_response;

    VX_wstall_if wstall_if();
    VX_join_if   join_if();

    VX_fetch fetch (
        .clk                (clk),
        .reset              (reset),
        .icache_stage_wid   (icache_stage_wid),
        .icache_stage_response(icache_stage_response),
        .wstall_if          (wstall_if),
        .join_if            (join_if),
        .schedule_delay     (schedule_delay),
        .jal_rsp_if         (jal_rsp_if),
        .warp_ctl_if        (warp_ctl_if),
        .icache_stage_delay (icache_stage_delay),
        .branch_rsp_if      (branch_rsp_if),
        .busy               (busy),
        .fe_inst_meta_fi    (fe_inst_meta_fi)
    );

    VX_generic_register #( 
        .N(64+`NW_BITS-1+1+`NUM_THREADS)
    ) f_d_reg (
        .clk   (clk),
        .reset (reset),
        .stall (icache_stage_delay),
        .flush (1'b0),
        .in    ({fe_inst_meta_fi.instruction, fe_inst_meta_fi.curr_PC, fe_inst_meta_fi.warp_num, fe_inst_meta_fi.valid}),
        .out   ({fe_inst_meta_fi2.instruction, fe_inst_meta_fi2.curr_PC, fe_inst_meta_fi2.warp_num, fe_inst_meta_fi2.valid})
    );

    VX_icache_stage #(
        .CORE_ID(CORE_ID)
    ) icache_stage (
        `SCOPE_SIGNALS_ISTAGE_BIND

        .clk                (clk),
        .reset              (reset),
        .total_freeze       (total_freeze),
        .icache_stage_delay (icache_stage_delay),
        .icache_stage_response(icache_stage_response),
        .icache_stage_wid   (icache_stage_wid),
        .fe_inst_meta_fi    (fe_inst_meta_fi2),
        .fe_inst_meta_id    (fe_inst_meta_id),
        .icache_rsp_if      (icache_rsp_if),
        .icache_req_if      (icache_req_if)
    );

    VX_generic_register #(
        .N(64 + `NW_BITS-1 + 1 + `NUM_THREADS)
    ) i_d_reg (
        .clk   (clk),
        .reset (reset),
        .stall (total_freeze),
        .flush (1'b0),
        .in    ({fe_inst_meta_id.instruction, fe_inst_meta_id.curr_PC, fe_inst_meta_id.warp_num, fe_inst_meta_id.valid}),
        .out   ({fd_inst_meta_de.instruction, fd_inst_meta_de.curr_PC, fd_inst_meta_de.warp_num, fd_inst_meta_de.valid})
    );

    VX_decode decode (
        .fd_inst_meta_de    (fd_inst_meta_de),
        .frE_to_bckE_req_if (frE_to_bckE_req_if),
        .wstall_if          (wstall_if),
        .join_if            (join_if)
    );    

    VX_generic_register #(
        .N(233 + `NW_BITS-1 + 1 + `NUM_THREADS)
    ) d_e_reg (
        .clk   (clk),
        .reset (reset),
        .stall (total_freeze),
        .flush (1'b0),
        .in   ({frE_to_bckE_req_if.csr_addr, frE_to_bckE_req_if.is_jal, frE_to_bckE_req_if.is_etype, frE_to_bckE_req_if.is_csr, frE_to_bckE_req_if.csr_immed, frE_to_bckE_req_if.csr_mask, frE_to_bckE_req_if.rd, frE_to_bckE_req_if.rs1, frE_to_bckE_req_if.rs2, frE_to_bckE_req_if.alu_op, frE_to_bckE_req_if.wb, frE_to_bckE_req_if.rs2_src, frE_to_bckE_req_if.itype_immed, frE_to_bckE_req_if.mem_read, frE_to_bckE_req_if.mem_write, frE_to_bckE_req_if.branch_type, frE_to_bckE_req_if.upper_immed, frE_to_bckE_req_if.curr_PC, frE_to_bckE_req_if.jal, frE_to_bckE_req_if.jal_offset, frE_to_bckE_req_if.next_PC, frE_to_bckE_req_if.valid, frE_to_bckE_req_if.warp_num, frE_to_bckE_req_if.is_wspawn, frE_to_bckE_req_if.is_tmc, frE_to_bckE_req_if.is_split, frE_to_bckE_req_if.is_barrier}),
        .out  ({bckE_req_if.csr_addr       , bckE_req_if.is_jal       , bckE_req_if.is_etype       ,bckE_req_if.is_csr       , bckE_req_if.csr_immed       , bckE_req_if.csr_mask       , bckE_req_if.rd       , bckE_req_if.rs1       , bckE_req_if.rs2       , bckE_req_if.alu_op       , bckE_req_if.wb       , bckE_req_if.rs2_src       , bckE_req_if.itype_immed       , bckE_req_if.mem_read       , bckE_req_if.mem_write       , bckE_req_if.branch_type       , bckE_req_if.upper_immed       , bckE_req_if.curr_PC       , bckE_req_if.jal       , bckE_req_if.jal_offset       , bckE_req_if.next_PC       , bckE_req_if.valid       , bckE_req_if.warp_num        , bckE_req_if.is_wspawn       , bckE_req_if.is_tmc       , bckE_req_if.is_split       , bckE_req_if.is_barrier       })
    );  

endmodule


