`include "VX_define.vh"

module VX_front_end #(
    parameter CORE_ID = 0
) (
    `SCOPE_SIGNALS_ISTAGE_IO

    input wire                clk,
    input wire                reset,

    input wire                schedule_delay,

    VX_warp_ctl_if            warp_ctl_if,

    VX_cache_core_rsp_if      icache_rsp_if,
    VX_cache_core_req_if      icache_req_if,

    VX_jal_rsp_if             jal_rsp_if,
    VX_branch_rsp_if          branch_rsp_if,

    VX_frE_to_bckE_req_if     bckE_req_if,
    output wire               busy
);

    VX_inst_meta_if        fe_inst_meta_fi();
    VX_inst_meta_if        fe_inst_meta_fi2();
    VX_inst_meta_if        fe_inst_meta_id();

    VX_frE_to_bckE_req_if  frE_to_bckE_req_if();
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

    VX_f_d_reg f_i_reg (
        .clk                (clk),
        .reset              (reset),
        .freeze             (icache_stage_delay),
        .fe_inst_meta_fd    (fe_inst_meta_fi),
        .fd_inst_meta_de    (fe_inst_meta_fi2)
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

    VX_i_d_reg i_d_reg (
        .clk                (clk),
        .reset              (reset),
        .freeze             (total_freeze),
        .fe_inst_meta_fd    (fe_inst_meta_id),
        .fd_inst_meta_de    (fd_inst_meta_de)
    );

    VX_decode decode (
        .fd_inst_meta_de    (fd_inst_meta_de),
        .frE_to_bckE_req_if (frE_to_bckE_req_if),
        .wstall_if          (wstall_if),
        .join_if            (join_if)
    );    

    wire no_br_stall = 0;

    VX_d_e_reg d_e_reg (
        .clk                (clk),
        .reset              (reset),
        .branch_stall       (no_br_stall),
        .freeze             (total_freeze),
        .frE_to_bckE_req_if (frE_to_bckE_req_if),
        .bckE_req_if        (bckE_req_if)
    );    

endmodule


