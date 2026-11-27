// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

// The warp scheduler of one frontend slice: the state of the slice's warps
// (wid = {wfs, slice_id}), its warp selection and a registered fetch request to
// the slice's fetch unit, which names warps by wfs. FU block port q carries the
// traffic of the local warps with wfs mod PORTS == q, addressed by core-wide wid;
// core-wide events arrive through sched_if already localized.
module VX_warp_sched import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,
    input wire [FETCH_FSW_W-1:0] slice_id,

    VX_slice_sched_if.slave sched_if,

    // ALU ports
    input sched_branch_t [FETCH_ALU_PORTS-1:0] branch,

    // SFU ports
    input sched_wctl_t [FETCH_SFU_PORTS-1:0] wctl,
    input wire [FETCH_SFU_PORTS-1:0][NW_WIDTH-1:0] dvstack_wid,
    output wire [FETCH_SFU_PORTS-1:0][DV_STACK_SIZEW-1:0] dvstack_ptr,
    output wire [FETCH_SFU_PORTS-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] dvstack_mscratch,
    input wire [FETCH_SFU_PORTS-1:0][NW_WIDTH-1:0] csr_rd_wid,
    output wire sched_csr_rd_t [FETCH_SFU_PORTS-1:0] csr_rd,
    input sched_csr_wr_t [FETCH_SFU_PORTS-1:0] csr_wr,
`ifdef VX_CFG_EXT_RTU_ENABLE
    input wire [FETCH_SFU_PORTS-1:0] rtu_unlock_valid,
    input wire [FETCH_SFU_PORTS-1:0][NW_WIDTH-1:0] rtu_unlock_wid,
`endif

    // per-warp pipeline events, indexed by wfs
    VX_decode_sched_if.slave decode_sched_if,
    input wire [PER_FETCH_WARPS-1:0] issued,
    input wire [PER_FETCH_WARPS-1:0] committed,
    input wire [PER_FETCH_WARPS-1:0] ibuf_pop,
    output wire [PER_FETCH_WARPS-1:0] pending_alm_empty,

    // fetch request
    VX_schedule_if.master   schedule_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam SW      = PER_FETCH_WARPS;
    localparam LW      = FETCH_WFS_W;
    localparam NPA     = FETCH_ALU_PORTS;
    localparam NPS     = FETCH_SFU_PORTS;
    localparam NPS_BITS = `CLOG2(NPS);
    localparam PSW     = SW / NPS;           // warps per SFU port
    localparam KW      = `LOG2UP(PSW);

    `UNUSED_VAR (schedule_if.ibuf_pop)

    // wfs -> SFU port-local index
    function automatic logic [KW-1:0] wfs_to_k(input logic [LW-1:0] wfs);
        wfs_to_k = KW'(wfs >> NPS_BITS);
    endfunction

    wire [LW-1:0] dec_lid = decode_sched_if.wid[LW-1:0];
    if (NW_WIDTH > LW) begin : g_dec_wid_unused
        `UNUSED_VAR (decode_sched_if.wid[NW_WIDTH-1:LW])
    end

    wire [NPA-1:0][LW-1:0] br_lid;
    for (genvar q = 0; q < NPA; ++q) begin : g_br_lid
        assign br_lid[q] = wid_to_wfs(branch[q].wid);
        `UNUSED_VAR (wid_to_fsw(branch[q].wid))
    end

    wire [NPS-1:0][LW-1:0] ctl_lid, csr_rd_lid, csr_wr_lid;
`ifdef VX_CFG_EXT_RTU_ENABLE
    wire [NPS-1:0][LW-1:0] rtu_lid;
`endif

    // split/join, one stack per SFU port over the port's warps
    wire [NPS-1:0]                join_valid;
    wire [NPS-1:0]                join_is_dvg;
    wire [NPS-1:0]                join_is_else;
    wire [NPS-1:0][KW-1:0]        join_k;
    wire [NPS-1:0][`VX_CFG_NUM_THREADS-1:0] join_tmask;
    wire [NPS-1:0][PC_BITS-1:0]   join_pc;

    for (genvar q = 0; q < NPS; ++q) begin : g_sfu_ports
        assign ctl_lid[q]    = wid_to_wfs(wctl[q].wid);
        assign csr_rd_lid[q] = wid_to_wfs(csr_rd_wid[q]);
        assign csr_wr_lid[q] = wid_to_wfs(csr_wr[q].wid);
        `UNUSED_VAR ({wid_to_fsw(wctl[q].wid), wid_to_fsw(csr_rd_wid[q]), wid_to_fsw(csr_wr[q].wid), wid_to_fsw(dvstack_wid[q])})
    `ifdef VX_CFG_EXT_RTU_ENABLE
        assign rtu_lid[q]    = wid_to_wfs(rtu_unlock_wid[q]);
        `UNUSED_VAR (wid_to_fsw(rtu_unlock_wid[q]))
    `endif

        VX_split_join #(
            .INSTANCE_ID ((NPS > 1) ? `SFORMATF(("%s-splitjoin%0d", INSTANCE_ID, q)) : `SFORMATF(("%s-splitjoin", INSTANCE_ID))),
            .NUM_WARPS   (PSW),
            .WID_W       (KW),
            .OUT_REG     (1)
        ) split_join (
            .clk         (clk),
            .reset       (reset),
            .split_valid (wctl[q].split_valid),
            .sjoin_valid (wctl[q].sjoin_valid),
            .wid         (wfs_to_k(ctl_lid[q])),
            .split       (wctl[q].split),
            .sjoin       (wctl[q].sjoin),
            .join_valid  (join_valid[q]),
            .join_is_dvg (join_is_dvg[q]),
            .join_is_else(join_is_else[q]),
            .join_wid    (join_k[q]),
            .join_tmask  (join_tmask[q]),
            .join_pc     (join_pc[q]),
            .stack_wid   (wfs_to_k(wid_to_wfs(dvstack_wid[q]))),
            .stack_ptr   (dvstack_ptr[q])
        );
    end

    wire [SW-1:0] active_warps, stalled_warps;
    wire [SW-1:0][`VX_CFG_NUM_THREADS-1:0] thread_masks;
    wire [SW-1:0][PC_BITS-1:0] warp_pcs;
    wire [SW-1:0] ibuf_full;
    wire [SW-1:0] pending_empty;
    wire [SW-1:0] cand_onehot;
    wire          cand_valid;
    wire          sched_fire;

    wire [SW-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] mscratch;
    wire [SW-1:0][`VX_CFG_XLEN-1:0] mstatus, mepc;
    wire [SW-1:0][TRAP_CAUSE_BITS-1:0] mcause;

    for (genvar l = 0; l < SW; ++l) begin : g_warps
        localparam [LW-1:0] LID = LW'(l);
        localparam QA = l % NPA;
        localparam QS = l % NPS;
        localparam [KW-1:0] K = KW'(l / NPS);

        VX_warp_sched_ctx #(
            .INSTANCE_ID (INSTANCE_ID)
        ) warp_ctx (
            .clk               (clk),
            .reset             (reset),
            .lid               (LID),
            .slice_id          (slice_id),
            .cta_hit           (sched_if.cta_fire && (sched_if.cta_lid == LID)),
            .cta_init          (sched_if.cta_init),
            .cta_PC            (sched_if.cta_PC),
            .cta_tmask         (sched_if.cta_tmask),
            .cta_param         (sched_if.cta_param),
            .wspawn_apply      (sched_if.wspawn_apply),
            .wspawn_sel        (sched_if.wspawn_wmask[l]),
            .wspawn_src_hit    (sched_if.wspawn_src && (sched_if.wspawn_src_lid == LID)),
            .wspawn_pc         (sched_if.wspawn_pc),
            .wspawn_mscratch   (sched_if.wspawn_mscratch),
            .bar_unlock_valid  (sched_if.bar_unlock_valid),
            .bar_unlock_sel    (sched_if.bar_unlock_mask[l]),
            .retire_hit        (sched_if.retire_valid && (sched_if.retire_lid == LID)),
            .mtvec             (sched_if.mtvec),
            .dec_hit           (decode_sched_if.valid && (dec_lid == LID)),
            .dec_unlock        (decode_sched_if.unlock),
            .dec_fs_dirty      (decode_sched_if.fs_dirty),
        `ifdef VX_CFG_EXT_C_ENABLE
            .dec_is_rvc        (decode_sched_if.is_rvc),
        `endif
            .branch            (branch[QA]),
            .br_hit            (branch[QA].valid && (br_lid[QA] == LID)),
            .wctl              (wctl[QS]),
            .ctl_hit           (ctl_lid[QS] == LID),
            .csr_wr            (csr_wr[QS]),
            .csr_wr_hit        (csr_wr_lid[QS] == LID),
        `ifdef VX_CFG_EXT_RTU_ENABLE
            .rtu_unlock_hit    (rtu_unlock_valid[QS] && (rtu_lid[QS] == LID)),
        `endif
            .join_hit          (join_valid[QS] && (join_k[QS] == K)),
            .join_is_dvg       (join_is_dvg[QS]),
            .join_is_else      (join_is_else[QS]),
            .join_tmask        (join_tmask[QS]),
            .join_pc           (join_pc[QS]),
            .sched_fire        (sched_fire),
            .sched_sel         (cand_onehot[l]),
            .ibuf_pop          (ibuf_pop[l]),
            .issued            (issued[l]),
            .committed         (committed[l]),
            .active            (active_warps[l]),
            .stalled           (stalled_warps[l]),
            .tmask             (thread_masks[l]),
            .pc                (warp_pcs[l]),
            .ibuf_full         (ibuf_full[l]),
            .pending_empty     (pending_empty[l]),
            .pending_alm_empty (pending_alm_empty[l]),
            .mscratch          (mscratch[l]),
            .mstatus           (mstatus[l]),
            .mepc              (mepc[l]),
            .mcause            (mcause[l]),
            .fs_off            (decode_sched_if.fs_off[l])
        );
    end

    // the slice's lowest ready warp
    wire [SW-1:0] ready_warps     = active_warps & ~stalled_warps;
    wire [SW-1:0] preferred_warps = ready_warps & ~ibuf_full;
`ifndef L1_ENABLE
    // without L1, the icache must never stall: it shares the bus with dcache responses
    wire [SW-1:0] full_warps     = preferred_warps;
    wire [SW-1:0] schedule_warps = preferred_warps;
    `UNUSED_VAR (sched_if.slice_ibuf_full)
`else
    // once every ibuffer of the core is full, any ready warp may fetch
    wire [SW-1:0] full_warps     = ready_warps;
    wire [SW-1:0] schedule_warps = (& sched_if.slice_ibuf_full) ? full_warps : preferred_warps;
`endif

    wire [`UP(FETCH_WFS_BITS)-1:0] cand_index;
    VX_priority_encoder #(
        .N (SW)
    ) wid_select (
        .data_in   (schedule_warps),
        .index_out (cand_index),
        .valid_out (cand_valid),
        .onehot_out(cand_onehot)
    );
    wire [LW-1:0] cand_lid = LW'(cand_index);

    // CTA CSRs and CTA ids, one store per SFU port over the port's warps
    wire [NPS-1:0][NCTA_WIDTH-1:0] port_cta_id;

    for (genvar q = 0; q < NPS; ++q) begin : g_csr_cta
        wire [LW-1:0] warp_waddr = sched_if.warp_waddr;
        wire [LW-1:0] id_waddr   = sched_if.id_waddr;

        VX_csr_cta #(
            .NUM_WARPS (PSW)
        ) csr_cta (
            .clk          (clk),
            .reset        (reset),
            .warp_write   (sched_if.warp_write && ((32'(warp_waddr) % NPS) == q)),
            .warp_waddr   (wfs_to_k(warp_waddr)),
            .warp_wdata   (sched_if.warp_wdata),
            .id_write     (sched_if.id_write && ((32'(id_waddr) % NPS) == q)),
            .id_waddr     (wfs_to_k(id_waddr)),
            .id_wdata     (sched_if.id_wdata),
            .rd_lid       (wfs_to_k(csr_rd_lid[q])),
            .rd_cta_rank  (csr_rd[q].cta_rank),
            .rd_lane      (csr_rd[q].cta_lane),
            .sched_lid    (wfs_to_k(cand_lid)),
            .sched_cta_id (port_cta_id[q])
        );

        assign csr_rd[q].thread_mask = thread_masks[csr_rd_lid[q]];
        assign csr_rd[q].mscratch    = mscratch[csr_rd_lid[q]];
        assign csr_rd[q].mstatus     = mstatus[csr_rd_lid[q]];
        assign csr_rd[q].mepc        = mepc[csr_rd_lid[q]];
        assign csr_rd[q].mcause      = `VX_CFG_XLEN'(mcause[csr_rd_lid[q]]);

        assign dvstack_mscratch[q]   = mscratch[wid_to_wfs(dvstack_wid[q])];
    end

    wire [NCTA_WIDTH-1:0] cand_cta_id = port_cta_id[32'(cand_lid) % NPS];

    // fetch request, registered at the slice boundary
    wire sched_ready;
    assign sched_fire = cand_valid && sched_ready;

    wire [UUID_WIDTH-1:0] sched_uuid;
`ifdef UUID_ENABLE
    VX_uuid_gen #(
        .NUM_WARPS (SW)
    ) uuid_gen (
        .clk     (clk),
        .reset   (reset),
        .core_id (core_id),
        .incr    (sched_fire),
        .wid     (wfs_to_wid(cand_lid, slice_id)),
        .uuid    (sched_uuid)
    );
`else
    assign sched_uuid = '0;
    `UNUSED_VAR (core_id)
`endif

    VX_elastic_buffer #(
        .DATAW   (`VX_CFG_NUM_THREADS + PC_BITS + NW_WIDTH + NCTA_WIDTH + UUID_WIDTH),
        .SIZE    (2),
        .OUT_REG (1)
    ) sched_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (cand_valid),
        .ready_in  (sched_ready),
        .data_in   ({thread_masks[cand_lid], warp_pcs[cand_lid], NW_WIDTH'(cand_lid), cand_cta_id, sched_uuid}),
        .data_out  ({schedule_if.data.tmask, schedule_if.data.PC, schedule_if.data.wid, schedule_if.data.cta_id, schedule_if.data.uuid}),
        .valid_out (schedule_if.valid),
        .ready_out (schedule_if.ready)
    );

    wire schedule_if_fire = schedule_if.valid && schedule_if.ready;

    // status
    assign sched_if.active        = active_warps;
    assign sched_if.stalled       = stalled_warps;
    assign sched_if.ibuf_full     = (& ibuf_full);
    assign sched_if.pending_empty = pending_empty;

    // performance events
    for (genvar q = 0; q < NPA; ++q) begin : g_branch_valid
        assign sched_if.branch_valid[q] = branch[q].valid;
    end
    for (genvar q = 0; q < NPS; ++q) begin : g_divergence
        assign sched_if.divergence[q] = wctl[q].split_valid && wctl[q].split.is_dvg;
    end
    assign sched_if.sched_valid_room = (| preferred_warps);
    assign sched_if.sched_valid_full = (| full_warps);
    assign sched_if.sched_fire    = schedule_if_fire;
    assign sched_if.sched_tmask   = schedule_if.data.tmask;

`ifdef EXT_SCHED_TIMEOUT_DUMP
    always @(posedge clk) begin
        if (sched_if.timeout_dump) begin
            for (integer l = 0; l < SW; ++l) begin
                $display("    wid=%0d stalled=%0d pc=0x%0h tmask=%b",
                         wfs_to_wid(LW'(l), slice_id), stalled_warps[l], to_fullPC(warp_pcs[l]), thread_masks[l]);
            end
        end
    end
`endif

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (sched_fire) begin
            `TRACE(1, ("%t: %s dispatch: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b (#%0d)\n", $time, INSTANCE_ID,
                wfs_to_wid(cand_lid, slice_id), cand_cta_id, to_fullPC(warp_pcs[cand_lid]), thread_masks[cand_lid], sched_uuid))
        end
    end
`endif

endmodule
